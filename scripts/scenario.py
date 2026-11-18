#!/usr/bin/env python3
"""Resolve composable benchmark scenarios into vLLM configs and shell exports."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from copy import deepcopy
from pathlib import Path

from typing import Any

from script_utils import atomic_json, atomic_text, canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "config" / "scenarios"
PLATFORM_DIR = ROOT / "config" / "platforms"


# Recursively merge scenario mappings without mutating either input.
def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


# Load one scenario file and require a top-level JSON object.
def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"scenario must contain a JSON object: {path}")
    return value


# Format one primitive value as a valid YAML scalar.
def yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(character in text for character in ":#\n'\"{}[],&*?|>%@`"):
        return json.dumps(text)
    if text != text.strip() or text[0] in "-? ":
        return json.dumps(text)
    return text


# Serialize a scenario mapping into the supported YAML subset.
def to_yaml(value: Any, indent: int = 0) -> str:
    if not isinstance(value, dict):
        raise ValueError("vLLM config must be a mapping")
    padding = "  " * indent
    lines: list[str] = []
    for key, item in value.items():
        if isinstance(item, dict) and item:
            lines.append(f"{padding}{key}:")
            lines.append(to_yaml(item, indent + 1))
        elif isinstance(item, list) and item:
            lines.append(f"{padding}{key}:")
            lines.extend(f"{padding}- {yaml_scalar(entry)}" for entry in item)
        elif isinstance(item, (dict, list)):
            lines.append(f"{padding}{key}: {yaml_scalar(type(item)())}")
        else:
            lines.append(f"{padding}{key}: {yaml_scalar(item)}")
    return "\n".join(lines)


# vLLM 0.31 boolean flags that are plain store_true, so they have no `--no-` form.
STORE_TRUE_FLAGS = frozenset(
    {
        "aggregate-engine-logging",
        "data-parallel-multi-port-external-lb",
        "disable-log-stats",
    }
)


# Spell top-level `false` flags as `no-<flag>: true` for `vllm serve --config`.
# vLLM expands the file in its top-level parser, which does not know the serve
# subcommand's `--no-<flag>` options, so it silently drops every `false` and the
# vLLM default applies (prefix caching and async scheduling stayed on). A `true`
# is always passed through as `--<flag>`. Store-true flags keep `false`: their
# default is already off.
def vllm_config_file(config: dict[str, Any]) -> dict[str, Any]:
    spelled: dict[str, Any] = {}
    for key, item in config.items():
        if item is False and key not in STORE_TRUE_FLAGS:
            spelled[f"no-{key}"] = True
        else:
            spelled[key] = item
    return spelled


# Atomically write a generated vLLM YAML configuration.
def write_yaml(path: Path, value: Any) -> None:
    atomic_text(path, to_yaml(value) + "\n")


# Atomically write validated shell exports for the serving job.
def write_env(path: Path, values: dict[str, str]) -> None:
    lines: list[str] = []
    for key in sorted(values):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if "\n" in values[key]:
            raise ValueError(f"environment value contains a newline: {key}")
        lines.append(f"export {key}={shlex.quote(values[key])}\n")
    atomic_text(path, "".join(lines))


# Split and validate a composed scenario name.
def parse_components(value: str) -> list[str]:
    components = value.split("+")
    if not components or any(
        not re.fullmatch(r"[a-z0-9][a-z0-9-]*", item) for item in components
    ):
        raise ValueError(f"invalid SCENARIO: {value!r}")
    if len(components) != len(set(components)):
        raise ValueError(f"duplicate scenario component in {value!r}")
    return components


# Merge platform defaults first, then scenario layers in composition order.
def resolve_scenario(scenario: str, platform: str) -> dict[str, Any]:
    components = parse_components(scenario)
    if "baseline" in components and len(components) > 1:
        raise ValueError("baseline is implicit and must not be combined explicitly")

    baseline_path = SCENARIO_DIR / "baseline.json"
    platform_path = PLATFORM_DIR / f"{platform}.json"
    if not platform_path.is_file():
        available = ", ".join(
            sorted(item.stem for item in PLATFORM_DIR.glob("*.json"))
        )
        raise ValueError(
            f"unknown platform {platform!r}; available platform defaults: {available}"
        )

    baseline = load_json(baseline_path)
    if "platform_overrides" in baseline:
        raise ValueError(
            "baseline platform defaults belong in config/platforms/<platform>.json"
        )
    resolved = deep_merge(baseline, load_json(platform_path))
    source_paths = [baseline_path, platform_path]
    effective_components: list[str] = []
    if components == ["baseline"]:
        effective_components.append("baseline")
    else:
        for component in components:
            path = SCENARIO_DIR / f"{component}.json"
            if not path.is_file():
                available = ", ".join(
                    sorted(item.stem for item in SCENARIO_DIR.glob("*.json"))
                )
                raise ValueError(
                    f"unknown scenario {component!r}; available: {available}"
                )
            layer = load_json(path)
            platform_overrides = layer.pop("platform_overrides", {})
            resolved = deep_merge(resolved, layer)
            if platform in platform_overrides:
                resolved = deep_merge(resolved, platform_overrides[platform])
            source_paths.append(path)
            effective_components.append(component)
    scenario_id = "+".join(effective_components)
    resolved["id"] = scenario_id
    resolved["components"] = effective_components
    resolved["source_files"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in source_paths
    ]

    constraints = resolved.get("constraints", {})
    if platform not in constraints.get("platforms", []):
        raise ValueError(f"scenario {scenario_id} does not support platform {platform}")
    mode = resolved.get("mode", "single")
    if mode not in {"single", "disaggregated"}:
        raise ValueError(f"unsupported serving mode: {mode}")
    if mode == "disaggregated" and len(effective_components) != 1:
        raise ValueError("disaggregated-pd cannot be composed with another scenario")
    return resolved


# Sum role-assigned nodes for a disaggregated scenario.
def deployment_nodes(resolved: dict[str, Any]) -> int | None:
    if resolved.get("mode", "single") != "disaggregated":
        return None
    deployments = resolved["deployments"]
    nodes = sum(int(deployments[role]["nodes"]) for role in ("prefill", "decode"))
    if nodes <= 0:
        raise ValueError(f"scenario {resolved['id']} has no deployment nodes")
    return nodes


# Select and validate the Slurm node count for a resolved scenario.
def select_nodes(resolved: dict[str, Any], requested: int | None = None) -> int:
    constraints = resolved.get("constraints", {})
    minimum = int(constraints.get("min_nodes", 1))
    configured_exact = constraints.get("exact_nodes")
    assigned = deployment_nodes(resolved)
    exact_nodes = int(configured_exact) if configured_exact is not None else assigned
    if exact_nodes is not None and assigned is not None and exact_nodes != assigned:
        raise ValueError(
            f"scenario {resolved['id']} requires {exact_nodes} nodes "
            f"but its deployments assign {assigned}"
        )

    nodes = exact_nodes if requested is None and exact_nodes is not None else requested
    nodes = minimum if nodes is None else nodes
    if nodes <= 0:
        raise ValueError("node count must be positive")
    if exact_nodes is not None and nodes != exact_nodes:
        raise ValueError(f"scenario {resolved['id']} requires exactly {exact_nodes} nodes")
    if nodes < minimum:
        raise ValueError(f"scenario {resolved['id']} requires at least {minimum} nodes")
    return nodes


# Validate settings tied to the selected communication backend.
def validate_server(resolved: dict[str, Any], environment: dict[str, str]) -> None:
    server = resolved["server"]
    backend = str(server["all2all-backend"])
    if backend.startswith("deepep_"):
        required = {
            "FI_PROVIDER": "cxi",
            "UCCL_EP_TRANSPORT": "cxi",
            "FI_CXI_ENABLE_WRITEDATA": "1",
        }
        missing = {
            key: value for key, value in required.items() if environment.get(key) != value
        }
        if missing:
            raise ValueError(
                f"UCCL scenario is missing required CXI environment: {missing}"
            )


# First node address of every role. Disaggregated roles take consecutive slices of the
# allocation in the launcher's order, prefill first, and coordinate from their own first node.
def role_head_ips(
    resolved: dict[str, Any], head_ip: str, node_ips: list[str] | None
) -> dict[str, str]:
    if resolved.get("mode", "single") != "disaggregated":
        return {"single": head_ip}
    if not node_ips:
        return {"prefill": head_ip, "decode": head_ip}
    prefill_nodes = int(resolved["deployments"]["prefill"]["nodes"])
    if len(node_ips) <= prefill_nodes:
        raise ValueError("--node-ips must list every allocated node, prefill nodes first")
    return {"prefill": node_ips[0], "decode": node_ips[prefill_nodes]}


# vLLM configs for one pipeline-parallel instance that spans the whole allocation: each node
# hosts consecutive ranks (TP innermost, so one stage per node when TP = GPUs per node), and the
# nodes join through vLLM's multi-node mp backend, with every node but the first headless.
def pipeline_instance_configs(
    server: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    tp_size: int,
    pp_size: int,
    head_ip: str,
    api_port: int,
    master_port: int,
) -> dict[int, dict[str, Any]]:
    instance_gpus = tp_size * pp_size
    if instance_gpus % gpus_per_node or instance_gpus // gpus_per_node != nodes:
        raise ValueError(
            f"TP={tp_size} x PP={pp_size} needs exactly {instance_gpus / gpus_per_node:g} nodes "
            f"of {gpus_per_node} GPUs, allocation has {nodes}"
        )
    common = deep_merge(
        server,
        {
            "tensor-parallel-size": tp_size,
            "pipeline-parallel-size": pp_size,
            "distributed-executor-backend": "mp",
            "nnodes": nodes,
            "master-addr": head_ip,
            "master-port": master_port,
        },
    )
    configs = {}
    for node_rank in range(nodes):
        config = deepcopy(common)
        config["node-rank"] = node_rank
        if node_rank == 0:
            config["host"] = "0.0.0.0"
            config["port"] = api_port
        else:
            config["headless"] = True
        configs[node_rank] = config
    return configs


# Build the complete vLLM configuration executed by every role-local node.
def build_vllm_instance_configs(
    resolved: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    head_ip: str,
    api_port: int,
    dp_rpc_port: int,
    node_ips: list[str] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    tp_size = int(resolved["topology"]["tensor_parallel_size"])
    if gpus_per_node % tp_size:
        raise ValueError(
            f"gpus_per_node={gpus_per_node} is not divisible by tensor_parallel_size={tp_size}"
        )
    dp_local = gpus_per_node // tp_size
    mode = resolved.get("mode", "single")
    pp_size = int(resolved["topology"].get("pipeline_parallel_size", 1))
    if pp_size > 1:
        if mode != "single":
            raise ValueError(
                "set pipeline_parallel_size per role in disaggregated mode, not in the topology"
            )
        return {
            "single": pipeline_instance_configs(
                resolved["server"],
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
                head_ip=head_ip,
                api_port=api_port,
                master_port=dp_rpc_port,
            )
        }
    if mode == "single":
        roles = {
            "single": {
                "nodes": nodes,
                "api_port": api_port,
                "rpc_port": dp_rpc_port,
                "server": resolved["server"],
            }
        }
    else:
        deployments = resolved["deployments"]
        roles = {
            role: {
                "nodes": int(deployments[role]["nodes"]),
                # A role may set its own tensor and pipeline parallelism; DP fills the role's GPUs.
                "tp": int(deployments[role].get("tensor_parallel_size", tp_size)),
                "pp": int(deployments[role].get("pipeline_parallel_size", 1)),
                "api_port": int(deployments[role]["api_port"]),
                "rpc_port": int(deployments[role]["rpc_port"]),
                "server": deep_merge(
                    resolved["server"],
                    deployments[role]["server"],
                ),
            }
            for role in ("prefill", "decode")
        }

    heads = role_head_ips(resolved, head_ip, node_ips)
    configs: dict[str, dict[int, dict[str, Any]]] = {}
    for role, deployment in roles.items():
        role_nodes = int(deployment["nodes"])
        role_tp = int(deployment.get("tp", tp_size))
        role_pp = int(deployment.get("pp", 1))
        replica_gpus = role_tp * role_pp
        # A replica (one DP rank: TP x PP GPUs) either fits in a node or spans whole nodes.
        if gpus_per_node % replica_gpus and replica_gpus % gpus_per_node:
            raise ValueError(
                f"{role}: TP={role_tp} x PP={role_pp} must divide or be a multiple of "
                f"gpus_per_node={gpus_per_node}"
            )
        nodes_per_replica = max(replica_gpus // gpus_per_node, 1)
        if role_nodes % nodes_per_replica:
            raise ValueError(
                f"{role}: a TP={role_tp} x PP={role_pp} replica spans {nodes_per_replica} nodes, "
                f"which does not divide the role's {role_nodes} nodes"
            )
        dp_total = role_nodes * gpus_per_node // replica_gpus
        dp_local = max(gpus_per_node // replica_gpus, 1)
        parallel = {
            "tensor-parallel-size": role_tp,
            "data-parallel-size": dp_total,
            "data-parallel-size-local": dp_local,
            "data-parallel-address": heads[role],
            "data-parallel-rpc-port": int(deployment["rpc_port"]),
        }
        if role_pp > 1:
            parallel["pipeline-parallel-size"] = role_pp
        if nodes_per_replica > 1:
            # Replicas span nodes: vLLM's mp backend joins the role's nodes into one torch
            # world (master on the role head) and infers each node's DP rank from --node-rank.
            parallel.update(
                {
                    "distributed-executor-backend": "mp",
                    "nnodes": role_nodes,
                    "master-addr": heads[role],
                    "master-port": int(deployment["rpc_port"]) + 1000,
                }
            )
        common = deep_merge(deployment["server"], parallel)
        configs[role] = {}
        for node_rank in range(role_nodes):
            config = deepcopy(common)
            if nodes_per_replica > 1:
                config["node-rank"] = node_rank
            if node_rank == 0:
                config["host"] = "0.0.0.0"
                config["port"] = int(deployment["api_port"])
            else:
                config["headless"] = True
                if nodes_per_replica == 1:
                    config["data-parallel-start-rank"] = node_rank * dp_local
            configs[role][node_rank] = config
    return configs


# Validate an allocation and write its runtime scenario files.
def materialize_scenario(
    resolved: dict[str, Any],
    *,
    platform: str,
    nodes: int,
    gpus_per_node: int,
    model: str,
    load_format: str,
    head_ip: str,
    api_port: int,
    dp_rpc_port: int,
    output_dir: Path,
    node_ips: list[str] | None = None,
) -> Path:
    resolved = deepcopy(resolved)
    selected_nodes = select_nodes(resolved, nodes)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved["resolved_for"] = {
        "platform": platform,
        "nodes": selected_nodes,
        "gpus_per_node": gpus_per_node,
        "load_format": load_format,
    }

    topology = resolved["topology"]
    tp_size = int(topology["tensor_parallel_size"])
    if gpus_per_node % tp_size:
        raise ValueError(
            f"gpus_per_node={gpus_per_node} is not divisible by tensor_parallel_size={tp_size}"
        )
    pp_size = int(topology.get("pipeline_parallel_size", 1))
    # A pipeline-parallel instance spans nodes, so it has no node-local data parallelism.
    topology["data_parallel_size_local"] = gpus_per_node // tp_size if pp_size == 1 else 0
    topology["data_parallel_size"] = selected_nodes * gpus_per_node // (tp_size * pp_size)
    # vLLM's expert-parallel group spans TP x DP within one pipeline stage.
    topology["expert_parallel_size"] = topology["data_parallel_size"] * tp_size

    server = resolved["server"]
    server["model"] = model
    server["load-format"] = load_format
    environment = {
        key: str(value) for key, value in resolved.get("environment", {}).items()
    }
    validate_server(resolved, environment)
    backend = str(server["all2all-backend"])
    mode = resolved.get("mode", "single")

    if mode == "disaggregated":
        deployments = resolved["deployments"]
        for role in ("prefill", "decode"):
            deployments[role]["resolved_server"] = deep_merge(
                server,
                deployments[role]["server"],
            )

    instance_configs = build_vllm_instance_configs(
        resolved,
        nodes=selected_nodes,
        gpus_per_node=gpus_per_node,
        head_ip=head_ip,
        api_port=api_port,
        dp_rpc_port=dp_rpc_port,
        node_ips=node_ips,
    )
    config_records: dict[str, dict[str, dict[str, Any]]] = {}
    for role, node_configs in instance_configs.items():
        config_records[role] = {}
        for node_rank, config in node_configs.items():
            config_path = output_dir / f"vllm-{role}-node{node_rank}.yaml"
            write_yaml(config_path, vllm_config_file(config))
            identity_config = deepcopy(config)
            for address in ("data-parallel-address", "master-addr"):
                if address in identity_config:
                    identity_config[address] = "<runtime-address>"
            config_records[role][str(node_rank)] = {
                "path": config_path.name,
                "sha256": file_sha256(config_path),
                "identity_sha256": canonical_sha256(identity_config),
            }
    resolved["vllm_configs"] = config_records

    services = resolved.get("services", {})
    mooncake = services.get("mooncake_store", {})
    mooncake_config_path = output_dir / "mooncake-config.json"
    if mooncake.get("enabled"):
        mooncake_config = {
            "mode": mooncake["mode"],
            "metadata_server": mooncake["metadata_server"],
            "master_server_address": f"{head_ip}:{mooncake['master_port']}",
            "global_segment_size": mooncake["global_segment_size"],
            "local_buffer_size": mooncake["local_buffer_size"],
            "protocol": mooncake["protocol"],
            "device_name": mooncake["device_name"],
            "enable_offload": bool(mooncake["enable_offload"]),
            "tenant_id": mooncake["tenant_id"],
        }
        atomic_json(mooncake_config_path, mooncake_config)
        environment["MOONCAKE_CONFIG_PATH"] = str(mooncake_config_path)

    resolved["environment"] = environment
    resolved_path = output_dir / "scenario.json"
    atomic_json(resolved_path, resolved)

    exports = {
        **environment,
        "SCENARIO_ID": str(resolved["id"]),
        "SERVING_MODE": str(mode),
        "SCENARIO_RESOLVED": str(resolved_path),
        "SCENARIO_ENV_NAMES": ",".join(sorted(environment)),
        "SCENARIO_VLLM_CONFIG_DIR": str(output_dir),
        "API_PORT": str(api_port),
        "DP_RPC_PORT": str(dp_rpc_port),
        "TP_SIZE": str(tp_size),
        "PP_SIZE": str(pp_size),
        "ALL2ALL": backend,
        "MAX_MODEL_LEN": str(server["max-model-len"]),
        "MAX_NUM_SEQS": str(server["max-num-seqs"]),
        "GPU_MEM_UTIL": str(server["gpu-memory-utilization"]),
        "KV_CACHE_DTYPE": str(server["kv-cache-dtype"]),
        "SERVED_MODEL_NAME": str(server["served-model-name"]),
        "MOONCAKE_STORE_ENABLED": "1" if mooncake.get("enabled") else "0",
        "MOONCAKE_MASTER_PORT": str(mooncake.get("master_port", "")),
    }
    if mode == "disaggregated":
        deployments = resolved["deployments"]
        heads = role_head_ips(resolved, head_ip, node_ips)
        exports.update(
            {
                "PREFILL_HEAD_IP": heads["prefill"],
                "DECODE_HEAD_IP": heads["decode"],
                "PREFILL_NODES": str(deployments["prefill"]["nodes"]),
                "PREFILL_TP_SIZE": str(deployments["prefill"].get("tensor_parallel_size", tp_size)),
                "DECODE_TP_SIZE": str(deployments["decode"].get("tensor_parallel_size", tp_size)),
                "PREFILL_PP_SIZE": str(deployments["prefill"].get("pipeline_parallel_size", 1)),
                "DECODE_PP_SIZE": str(deployments["decode"].get("pipeline_parallel_size", 1)),
                "DECODE_NODES": str(deployments["decode"]["nodes"]),
                "PREFILL_API_PORT": str(deployments["prefill"]["api_port"]),
                "DECODE_API_PORT": str(deployments["decode"]["api_port"]),
                "PREFILL_RPC_PORT": str(deployments["prefill"]["rpc_port"]),
                "DECODE_RPC_PORT": str(deployments["decode"]["rpc_port"]),
                "PREFILL_BOOTSTRAP_PORT": str(
                    deployments["prefill"]["bootstrap_port"]
                ),
                "PROXY_PORT": str(deployments["proxy"]["port"]),
            }
        )
    write_env(output_dir / "scenario.env", exports)
    return resolved_path


# Resolve CLI inputs and materialize the selected runtime scenario.
def resolve(args: argparse.Namespace) -> None:
    resolved = resolve_scenario(args.scenario, args.platform)
    resolved_path = materialize_scenario(
        resolved,
        platform=args.platform,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        model=args.model,
        load_format=args.load_format,
        head_ip=args.head_ip,
        api_port=args.api_port,
        dp_rpc_port=args.dp_rpc_port,
        output_dir=args.output_dir,
        node_ips=[ip for ip in args.node_ips.split(",") if ip] or None,
    )
    print(resolved_path)


# Parse command-line options used inside the serving allocation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--platform", required=True, choices=("cuda", "rocm"))
    parser.add_argument("--nodes", required=True, type=int)
    parser.add_argument("--gpus-per-node", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-format", required=True)
    parser.add_argument("--head-ip", required=True)
    parser.add_argument(
        "--node-ips",
        default="",
        help="comma-separated addresses of all allocated nodes, in launcher order",
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--dp-rpc-port", type=int, default=13345)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    resolve(parse_args())
