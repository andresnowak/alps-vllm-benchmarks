"""Multi-turn chat points for scripts/benchmark.py.

A suite row with `driver` set to `multi-turn` runs vLLM's multi-turn chat benchmark
(tools/vllm-multi-turn) instead of `vllm bench serve`. Besides the usual columns it needs:

  dataset            plain-text file <dataset>.txt in the dataset dir; conversations are cut from it
  turns              messages per conversation, user and assistant (even)
  prefix_len         tokens of context unique to each conversation, in its first message
  common_prefix_len  tokens shared by every conversation, like a system prompt (0 for none)
  conversations      conversations generated per invocation
  active             conversations open at once over all clients

input_len and output_len are the tokens of each user and assistant message, concurrency is the
number of clients (each has one request in flight), and prompts is the number of requests
measured.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

DRIVER = "multi-turn"
TOOL = Path(__file__).resolve().parents[1] / "tools/vllm-multi-turn/benchmark_serving_multi_turn.py"
FIELDS = ("turns", "prefix_len", "common_prefix_len", "conversations", "active")


# Read and check the multi-turn columns of one suite row into its point.
def parse_row(row: dict[str, str], point: dict[str, Any], line: int) -> None:
    if not (row.get("dataset") or "").strip():
        raise ValueError(f"line {line}: multi-turn points need a dataset (a text file)")
    point["driver"] = DRIVER
    for field in FIELDS:
        value = (row.get(field) or "").strip()
        if not value:
            raise ValueError(f"line {line}: multi-turn points need {field}")
        point[field] = int(value)
        if point[field] < (0 if field == "common_prefix_len" else 1):
            raise ValueError(f"line {line}: {field} is out of range")


# Checks that need the point's shared columns as well.
def check_point(point: dict[str, Any], line: int) -> None:
    if point["turns"] % 2:
        raise ValueError(f"line {line}: turns must be even (user and assistant messages)")
    if point["active"] < point["concurrency"]:
        raise ValueError(f"line {line}: active conversations must reach concurrency")
    # Twice the measured requests, so no client runs out of conversations before the end.
    if point["conversations"] * point["turns"] // 2 < 2 * point["prompts"]:
        raise ValueError(f"line {line}: conversations x turns / 2 must be at least twice prompts")


def text_file(dataset_dir: Path, point: dict[str, Any]) -> Path:
    return dataset_dir / f"{point['dataset']}.txt"


# The tool's conversation config, every length fixed by the suite point.
def conversations_config(point: dict[str, Any], text: Path) -> dict[str, Any]:
    def constant(value: int) -> dict[str, Any]:
        return {"distribution": "constant", "value": value}

    return {
        "filetype": "generate_conversations",
        "num_conversations": point["conversations"],
        "text_files": [str(text)],
        "print_stats": False,
        "prompt_input": {
            "num_turns": constant(point["turns"]),
            "common_prefix_num_tokens": constant(point["common_prefix_len"]),
            "prefix_num_tokens": constant(point["prefix_len"]),
            "num_tokens": constant(point["input_len"]),
        },
        "prompt_output": {"num_tokens": constant(point["output_len"])},
    }


# The tool cuts the same text for the same lengths whatever its seed, so a later invocation
# would find every prefix already in the prefix cache. Each invocation reads the text rotated
# to an offset drawn from its seed instead, so no two invocations share conversation text.
def rotated_text(source: Path, seed: int, directory: Path) -> tuple[Path, int]:
    text = source.read_text()
    start = random.Random(seed).randrange(len(text))
    offset = text.find(" ", start) + 1 or 0
    path = directory / f"{source.stem}-{seed}.txt"
    path.write_text(text[offset:] + "\n" + text[:offset])
    return path, offset


# One invocation: clients send the turns of their open conversations, and the run stops after
# exactly the point's request count.
def command(
    point: dict[str, Any],
    *,
    python: str,
    model: str,
    served_model_name: str,
    base_url: str,
    config_file: Path,
    result_file: Path,
    seed: int,
    request_timeout: int,
) -> list[str]:
    return [
        python,
        str(TOOL),
        "--input-file",
        str(config_file),
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--url",
        base_url,
        "--num-clients",
        str(point["concurrency"]),
        "--max-active-conversations",
        str(point["active"]),
        "--max-num-requests",
        str(point["prompts"]),
        "--no-early-stop",
        "--seed",
        str(seed),
        "--request-timeout-sec",
        str(request_timeout),
        "--stats-json-output",
        str(result_file),
    ]


# The tool's per-request stats as the metrics that vllm bench serve reports.
def load_result(result_file: Path, point: dict[str, Any]) -> tuple[dict[str, Any], int]:
    stats = json.loads(result_file.read_text())
    if len(stats) != point["prompts"]:
        raise RuntimeError(f"{point['name']}: completed={len(stats)}/{point['prompts']}")
    # Every answer must have exactly output_len tokens (ignore_eos). The tool takes the count
    # from the server's usage, so reasoning tokens are included; a smaller count means the
    # client missed part of the output and its TTFT and TPOT are wrong.
    wrong = [r["output_num_tokens"] for r in stats if r["output_num_tokens"] != point["output_len"]]
    if wrong:
        raise RuntimeError(
            f"{point['name']}: {len(wrong)} of {len(stats)} requests returned other than "
            f"{point['output_len']} output tokens (first: {wrong[0]})"
        )
    start = min(request["start_time_ms"] for request in stats)
    end = max(request["start_time_ms"] + request["latency_ms"] for request in stats)
    duration = (end - start) / 1000
    total_input = sum(request["input_num_tokens"] for request in stats)
    total_output = sum(request["output_num_tokens"] for request in stats)
    result: dict[str, Any] = {
        "completed": len(stats),
        "failed": 0,
        "duration": duration,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(stats) / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "mean_input_tokens": total_input / len(stats),
        # The tool's own estimate from the conversation history, not the server's count.
        "mean_approx_cached_percent": statistics.fmean(
            request["approx_cached_percent"] for request in stats
        ),
    }
    for metric, field in (("ttft", "ttft_ms"), ("tpot", "tpot_ms"), ("e2el", "latency_ms")):
        values = [float(request[field]) for request in stats]
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        result[f"mean_{metric}_ms"] = statistics.fmean(values)
        result[f"std_{metric}_ms"] = statistics.pstdev(values)
        result[f"median_{metric}_ms"] = statistics.median(values)
        for percentile in (50, 90, 95, 99):
            result[f"p{percentile}_{metric}_ms"] = cuts[percentile - 1]
    return result, total_input
