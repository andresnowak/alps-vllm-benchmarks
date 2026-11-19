#!/usr/bin/env bash
# Per-node worker inside the selected software-stack runtime. The scenario resolver writes one
# complete vLLM YAML for each role-local node. This script selects that file, captures the runtime
# fingerprint, and launches it without appending vLLM configuration flags.
set -euo pipefail

: "${MODEL:?MODEL must be an absolute checkpoint path}"
: "${SCENARIO_VLLM_CONFIG_DIR:?SCENARIO_VLLM_CONFIG_DIR is required}"
: "${LOGDIR:?LOGDIR is required}"
: "${SCENARIO_ENV:?SCENARIO_ENV must be the generated scenario.env}"
: "${ROOT:?ROOT must be the repository root}"
: "${SERVING_MODE:?SERVING_MODE must be single or disaggregated}"

# shellcheck source=/dev/null
source "${SCENARIO_ENV}"

# Container jobs see the checkpoint through the EDF bind mount; uenv jobs use its host path
# directly. A missing checkpoint must fail loudly here instead of sending vLLM into an offline HF
# hub lookup that produces a confusing LocalEntryNotFoundError. Retry briefly because the NFS
# source can take a moment to appear.
for _ in {1..30}; do
  [[ -f "${MODEL}/config.json" ]] && break
  sleep 2
done
[[ -f "${MODEL}/config.json" ]] || {
  echo "model directory is not visible in this container: ${MODEL}" >&2
  echo "the bind mount from the launcher is missing on node $(hostname)" >&2
  exit 2
}
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP_SIZE="${TP_SIZE:-2}"
(( GPUS_PER_NODE % TP_SIZE == 0 )) || {
  echo "GPUS_PER_NODE=${GPUS_PER_NODE} is not divisible by TP_SIZE=${TP_SIZE}" >&2
  exit 2
}

# One Slurm task runs per serving node. Prefill and decode share one srun step whose task ranks
# are global, so the launcher passes each role's first rank as ROLE_TASK_OFFSET.
NODE_RANK=$(( ${SLURM_PROCID:?SLURM_PROCID is required} - ${ROLE_TASK_OFFSET:-0} ))
# Instances span TP x PP GPUs; a pipeline-parallel instance spans several nodes.
PP_SIZE="${PP_SIZE:-1}"
# A replica that spans nodes counts as one local DP rank, as in vLLM.
DP_LOCAL=$(( GPUS_PER_NODE >= TP_SIZE * PP_SIZE ? GPUS_PER_NODE / (TP_SIZE * PP_SIZE) : 1 ))

role="${DEPLOYMENT_ROLE:-single}"
if [[ "${SERVING_MODE}" == "single" ]]; then
  [[ "${role}" == "single" ]] || { echo "SERVING_MODE=single but DEPLOYMENT_ROLE=${role}" >&2; exit 2; }
  DP_TOTAL=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE / (TP_SIZE * PP_SIZE)))
else
  if [[ "${role}" != "prefill" && "${role}" != "decode" ]]; then
    echo "SERVING_MODE=disaggregated requires DEPLOYMENT_ROLE=prefill|decode, got '${role}'" >&2
    exit 2
  fi
  # Each role may use its own tensor and pipeline parallelism.
  if [[ "${role}" == "prefill" ]]; then
    TP_SIZE="${PREFILL_TP_SIZE:-${TP_SIZE}}"
    PP_SIZE="${PREFILL_PP_SIZE:-${PP_SIZE}}"
    DP_TOTAL=$((PREFILL_NODES * GPUS_PER_NODE / (TP_SIZE * PP_SIZE)))
  else
    TP_SIZE="${DECODE_TP_SIZE:-${TP_SIZE}}"
    PP_SIZE="${DECODE_PP_SIZE:-${PP_SIZE}}"
    DP_TOTAL=$((DECODE_NODES * GPUS_PER_NODE / (TP_SIZE * PP_SIZE)))
  fi
  # A replica that spans nodes still counts as one local DP rank, as in vLLM.
  DP_LOCAL=$(( GPUS_PER_NODE >= TP_SIZE * PP_SIZE ? GPUS_PER_NODE / (TP_SIZE * PP_SIZE) : 1 ))
fi
# Apply the selected git patches to this container's packages before anything imports them; a
# patch that neither applies nor is already in the image stops the node here.
PATCH_STATUS="${LOGDIR}/patches-${role}-node${NODE_RANK}.json"
python3 "${ROOT}/scripts/runtime-patches.py" --status "${PATCH_STATUS}"
export RUNTIME_PATCHES_STATUS_FILE="${PATCH_STATUS}"
CONFIG="${SCENARIO_VLLM_CONFIG_DIR}/vllm-${role}-node${NODE_RANK}.yaml"
[[ -f "${CONFIG}" ]] || { echo "vLLM config is missing: ${CONFIG}" >&2; exit 2; }
CACHE_STATUS="${LOGDIR}/jit-cache-${role}-node${NODE_RANK}.json"
python3 "${ROOT}/scripts/jit-cache.py" hydrate --status "${CACHE_STATUS}"
export JIT_CACHE_STATUS_FILE="${CACHE_STATUS}"

# Every node captures its container-local /tmp cache after the full benchmark
# warmup, or after smoke when no benchmark suite is selected.
if [[ "${JIT_CACHE_MODE:-readwrite}" == "readwrite" ]]; then
  python3 "${ROOT}/scripts/jit-cache.py" agent \
    --status "${CACHE_STATUS}" \
    --trigger "${JIT_CACHE_TRIGGER:?JIT_CACHE_TRIGGER is required}" \
    --staging "${JIT_CACHE_STAGING:?JIT_CACHE_STAGING is required}" \
    --role "${role}" \
    > "${LOGDIR}/jit-cache-agent-${role}-node${NODE_RANK}.log" 2>&1 &
  echo "[node ${NODE_RANK}] JIT cache agent started for role=${role} (pid $!)"
fi

printf '[node %s] host=%s platform=%s role=%s scenario=%s TP=%s PP=%s DP=%s local-DP=%s EP=%s\n' \
  "${NODE_RANK}" "$(hostname)" "${PLATFORM:-unknown}" "${role}" "${SCENARIO_ID:-unknown}" \
  "${TP_SIZE}" "${PP_SIZE}" "${DP_TOTAL}" "${DP_LOCAL}" "$((DP_TOTAL * TP_SIZE))"
CONFIG_ARGS=(--config "${CONFIG}")
python -c 'import torch, vllm; print(f"torch={torch.__version__} cuda={torch.version.cuda} hip={torch.version.hip} vllm={vllm.__version__}")'
SERVER_COMMAND=(vllm serve "${CONFIG_ARGS[@]}")

# Record the config-only command executed by this node.
python3 "${ROOT}/scripts/runtime-fingerprint.py" \
  --output "${LOGDIR}/runtime-${role}-node${NODE_RANK}.json" \
  --command-output "${LOGDIR}/server-command-${role}-node${NODE_RANK}.json" \
  -- "${SERVER_COMMAND[@]}"

if [[ "${MOONCAKE_STORE_ENABLED:-0}" == "1" && "${NODE_RANK}" == "0" ]]; then
  command -v mooncake_master >/dev/null || {
    echo "MOONCAKE_STORE_ENABLED=1 but mooncake_master is not in the image" >&2
    exit 2
  }
  MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:?MOONCAKE_MASTER_PORT is required}"
  mooncake_master --port "${MOONCAKE_MASTER_PORT}" \
    > "${LOGDIR}/mooncake-master.log" 2>&1 &
  echo "[node 0] mooncake_master started on ${MOONCAKE_MASTER_PORT} (pid $!)"
fi

exec "${SERVER_COMMAND[@]}" \
  2>&1 | tee "${LOGDIR}/serve-${role}-node${NODE_RANK}.log"
