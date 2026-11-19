#!/usr/bin/env bash
# Build the UCCL-EP overlay pinned by config/platforms/cuda.env against that profile's image.
# Run from the repository root on a login node; the build runs on one GH200 node. Rebuild by
# hand after changing IMAGE_REF or UCCL_OVERLAY_COMMIT: the directory name carries both, so a job
# whose overlay does not exist stops at start and points here. An existing build is reused.
#
# Overrides: SLURM_ACCOUNT (csstaff), SLURM_PARTITION (normal), BUILD_TIME (01:00:00).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY=(uv run --no-project --python 3.12 -- python)
# shellcheck source=/dev/null
source config/platforms/cuda.env
: "${UCCL_OVERLAY_COMMIT:?cuda.env does not pin UCCL_OVERLAY_COMMIT}"
: "${UCCL_OVERLAY_ROOT:?cuda.env does not set UCCL_OVERLAY_ROOT}"

IMAGE_DIGEST="$("${PY[@]}" scripts/resolve-image-digest.py "${IMAGE_REF}")"
# Same name as sbatch/serve.sbatch computes: uccl-<commit 12>-<image digest 12>.
DIGEST="${IMAGE_DIGEST#sha256:}"
OVERLAY="${UCCL_OVERLAY_ROOT}/uccl-${UCCL_OVERLAY_COMMIT:0:12}-${DIGEST:0:12}"
if [[ -e "${OVERLAY}" ]]; then
  echo "UCCL overlay already built: ${OVERLAY}"
  real="$(realpath "${OVERLAY}")"
  echo "to rebuild it (e.g. after changing the build flags or a UCCL patch), remove it first:"
  echo "  chmod -R u+w '${real}' && rm -rf '${real}' '${OVERLAY}'"
  exit 0
fi

LOGDIR="${UCCL_OVERLAY_ROOT}/build-logs/$(basename "${OVERLAY}")-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOGDIR}"
EDF="${LOGDIR}/build.toml"
cat > "${EDF}" <<EOF
image = "${IMAGE_REF}"
mounts = ["${ROOT}:${ROOT}", "${UCCL_OVERLAY_ROOT}:${UCCL_OVERLAY_ROOT}"]
workdir = "${ROOT}"
writable = true
entrypoint = true

[annotations]
com.hooks.cxi.enabled = "false"
com.hooks.netstack.source = "host"

[env]
XDG_CACHE_HOME = "/tmp/xdg-cache"
PIP_CACHE_DIR = "/tmp/pip-cache"
EOF

echo "building UCCL ${UCCL_OVERLAY_COMMIT:0:12} for ${IMAGE_REF}"
echo "  digest:  ${IMAGE_DIGEST}"
echo "  overlay: ${OVERLAY}"
echo "  log:     ${LOGDIR}/build.log (the job may queue before it starts)"
export ROOT OVERLAY UCCL_OVERLAY_COMMIT IMAGE_REF IMAGE_DIGEST
srun \
  --job-name=uccl-overlay-build \
  --account="${SLURM_ACCOUNT:-csstaff}" \
  --partition="${SLURM_PARTITION:-normal}" \
  --nodes=1 --ntasks=1 --exclusive \
  --time="${BUILD_TIME:-01:00:00}" \
  --network=disable_rdzv_get \
  --environment="${EDF}" \
  bash "${ROOT}/tools/uccl-overlay/build.sh" > "${LOGDIR}/build.log" 2>&1 || {
  echo "UCCL overlay build failed; last lines of ${LOGDIR}/build.log:" >&2
  tail -n 30 "${LOGDIR}/build.log" >&2
  exit 1
}
[[ -f "${OVERLAY}/overlay.json" ]] || { echo "the build did not produce ${OVERLAY}" >&2; exit 1; }
echo "UCCL overlay ready: ${OVERLAY}"
