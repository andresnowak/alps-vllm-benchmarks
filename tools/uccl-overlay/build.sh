#!/usr/bin/env bash
# Build the UCCL-EP overlay inside the pinned vLLM image on a GH200 node. Started by
# tools/build-uccl-overlay.sh; do not run it directly. The build tree is node-local and
# discarded. The overlay appears at OVERLAY only after every check passes.
set -euo pipefail

: "${ROOT:?}" "${OVERLAY:?}" "${UCCL_OVERLAY_COMMIT:?}" "${IMAGE_REF:?}" "${IMAGE_DIGEST:?}"
UCCL_REPO="https://github.com/uccl-project/uccl.git"
UCCL_MAX_JOBS="${UCCL_MAX_JOBS:-16}"
# Build-only Python package of UCCL main since its nanobind bindings; the version eth-cscs/alps-extended-images#67 pins.
UCCL_NANOBIND_VERSION="${UCCL_NANOBIND_VERSION:-3.0.1}"
WORK="/tmp/uccl-overlay-build-${SLURM_JOB_ID:-local}"
STAGING="${OVERLAY}.staging-${SLURM_JOB_ID:-$$}"

step() { printf '[uccl-overlay %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
cleanup() { chmod -R u+w "${STAGING}" 2>/dev/null || true; rm -rf "${WORK}" "${STAGING}"; }
trap cleanup EXIT
rm -rf "${WORK}" "${STAGING}"
mkdir -p "${WORK}" "${STAGING}"

step "fetching ${UCCL_REPO}@${UCCL_OVERLAY_COMMIT}"
git init -q "${WORK}/uccl"
git -C "${WORK}/uccl" fetch -q --depth=1 "${UCCL_REPO}" "${UCCL_OVERLAY_COMMIT}"
git -C "${WORK}/uccl" checkout -q --detach FETCH_HEAD
[[ "$(git -C "${WORK}/uccl" rev-parse HEAD)" == "${UCCL_OVERLAY_COMMIT}" ]]
# Upstream UCCL is built as-is. A patch placed in tools/uccl-overlay/patches/ is applied and
# listed in overlay.json; adding one needs a rebuild by hand.
shopt -s nullglob
for patch_file in "${ROOT}"/tools/uccl-overlay/patches/*.patch; do
  step "applying $(basename "${patch_file}")"
  git -C "${WORK}/uccl" apply --check "${patch_file}"
  git -C "${WORK}/uccl" apply "${patch_file}"
done
shopt -u nullglob

# The image keeps the runtime libraries UCCL links against but its build cleanup removed their
# headers. Fetch the -dev packages into a build-only prefix; the image itself is not modified.
# Prefer the installed runtime version. If the archive no longer has it, accept only an older
# version: these libraries are backward compatible, so older headers are safe to link against
# the newer runtime library, while newer headers are not.
step "staging build-only headers for ibverbs, libnl, and numa"
DEPS="${WORK}/deps"
APT_OPTIONS=(-q -o Debug::NoLocking=1 -o "Dir::State::Lists=${WORK}/apt/lists"
  -o "Dir::Cache=${WORK}/apt/cache")
APT=(apt-get "${APT_OPTIONS[@]}")
APT_CACHE=(apt-cache "${APT_OPTIONS[@]}")
mkdir -p "${WORK}/apt/lists/partial" "${WORK}/apt/cache/archives/partial" "${DEPS}/debs"
"${APT[@]}" update >/dev/null
MULTIARCH="$(gcc -print-multiarch)"
HEADER_PACKAGES=()
for pair in libibverbs1:libibverbs-dev libnl-3-200:libnl-3-dev \
  libnl-route-3-200:libnl-route-3-dev libnuma1:libnuma-dev; do
  runtime="${pair%%:*}" dev="${pair##*:}"
  version="$(dpkg-query -W -f='${Version}' "${runtime}")"
  # Newest archive version at or below the runtime version; equal is the normal case.
  headers=""
  while read -r available; do
    if dpkg --compare-versions "${available}" le "${version}" && \
      { [[ -z "${headers}" ]] || dpkg --compare-versions "${available}" gt "${headers}"; }; then
      headers="${available}"
    fi
  done < <("${APT_CACHE[@]}" madison "${dev}" | awk -F'|' '{gsub(/ /, "", $2); print $2}')
  [[ -n "${headers}" ]] || {
    echo "the archive has no ${dev} at or below runtime ${runtime}=${version}" >&2
    exit 1
  }
  [[ "${headers}" == "${version}" ]] || \
    step "${dev}=${version} is not in the archive; using older headers ${headers}"
  (cd "${DEPS}/debs" && "${APT[@]}" download "${dev}=${headers}" >/dev/null)
  HEADER_PACKAGES+=("${dev}=${headers} (runtime ${runtime}=${version})")
done
for deb in "${DEPS}"/debs/*.deb; do dpkg-deb -x "${deb}" "${DEPS}/root"; done
# The -dev symlinks are relative to their package; point them at the image's runtime libraries.
for lib in ibverbs nl-3 nl-route-3 numa; do
  runtime_lib="$(find "/usr/lib/${MULTIARCH}" -maxdepth 1 -name "lib${lib}.so.[0-9]*" | sort | head -1)"
  [[ -n "${runtime_lib}" ]] || { echo "runtime library lib${lib} is missing from the image" >&2; exit 1; }
  ln -sfn "${runtime_lib}" "${DEPS}/root/usr/lib/${MULTIARCH}/lib${lib}.so"
done
export CPATH="${DEPS}/root/usr/include:${DEPS}/root/usr/include/libnl3${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${DEPS}/root/usr/lib/${MULTIARCH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

step "staging build-only nanobind==${UCCL_NANOBIND_VERSION}"
python3 -m pip install -q --no-deps --no-cache-dir --target "${WORK}/pydeps" "nanobind==${UCCL_NANOBIND_VERSION}"
export PYTHONPATH="${WORK}/pydeps${PYTHONPATH:+:${PYTHONPATH}}"
step "building uccl.ep with MAX_JOBS=${UCCL_MAX_JOBS}"
# The CUDA/GH200 flags of alps-extended-images apps/vllm/Containerfile. Changing them, or adding
# a patch, needs a rebuild by hand (remove the old overlay, then run tools/build-uccl-overlay.sh).
BUILD_ENV=(LIBFABRIC_HOME=/usr USE_LIBFABRIC_CXI=1 PER_EXPERT_BATCHING=1 NUM_MAX_NVL_PEERS=4
  MAX_NUM_GPUS=4 TORCH_CUDA_ARCH_LIST=9.0)
(
  cd "${WORK}/uccl/ep"
  env "${BUILD_ENV[@]}" MAX_JOBS="${UCCL_MAX_JOBS}" INSTALL_DIR="${STAGING}/uccl" \
    python3 setup.py build --build-base "${WORK}/ep-build" install
) 2>&1 | tee "${WORK}/build.log"
grep -q "GH200 detected" "${WORK}/build.log" || {
  echo "UCCL build did not detect GH200; it must run on a GH200 node" >&2
  exit 1
}
printf '%s\n' '"""UCCL expert-parallel extension."""' > "${STAGING}/uccl/__init__.py"
cp -LR "${WORK}/uccl/ep/deep_ep_wrapper/deep_ep" "${STAGING}/deep_ep"
find "${STAGING}" -name __pycache__ -prune -exec rm -rf {} +

step "checking the built overlay"
missing="$(ldd "${STAGING}/uccl/ep.abi3.so" | awk '/not found/' || true)"
[[ -z "${missing}" ]] || { echo "uccl.ep has unresolved libraries:" >&2; echo "${missing}" >&2; exit 1; }
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${STAGING}" STAGING="${STAGING}" python3 - <<'PY'
import os
import deep_ep
import uccl.ep

staging = os.environ["STAGING"]
for module in (uccl.ep, deep_ep):
    assert module.__file__.startswith(staging + "/"), module.__file__
print(f"import check passed: uccl.ep={uccl.ep.__file__} deep_ep={deep_ep.__file__}")
PY

# What the runtime record and provenance report about this overlay.
BUILD_ENV_TEXT="${BUILD_ENV[*]}" UCCL_NANOBIND_VERSION="${UCCL_NANOBIND_VERSION}" HEADERS="$(printf '%s\n' "${HEADER_PACKAGES[@]}")" \
PATCHES="$(cd "${ROOT}/tools/uccl-overlay/patches" 2>/dev/null && sha256sum ./*.patch 2>/dev/null || true)" \
python3 - "${STAGING}/overlay.json" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone

import torch

json.dump({
    "component": "uccl-ep",
    "repo": "https://github.com/uccl-project/uccl.git",
    "commit": os.environ["UCCL_OVERLAY_COMMIT"],
    "image_ref": os.environ["IMAGE_REF"],
    "image_digest": os.environ["IMAGE_DIGEST"],
    "build_env": dict(item.split("=", 1) for item in os.environ["BUILD_ENV_TEXT"].split()),
    "patches": [line.split()[1].removeprefix("./") for line in os.environ["PATCHES"].splitlines()],
    "patch_sha256": [line.split()[0] for line in os.environ["PATCHES"].splitlines()],
    "header_packages": os.environ["HEADERS"].splitlines(),
    "build_python_packages": [f"nanobind=={os.environ['UCCL_NANOBIND_VERSION']}"],
    "built": {"at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "slurm_job_id": os.getenv("SLURM_JOB_ID"), "torch": torch.__version__,
              "cuda": torch.version.cuda},
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
chmod -R a-w "${STAGING}"
mv -T "${STAGING}" "${OVERLAY}"
step "overlay ready: ${OVERLAY}"
