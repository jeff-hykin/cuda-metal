#!/usr/bin/env bash
# tex2D() reads a cumetalTextureRecord_t through the texture handle, so this gate covers both the
# host side that publishes the record and the device side that samples through it. It has to run
# with CUMETAL_USE_METAL_DEVICE_ADDRESSES=1: under the default CPU shared mapping the handle is not
# a GPU-dereferenceable address.
set -euo pipefail

CUMETALC="${1:?usage: run_texture_object.sh <cumetalc> <source.cu> <workdir>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
    exit 77
fi

CLANG_BIN="${CUMETAL_CUDA_CLANG:-${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}}"
if [[ ! -x "${CLANG_BIN}" ]]; then
    CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "SKIP: no clang++ available to drive CUDA compilation"
    exit 77
fi

mkdir -p "${WORK_DIR}"
OUT_BIN="${WORK_DIR}/texture_object"
rm -f "${OUT_BIN}"

BUILD_STATUS=0
"${CUMETALC}" "${SOURCE_CU}" -o "${OUT_BIN}" || BUILD_STATUS=$?
if [[ ${BUILD_STATUS} -ne 0 ]]; then
    echo "FAIL: cumetalc exited ${BUILD_STATUS} building ${SOURCE_CU}"
    exit 1
fi

RUN_STATUS=0
RUN_OUTPUT="$(CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 CUMETAL_TRACE_GPU=1 "${OUT_BIN}" 2>&1)" || RUN_STATUS=$?
echo "${RUN_OUTPUT}"

if [[ ${RUN_STATUS} -ne 0 ]]; then
    echo "FAIL: texture test exited ${RUN_STATUS}"
    exit 1
fi
if ! grep -q "PASS:" <<<"${RUN_OUTPUT}"; then
    echo "FAIL: texture test did not report a numerical PASS"
    exit 1
fi
# Correct values alone would also come out of a host fallback; require GPU provenance.
if ! grep -q "device=apple_gpu" <<<"${RUN_OUTPUT}"; then
    echo "FAIL: no Apple GPU provenance; the sampling kernels did not dispatch to the GPU"
    exit 1
fi

echo "PASS: texture objects sampled on the Apple GPU"
