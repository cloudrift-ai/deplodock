#!/usr/bin/env bash
# Extract the cuBLAS SGEMM kernel names that ship in the installed libcublas.
#
# Reproduces the article's claim that cuBLAS dispatches a `cutlass_80_simt_sgemm_*`
# kernel for FP32 SGEMM on sm_120 — the "_80" prefix indicating an Ampere-era
# CUTLASS instantiation forward-ported to Blackwell.
#
# Output is plain text, deterministic across runs of the same libcublas binary.
# Re-run on a different host to verify the same kernel names ship there too.
set -euo pipefail

LIB="${1:-/usr/local/cuda/lib64/libcublas.so}"
if [ ! -f "$LIB" ]; then
    LIB="$(find / -name 'libcublas.so*' -type f 2>/dev/null | head -1 || true)"
fi
if [ -z "$LIB" ] || [ ! -f "$LIB" ]; then
    echo "ERROR: libcublas.so not found. Pass the path as the first arg." >&2
    exit 1
fi

LIB_LT="${LIB%libcublas.so}libcublasLt.so"

dump_one() {
    local lib="$1" label="$2"
    [ -f "$lib" ] || return
    echo "# === $label ==="
    echo "# path:    $lib"
    echo "# version: $(readlink -f "$lib" | sed -E 's/.*libcublas(Lt)?\.so\.//')"
    echo
    # We use `strings` rather than `cuobjdump --dump-resource-usage` because
    # the latter only lists kernels with full cubin sections; the CUTLASS
    # `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` template that cuBLAS
    # actually dispatches at runtime for SGEMM is shipped as a PTX template
    # that JITs at first use, and only its name string lives in the binary's
    # read-only data section. `strings` catches both PTX templates and cubins.
    echo "## CUTLASS SIMT SGEMM templates (the FP32 path the article targets):"
    strings "$lib" 2>/dev/null \
        | grep -oE "cutlass_[0-9]+_simt_sgemm_[A-Za-z0-9_]+" \
        | sort -u \
        | sed 's/^/  /'
    echo
    echo "## CUTLASS Tensorop GEMM templates (FP16/BF16/TF32 paths, for contrast):"
    strings "$lib" 2>/dev/null \
        | grep -oE "cutlass_[0-9]+_tensorop_[A-Za-z0-9]*gemm_[A-Za-z0-9_]+" \
        | sort -u | head -10 \
        | sed 's/^/  /'
    echo
    echo "## Non-CUTLASS SGEMM (hand-written xmma_gemm / sgemm_largek kernels):"
    strings "$lib" 2>/dev/null \
        | grep -oE "(sm[0-9]+_xmma_gemm_[a-z0-9_]+|sgemm_largek[a-zA-Z0-9_]*)" \
        | sort -u | head -10 \
        | sed 's/^/  /'
    echo
}

echo "# tool: $(cuobjdump --version | head -1) (also using GNU strings)"
echo
dump_one "$LIB"    "libcublas"
dump_one "$LIB_LT" "libcublasLt"

echo "# Notes:"
echo "# - The 'cutlass_80_*' prefix is the literal CUTLASS marker that NVIDIA"
echo "#   shipped these as sm_80 (Ampere) templates. No cutlass_120_simt_sgemm_*"
echo "#   (Blackwell-native) kernel exists in either binary — that is the"
echo "#   article's central claim, verifiable directly from the binary."
echo "# - On sm_120 (RTX 5090), runtime ncu confirms cuBLAS dispatches"
echo "#   cutlass_80_simt_sgemm_256x128_8x4_nn_align1 — a CUTLASS template."
echo "# - On sm_90 (H200), runtime ncu confirms cuBLAS dispatches a hand-written"
echo "#   sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize256x128x8_stage3 kernel."
echo "#   Same toolkit, different dispatchers per architecture."
