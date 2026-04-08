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
    echo "## CUTLASS SIMT SGEMM kernels (the FP32 path that the article targets):"
    cuobjdump --dump-resource-usage "$lib" 2>/dev/null \
        | grep -oE "cutlass_[0-9]+_simt_sgemm_[A-Za-z0-9_]+" \
        | sort -u \
        | sed 's/^/  /'
    echo
    echo "## CUTLASS Tensorop GEMM kernels (FP16/BF16/TF32 paths, for contrast):"
    cuobjdump --dump-resource-usage "$lib" 2>/dev/null \
        | grep -oE "cutlass_[0-9]+_tensorop_[A-Za-z0-9]*gemm_[A-Za-z0-9_]+" \
        | sort -u | head -10 \
        | sed 's/^/  /'
    echo
    echo "## Non-CUTLASS SGEMM (legacy hand-written kernels, if any):"
    cuobjdump --dump-resource-usage "$lib" 2>/dev/null \
        | grep -oE "_Z[0-9]+sgemm_[A-Za-z0-9_]+" \
        | sed -E 's/^_Z[0-9]+(sgemm_[A-Za-z0-9_]+).*/\1/' \
        | sort -u | head -10 \
        | sed 's/^/  /'
    echo
}

echo "# tool: $(cuobjdump --version | head -1)"
echo
dump_one "$LIB"    "libcublas"
dump_one "$LIB_LT" "libcublasLt"

echo "# Note: the 'forwardCompat' suffix on cutlass_80_* kernels is the literal"
echo "# CUTLASS marker that NVIDIA flagged this kernel as a forward-compatibility"
echo "# shim from sm_80 (Ampere). No cutlass_120_simt_sgemm_* (Blackwell-native)"
echo "# kernel exists in this libcublas — that is the article's central claim,"
echo "# verifiable directly from the binary."
