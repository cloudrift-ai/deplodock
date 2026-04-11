#!/usr/bin/env bash
# Compare per-warp stall reasons for our TMA kernel vs cuBLAS at the same shape.
#
# Usage: scripts/diagnostics/ncu_stall_compare.sh [SIZE] [BATCH]
#   default: SIZE=8192 BATCH=1
#
# What it captures:
# 1. Per-warp stall reason breakdown for `fused_matmul` (our kernel)
# 2. Same for cuBLAS's dispatched SGEMM kernel at the same shape
# 3. Side-by-side delta showing where the FMA pipe utilization gap comes from
#
# This is the experiment that powers the "PTX is for Noobs: SASS Deep Dive"
# section of the article — specifically the claim that the constant ~5 pp
# generator-vs-hand-tuned-PTX gap shows up as `dispatch_stall` and
# `short_scoreboard` in the warp scheduler, not as wait-on-load latency.
#
# Requires ncu with elevated permissions (run under `sudo` or with
# nvidia-modprobe perms set per ERR_NVGPUCTRPERM).
set -euo pipefail

SIZE="${1:-8192}"
BATCH="${2:-1}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# All "warp stalled per issue-active cycle" metrics. Excludes the *_pipe_l1tex
# / *_pipe_mio sub-categorizations which we don't need.
METRICS=$(ncu --query-metrics 2>&1 \
    | grep -E "^smsp__average_warps_issue_stalled_" \
    | grep -v pipe_l1tex \
    | grep -v pipe_mio \
    | awk '{print $1}' \
    | tr '\n' ',' \
    | sed 's/,$//')

if [ -z "$METRICS" ]; then
    echo "ERROR: ncu --query-metrics returned no smsp__average_warps_issue_stalled_* metrics" >&2
    exit 1
fi

TMPDIR="$(mktemp -d /tmp/deplodock_stall_XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

# Build a small probe binary that just calls cublasSgemm at the right shape.
cat > "$TMPDIR/cublas_probe.cu" <<EOF
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main() {
    int N = $SIZE; int B = $BATCH;
    size_t bytes = (size_t)B * N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemset(dA, 0, bytes);
    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;
    if (B == 1) {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                    dA, N, dB, N, &beta, dC, N);
    } else {
        cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
            dA, N, (long long)N*N, dB, N, (long long)N*N, &beta,
            dC, N, (long long)N*N, B);
    }
    cudaDeviceSynchronize();
    cublasDestroy(h); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
EOF
arch="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
nvcc -O3 -arch="$arch" -lcublas -o "$TMPDIR/cublas_probe" "$TMPDIR/cublas_probe.cu" 2>&1 | tail -3

# Compile our bench at the same shape so we have a single-kernel binary
"$REPO_DIR/venv/bin/python" - <<PY > /dev/null
import sys, pathlib
sys.path.insert(0, "$REPO_DIR")
import dataclasses
from deplodock.compiler.benchmark import run_adaptive_benchmark_suite
from deplodock.compiler.backend.cuda.runner import generate_benchmark_program, _detect_arch
from deplodock.compiler.backend.cuda.tuning import default_matmul_strategy_map
from deplodock.compiler.backend.cuda.lower import lower_graph
from deplodock.compiler.backend.cuda.codegen import emit_kernel
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import FusedReduceElementwiseOp, InputOp
from deplodock.compiler.rewriter import Rewriter

g = Graph()
a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
c = g.add_node(FusedReduceElementwiseOp("sum", "mul", 1), [a, b], Tensor("C", ("M", "N")))
g.inputs = [a, b]; g.outputs = [c]

strategy_map, _ = default_matmul_strategy_map()
size = $SIZE; batch = $BATCH
selected = strategy_map[-1][1]
for thr, cfg in strategy_map:
    if size <= thr: selected = cfg; break
if batch > 1:
    selected = dataclasses.replace(selected, batch_count=batch, k_splits=1)
kernel = lower_graph(Rewriter().apply(g.copy()), config=selected)
src = emit_kernel(kernel)
dim_args = {"M": size, "N": size, "K": size}
if selected.k_splits > 1: dim_args["k_splits"] = selected.k_splits
if selected.batch_count > 1: dim_args["batch"] = selected.batch_count
prog = generate_benchmark_program(src, kernel, dim_args, num_iterations=2,
    compare_cublas=False, coarsen_cols=selected.coarsen_cols, coarsen_rows=selected.coarsen_rows,
    cublas_math_mode="default")
pathlib.Path("$TMPDIR/fused_probe.cu").write_text(prog)
PY
nvcc -O3 --fmad=true -arch="$arch" -lcuda -lcurand -o "$TMPDIR/fused_probe" "$TMPDIR/fused_probe.cu" 2>&1 | tail -3

# Helper: run ncu on a binary, filter to a kernel pattern, and print a clean
# stall reason table.
profile_kernel() {
    local label="$1" binary="$2" filter="$3"
    echo
    echo "=== $label ==="
    ncu --target-processes all --import-source no \
        --metrics "$METRICS" \
        --print-summary none --csv \
        "$binary" 2>&1 | grep -v "^==" | grep -F "$filter" | "$REPO_DIR/venv/bin/python" -c "
import sys, csv
# Each ncu run reports multiple sub-fields per metric (.pct, .ratio, .max_rate).
# We only want the .pct value, which is 'warps stalled per issue-active cycle'.
seen = {}
for line in sys.stdin:
    try:
        row = next(csv.reader([line.strip()]))
    except Exception: continue
    if len(row) < 15: continue
    full = row[12]
    if not full.startswith('smsp__average_warps_issue_stalled_'):
        continue
    if not full.endswith('.pct'):
        continue
    metric = full.replace('smsp__average_warps_issue_stalled_', '').replace('_per_issue_active.pct','')
    val_str = row[14].replace(',','')
    try: val = float(val_str)
    except ValueError: continue
    if metric not in seen or val > seen[metric]:
        seen[metric] = val
items = sorted(seen.items(), key=lambda kv: -kv[1])
total = sum(v for _, v in items)
for m, v in items:
    if v < 0.05: continue
    pct = v / total * 100 if total > 0 else 0
    print(f'  {m:25} {v:8.2f}%  ({pct:5.1f}% of stall total)')"
}

echo "# Per-warp stall reason comparison (size=$SIZE batch=$BATCH)"
echo
echo "Each value is 'warps stalled on this reason per issue-active cycle'."
echo "Sum can exceed 100% because multiple warps can stall in parallel."
profile_kernel "fused_matmul (TMA double-buffer, generated)" "$TMPDIR/fused_probe" "fused_matmul"
profile_kernel "cuBLAS dispatched kernel" "$TMPDIR/cublas_probe" "Kernel2"
