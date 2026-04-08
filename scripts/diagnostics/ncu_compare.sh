#!/usr/bin/env bash
# Profile our TMA kernel and cuBLAS side-by-side at one matrix size with ncu.
#
# Reproduces the ncu metrics tables in the article (IPC, FMA pipe utilization,
# issue active, DRAM throughput). Run as:
#
#     scripts/diagnostics/ncu_compare.sh [SIZE] [BATCH]
#
# Defaults to SIZE=4096, BATCH=1. The script:
#   1. Runs `bench_matmul.py` once to compile a fresh bench binary into a
#      tempdir under /tmp/deplodock_bench_*
#   2. Locates that binary
#   3. Re-runs it under `ncu --target-processes all --metrics ...`
#
# Note: ncu typically requires either root, `setcap cap_sys_admin`, or the
# `nvidia-modprobe -m` performance-counters policy to be relaxed. If you see
# "ERR_NVGPUCTRPERM", run as root or follow:
#   https://developer.nvidia.com/ERR_NVGPUCTRPERM
set -euo pipefail

SIZE="${1:-4096}"
BATCH="${2:-1}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

METRICS=(
    sm__cycles_active.avg
    sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
    sm__inst_executed.avg.per_cycle_active
    sm__cycles_active.avg.pct_of_peak_sustained_active
    smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
    smsp__cycles_active.avg.pct_of_peak_sustained_active
    sm__warps_active.avg.pct_of_peak_sustained_active
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    launch__registers_per_thread
    launch__shared_mem_per_block
)

METRICS_CSV=$(IFS=, ; echo "${METRICS[*]}")

# Trigger a compile so we can find the bench binary.
echo "## Compiling bench binary at SIZE=$SIZE BATCH=$BATCH ..." >&2
"$REPO_DIR/venv/bin/python" "$REPO_DIR/scripts/bench_matmul.py" \
    --strategy adaptive \
    --sizes "$SIZE" \
    --batches "$BATCH" \
    --iterations 2 >/dev/null

# The runner uses tempfile.TemporaryDirectory, which deletes on exit. We need
# to recompile under our control. Generate the program inline by reusing the
# same Python helper.
TMPDIR="$(mktemp -d /tmp/deplodock_ncu_XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

"$REPO_DIR/venv/bin/python" - <<PY
import sys, pathlib
sys.path.insert(0, "$REPO_DIR")
from deplodock.compiler.benchmark import run_adaptive_benchmark_suite
from deplodock.compiler.cuda.lower import MatmulConfig
from deplodock.compiler.cuda.runner import generate_benchmark_program, _detect_arch
from deplodock.compiler.cuda.tuning import default_matmul_strategy_map
from deplodock.compiler.cuda.lower import lower_graph
from deplodock.compiler.cuda.codegen import emit_kernel
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import FusedReduceElementwiseOp, InputOp
from deplodock.compiler.rewriter import Rewriter

g = Graph()
a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
c = g.add_node(FusedReduceElementwiseOp("sum", "mul", 1), [a, b], Tensor("C", ("M", "N")))
g.inputs = [a, b]; g.outputs = [c]

strategy_map, _ = default_matmul_strategy_map()
size = $SIZE
batch = $BATCH
selected = strategy_map[-1][1]
for thr, cfg in strategy_map:
    if size <= thr:
        selected = cfg; break
import dataclasses
if batch > 1:
    selected = dataclasses.replace(selected, batch_count=batch, k_splits=1)
kernel = lower_graph(Rewriter().apply(g.copy()), config=selected)
src = emit_kernel(kernel)
dim_args = {"M": size, "N": size, "K": size}
if selected.k_splits > 1: dim_args["k_splits"] = selected.k_splits
if selected.batch_count > 1: dim_args["batch"] = selected.batch_count
prog = generate_benchmark_program(src, kernel, dim_args, num_iterations=5,
    compare_cublas=True, coarsen_cols=selected.coarsen_cols, coarsen_rows=selected.coarsen_rows,
    cublas_math_mode="default")
pathlib.Path("$TMPDIR/bench.cu").write_text(prog)
print("$TMPDIR/bench.cu", file=sys.stderr)
PY

ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | sed 's/^/sm_/')"
echo "## Compiling for $ARCH ..." >&2
nvcc -O3 --fmad=true -arch="$ARCH" -lcuda -lcublas -lcurand \
    -o "$TMPDIR/bench" "$TMPDIR/bench.cu"

echo
echo "# === ncu metrics @ ${SIZE}x${SIZE} batch=${BATCH} ==="
echo "# (kernel name 'fused_matmul' = ours; 'sgemm'/'simt_sgemm' = cuBLAS)"
echo
ncu --target-processes all \
    --metrics "$METRICS_CSV" \
    --csv \
    "$TMPDIR/bench" 2>&1 || {
    echo
    echo "## ncu failed — likely permissions. See ERR_NVGPUCTRPERM in the docs."
    exit 1
}
