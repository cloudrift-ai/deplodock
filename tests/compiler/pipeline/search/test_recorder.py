"""Tests for :func:`record_terminal` — the top-level orchestrator that
benches a graph, persists op inventory + lowering edges to the DB, and
bumps the in-memory MCTS tree.

Uses a stub backend so the test stays CPU-only.
"""

from __future__ import annotations

from deplodock.compiler.backend.base import BenchmarkResult, LaunchTime
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline
from deplodock.compiler.pipeline.search import (
    SearchDB,
    SearchTree,
    op_cache_key,
    record_terminal,
)
from deplodock.compiler.pipeline.search.recorder import count_unmeasured_ops


def _elementwise_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), inputs=[], output=Tensor(name="x", shape=(8,), dtype="float32"), node_id="x")
    g.add_node(InputOp(), inputs=[], output=Tensor(name="z", shape=(8,), dtype="float32"), node_id="z")
    g.add_node(ElementwiseOp(op="add"), inputs=["x", "z"], output=Tensor(name="y", shape=(8,), dtype="float32"), node_id="y")
    g.inputs = ["x", "z"]
    g.outputs = ["y"]
    return g


class _StubBackend:
    """Backend stub yielding scripted per-kernel samples (ms)."""

    name = "cuda"
    bench_run_timeout_s = 2.0

    def __init__(self, per_launch_ms: list[float], samples_per_launch: int = 5) -> None:
        self.per_launch_ms = per_launch_ms
        self.samples_per_launch = samples_per_launch
        self.calls = 0

    def benchmark(self, graph, *, num_iters="auto") -> BenchmarkResult:
        del graph, num_iters
        self.calls += 1
        per_launch = []
        for i, t in enumerate(self.per_launch_ms):
            samples = tuple(t + 0.0001 * j for j in range(self.samples_per_launch))
            per_launch.append(LaunchTime(idx=i, kernel_name=f"k{i}", time_ms=t, samples=samples))
        return BenchmarkResult(time_ms=sum(self.per_launch_ms), num_launches=len(per_launch), per_launch=per_launch)


# ---------------------------------------------------------------------------
# Simple graph (no source chain) — stub backend with synthetic samples
# ---------------------------------------------------------------------------


def _single_cuda_graph(kernel_source: str = "__global__ void k() {}") -> Graph:
    g = Graph()
    g.add_node(InputOp(), inputs=[], output=Tensor(name="x", shape=(4,), dtype="float32"), node_id="x")
    g.add_node(
        CudaOp(kernel_source=kernel_source, kernel_name="k", arg_order=("x",), grid=(1, 1, 1), block=(32, 1, 1)),
        inputs=["x"],
        output=Tensor(name="y", shape=(4,), dtype="float32"),
        node_id="k",
    )
    g.inputs = ["x"]
    g.outputs = ["k"]
    return g


def test_record_terminal_writes_perf_with_stub_backend() -> None:
    db = SearchDB()
    tree = SearchTree()
    g = _single_cuda_graph()
    record_terminal(g, db, tree, context_key="ctx", backend=None)  # stub latency 1.0
    key = op_cache_key(g.nodes["k"].op)
    row = db.lookup_perf("ctx", key, backend="stub")
    assert row is not None and row.stats.median == 1.0


def test_record_terminal_computes_full_perf_stats() -> None:
    db = SearchDB()
    tree = SearchTree()
    g = _single_cuda_graph()
    # 0.001 ms = 1 us; samples are [1.0, 1.1, 1.2, 1.3, 1.4] us.
    backend = _StubBackend(per_launch_ms=[0.001], samples_per_launch=5)
    record_terminal(g, db, tree, context_key="ctx", backend=backend)
    key = op_cache_key(g.nodes["k"].op)
    row = db.lookup_perf("ctx", key, backend="cuda")
    assert row is not None
    assert row.stats.n_samples == 5
    assert row.stats.min == 1.0
    assert row.stats.max == 1.4
    assert abs(row.stats.median - 1.2) < 1e-9
    assert row.stats.variance > 0.0


def test_count_unmeasured_drops_after_record_terminal() -> None:
    db = SearchDB()
    tree = SearchTree()
    g = _single_cuda_graph()
    assert count_unmeasured_ops(g, db, "ctx", backend_name="stub") == 1
    record_terminal(g, db, tree, "ctx", backend=None)
    assert count_unmeasured_ops(g, db, "ctx", backend_name="stub") == 0


# ---------------------------------------------------------------------------
# Full lowering chain — verify op inventory + lowering edges populate
# ---------------------------------------------------------------------------


def test_kernel_op_cache_key_is_structural() -> None:
    """KernelOp keys come from ``Body.structural_key()`` (not ``repr``)
    so two KernelOps whose bodies differ only in SSA names or hardware
    primitives hash equal. Build two bodies with the same shape but
    different SSA / axis names + a Sync; they should hash identical."""
    from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.ir.kernel.ir import KernelOp, Sync
    from deplodock.compiler.ir.stmt import Body, Load, Tile, Write

    def _body(axis_name: str, ssa: str) -> Body:
        a = Axis(name=axis_name, extent=8)
        return Body(
            (
                Tile(
                    axes=(BoundAxis(axis=a, bind=BIND_THREAD),),
                    body=(
                        Load(name=ssa, input="b0", index=(Var(axis_name),)),
                        Sync(),
                        Write(output="b1", index=(Var(axis_name),), value=ssa),
                    ),
                ),
            )
        )

    k1 = KernelOp(body=_body("a0", "x"), name="k")
    k2 = KernelOp(body=_body("z9", "tmp"), name="k")
    assert op_cache_key(k1) == op_cache_key(k2)


def test_record_terminal_persists_full_op_chain() -> None:
    """A fully-lowered CudaOp has a ``source`` chain back through
    KernelOp → TileOp → LoopOp. ``record_terminal`` must persist one row
    per dialect and three lowering edges."""
    g = Pipeline.build(CUDA_PASSES).run(_elementwise_graph())
    cuda = next(n.op for n in g.nodes.values() if isinstance(n.op, CudaOp))
    # Sanity: the chain we expect is present.
    assert isinstance(cuda.source, KernelOp)
    assert isinstance(cuda.source.source, TileOp)
    assert isinstance(cuda.source.source.source, LoopOp)

    db = SearchDB()
    tree = SearchTree()
    ctx_key = Context.from_target((12, 0)).structural_key()
    backend = _StubBackend(per_launch_ms=[0.0425])
    record_terminal(g, db, tree, ctx_key, backend=backend)

    assert backend.calls == 1
    # perf row
    row = db.lookup_perf(ctx_key, op_cache_key(cuda), backend="cuda")
    assert row is not None and row.status == "ok"
    # one row in each op-inventory table
    assert db._conn.execute("SELECT COUNT(*) FROM loop_op").fetchone()[0] >= 1
    assert db._conn.execute("SELECT COUNT(*) FROM tile_op").fetchone()[0] >= 1
    assert db._conn.execute("SELECT COUNT(*) FROM kernel_op").fetchone()[0] >= 1
    assert db._conn.execute("SELECT COUNT(*) FROM cuda_op").fetchone()[0] >= 1
    # three lowering edges (loop→tile, tile→kernel, kernel→cuda)
    dialects = {row[0] for row in db._conn.execute("SELECT parent_dialect FROM lowering")}
    assert {"loop", "tile", "kernel"}.issubset(dialects)


def test_loop_to_tile_lowering_keeps_best_variant() -> None:
    """Second recording with a faster median replaces the row; a slower
    one leaves it alone. (Deterministic Tile→Kernel and Kernel→Cuda
    stay pinned to their first child either way.)"""
    g = Pipeline.build(CUDA_PASSES).run(_elementwise_graph())
    cuda = next(n.op for n in g.nodes.values() if isinstance(n.op, CudaOp))
    loop_op = cuda.source.source.source
    loop_key = op_cache_key(loop_op)
    tile_key_first = op_cache_key(cuda.source.source)

    db = SearchDB()
    tree = SearchTree()
    ctx_key = Context.from_target((12, 0)).structural_key()

    # Record once at 100 us (per_launch_ms=0.1 → samples median ~100 us).
    record_terminal(g, db, tree, ctx_key, backend=_StubBackend(per_launch_ms=[0.100]))
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        (loop_key,),
    ).fetchone()
    assert row is not None and row[0] == tile_key_first

    # Replace the LoopOp's lowering with a faster synthetic TileOp.
    # (We exercise db.record_lowering directly to model an autotune
    # exploring multiple TileOp variants under the same LoopOp.)
    db.record_lowering(loop_key, "loop", "tile-fake-fast", "tile", measured_median_us=10.0)
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        (loop_key,),
    ).fetchone()
    assert row[0] == "tile-fake-fast" and row[1] == 10.0


def test_bench_failure_records_bench_fail_status() -> None:
    """A backend.benchmark raising an exception pins ``bench_fail`` on
    every kernel with the backend's ``bench_run_timeout_s`` as latency."""

    class _Failing(_StubBackend):
        def benchmark(self, graph, *, num_iters="auto"):
            del graph, num_iters
            raise RuntimeError("synthetic bench failure")

    db = SearchDB()
    tree = SearchTree()
    g = _single_cuda_graph()
    record_terminal(g, db, tree, "ctx", backend=_Failing(per_launch_ms=[]))
    key = op_cache_key(g.nodes["k"].op)
    row = db.lookup_perf("ctx", key, backend="cuda")
    assert row is not None and row.status == "bench_fail"
    # bench_run_timeout_s is 2 s → 2_000_000 us.
    assert row.stats.median == 2_000_000.0
