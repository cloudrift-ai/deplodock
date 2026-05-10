"""Verify the lowering-chain backref ``CudaOp -> KernelOp -> TileOp -> LoopOp``.

Each lowering rule (``tileify``, ``materialize_tile``, ``lower_kernelop``)
stamps its predecessor op on the produced op's ``source`` field. From a
fully lowered CudaOp the originating LoopOp is reachable via
``cuda.source.source.source``. The chain is attribution metadata —
``Graph.structural_key`` and ``op_cache_key`` must ignore it so equivalent
kernels still dedup.
"""

from __future__ import annotations

from deplodock.compiler.backend.base import BenchmarkResult, LaunchTime
from deplodock.compiler.cache import TuningCache, op_cache_key, record_terminal
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import CUDA_PASSES, run_pipeline


def _elementwise_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), inputs=[], output=Tensor(name="x", shape=(8,), dtype="float32"), node_id="x")
    g.add_node(InputOp(), inputs=[], output=Tensor(name="z", shape=(8,), dtype="float32"), node_id="z")
    g.add_node(ElementwiseOp(op="add"), inputs=["x", "z"], output=Tensor(name="y", shape=(8,), dtype="float32"), node_id="y")
    g.inputs = ["x", "z"]
    g.outputs = ["y"]
    return g


def test_lowering_chain_reaches_loop_op() -> None:
    g = run_pipeline(_elementwise_graph(), CUDA_PASSES)
    cuda_nodes = [n for n in g.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_nodes, "pipeline should produce at least one CudaOp"
    for n in cuda_nodes:
        cuda = n.op
        assert isinstance(cuda.source, KernelOp), f"CudaOp.source must be KernelOp, got {type(cuda.source).__name__}"
        assert isinstance(cuda.source.source, TileOp), "KernelOp.source must be TileOp"
        assert isinstance(cuda.source.source.source, LoopOp), "TileOp.source must be LoopOp"


def test_source_excluded_from_structural_key() -> None:
    """Two CudaOps that differ only in the chain they came from (i.e. only
    in ``.source``) should hash equal — the cache must dedup them."""
    g = run_pipeline(_elementwise_graph(), CUDA_PASSES)
    cuda = next(n.op for n in g.nodes.values() if isinstance(n.op, CudaOp))
    other = CudaOp(
        kernel_source=cuda.kernel_source,
        kernel_name=cuda.kernel_name,
        arg_order=cuda.arg_order,
        grid=cuda.grid,
        block=cuda.block,
        smem_bytes=cuda.smem_bytes,
        tma_descriptors=cuda.tma_descriptors,
        zero_outputs=cuda.zero_outputs,
        source=None,  # different lowering chain
    )
    assert op_cache_key(cuda) == op_cache_key(other)


class _StubBackend:
    """Backend stub that yields scripted per-kernel latencies."""

    def __init__(self, per_launch_ms: list[float]) -> None:
        self.per_launch_ms = per_launch_ms
        self.calls: int = 0

    def benchmark(self, graph, *, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        del graph, warmup, num_iters
        self.calls += 1
        per_launch = [LaunchTime(idx=i, kernel_name=f"k{i}", time_ms=t) for i, t in enumerate(self.per_launch_ms)]
        return BenchmarkResult(time_ms=sum(self.per_launch_ms), num_launches=len(per_launch), per_launch=per_launch)


def test_record_terminal_inserts_cuda_perf_row() -> None:
    """A bench call inserts one ``cuda_perf`` row keyed on the CudaOp."""
    g = run_pipeline(_elementwise_graph(), CUDA_PASSES)
    cuda_node = next(n for n in g.nodes.values() if isinstance(n.op, CudaOp))
    cuda = cuda_node.op
    cache = TuningCache()  # fresh, no prior measurements
    ctx_key = Context.from_target((12, 0)).structural_key()
    backend = _StubBackend(per_launch_ms=[0.0425])  # 42.5 us

    record_terminal(g, cache, ctx_key, backend=backend)

    assert backend.calls == 1
    row = cache.cuda_perf(ctx_key, op_cache_key(cuda))
    assert row is not None
    assert row.latency_us == 42.5
    assert row.status == "ok"
