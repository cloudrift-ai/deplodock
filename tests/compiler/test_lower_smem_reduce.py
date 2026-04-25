"""Multi-phase cooperative-reduction emission tests.

The cooperative-reduce strategy rewrites a fused softmax / RMSNorm
``Tile`` into a per-row-block kernel: each CUDA block owns one output
row, threads inside the block cooperate on the reduction axis via
``__shared__`` partials + tree-halve, and broadcast the result to
subsequent phases via a smem ``Load`` at index 0.

The threshold gate (K ≥ ``COOP_THRESHOLD`` = 128) means small reduction
extents stay on the per-thread serial path.
"""

from __future__ import annotations

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.frontend.ir import SoftmaxOp


def _cuda_nodes(g: Graph) -> list:
    return [n for n in g.nodes.values() if isinstance(n.op, CudaOp)]


def _softmax_graph(shape: tuple[int, int]) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", shape), node_id="x")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["x"], output=Tensor("y", shape), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_softmax_cooperative_above_threshold():
    """Softmax over K=4096: cooperative path — two ``__shared__`` accumulator
    buffers, two tree-halves, broadcast loads between phases."""
    source = _cuda_nodes(CudaBackend().compile(_softmax_graph((4, 4096))))[0].op.kernel_source
    # Two cooperative reductions ⇒ two smem buffers.
    assert source.count("__shared__ float ") == 2
    # Each phase has at least one barrier; tree-halve has another inside its loop.
    assert source.count("__syncthreads();") >= 4
    # Two tree-halves (one per reduction phase).
    assert source.count("for (int s = 128; s > 0; s >>= 1)") == 2
    # All loops over k stride by BLOCK = 256 — two reductions + one output.
    assert source.count("+= 256") == 3
    assert "< 4096" in source
    # One CUDA block per row.
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


def test_softmax_serial_below_threshold():
    """K=8 < COOP_THRESHOLD: stays on the per-thread serial path — no smem,
    no syncs, one thread per output row."""
    source = _cuda_nodes(CudaBackend().compile(_softmax_graph((4, 8))))[0].op.kernel_source
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    # Two reduce loops + one output loop = 3 total `for` headers.
    assert source.count("for (int ") == 3
    assert "< 8" in source
