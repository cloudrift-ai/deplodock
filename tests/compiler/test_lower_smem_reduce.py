"""Per-row-reduce emission tests (was: Strategy B smem-reduce tests).

The unified emitter handles softmax / RMSNorm shapes with one thread per
output row, serial per-thread reduction (no smem cooperation, no
``__syncthreads``). Cross-thread split-K is a follow-up.

Pre-unification this module asserted the smem tree-halve pattern of
Strategy B; that path is gone. The current expectations: a flat
``tid``-guarded body with one ``acc`` per reduce block and one output
loop walking the output axis serially.
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


def test_softmax_large_thread_per_row():
    """Softmax over a 4096-wide axis: one thread per row, two serial reduce loops, one output loop."""
    source = _cuda_nodes(CudaBackend().compile(_softmax_graph((4, 4096))))[0].op.kernel_source
    # No smem / sync — split-K not implemented yet.
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    # Two reduce loops (max, sum) plus one output loop, all walking 4096.
    assert source.count("for (int k") == 2
    assert "for (int o" in source
    assert "< 4096" in source
    assert "blockIdx.x" in source


def test_softmax_small_thread_per_row():
    """Small reduce extent: same one-thread-per-row shape."""
    source = _cuda_nodes(CudaBackend().compile(_softmax_graph((4, 8))))[0].op.kernel_source
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    assert source.count("for (int k") == 2
    assert "for (int o" in source
    assert "< 8" in source
