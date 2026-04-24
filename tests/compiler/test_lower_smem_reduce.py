"""Strategy B (smem reduce) emission tests.

The lowering rule picks Strategy B when the reduce extent is ≥ BLOCK and the
outer-free numel is small enough to leave room for block-level parallelism.
Under that strategy the kernel must contain a shared-memory tile, at least
one ``__syncthreads()``, and a tree-halving pattern that broadcasts the final
reduction result via ``smem[0]``.

Small-shape tests (existing ``test_emit.py``) cover Strategy A — this module
only exercises the smem branch.
"""

from __future__ import annotations

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp


def _cuda_nodes(g: Graph) -> list:
    return [n for n in g.nodes.values() if isinstance(n.op, CudaOp)]


def _softmax_large_graph() -> Graph:
    """Softmax over a 4096-wide axis — forces Strategy B in the lowering pass."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 4096)), node_id="x")
    g.add_node(op=ConstantOp(name="axis", value=-1.0), inputs=[], output=Tensor("axis", (1,)), node_id="axis")
    g.add_node(
        op=ElementwiseOp(op="softmax"),
        inputs=["x", "axis"],
        output=Tensor("y", (4, 4096)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_softmax_large_uses_shared_memory():
    source = _cuda_nodes(CudaBackend().compile(_softmax_large_graph()))[0].op.kernel_source
    assert "__shared__" in source, f"expected smem tile, got:\n{source}"
    assert "__syncthreads" in source
    assert "smem0[0]" in source
    assert "blockIdx.x" in source


def test_softmax_large_emits_tree_halve_rounds():
    """BLOCK=256 → log2(256)=8 halving rounds; each stride should show up in source."""
    source = _cuda_nodes(CudaBackend().compile(_softmax_large_graph()))[0].op.kernel_source
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        assert f"+ {stride}]" in source, f"missing halve stride {stride} in:\n{source}"


def test_small_reduce_stays_serial():
    """Small reduce extent (< BLOCK) must keep the one-thread-per-output (serial) shape."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ConstantOp(name="axis", value=-1.0), inputs=[], output=Tensor("axis", (1,)), node_id="axis")
    g.add_node(
        op=ElementwiseOp(op="softmax"),
        inputs=["x", "axis"],
        output=Tensor("y", (4, 8)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    source = _cuda_nodes(CudaBackend().compile(g))[0].op.kernel_source
    assert "__shared__" not in source
    assert "__syncthreads" not in source
