"""Higher-level fragment-construction helpers for decomposition rules.

The lower-level primitives (``broadcast_to``, ``squeeze_axis``,
``matmul_unsqueeze``) live in their own modules and are re-exported here so
rules only need a single import.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, placeholder
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource, ReduceOp
from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to, squeeze_axis
from deplodock.compiler.pipeline.passes.frontend.decomposition._matmul_helpers import matmul_unsqueeze

__all__ = [
    "broadcast_to",
    "const_bc",
    "gqa_broadcast",
    "matmul_decompose",
    "matmul_unsqueeze",
    "open_fragment",
    "reduction_shape",
    "single_indexmap",
    "softmax_decompose",
    "squeeze_axis",
]


def open_fragment(graph: Graph, exts: Iterable[str | Node]) -> Graph:
    """Return a fresh fragment with InputOp sentinels for every ext.

    ``exts`` may be a mix of node ids and ``Node`` objects — Nodes get
    their ``id`` extracted, ids are looked up in ``graph``.
    """
    frag = Graph()
    ids = sorted({e.id if isinstance(e, Node) else e for e in exts})
    for eid in ids:
        t = graph.nodes[eid].output
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(t.name, t.shape, t.dtype), node_id=eid)
    return frag


def single_indexmap(frag: Graph, x_id: str, *, out_shape: tuple, coord_map, name: str, dtype) -> str:
    """Wrap a single-source IndexMapOp with the given coord_map."""
    return frag.add_node(
        op=IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[x_id],
        output=Tensor(name, tuple(out_shape), dtype),
    )


def reduction_shape(shape: tuple, axis: int) -> tuple:
    """Replace ``shape[axis]`` with 1 (keepdim=True reduction shape)."""
    a = axis if axis >= 0 else len(shape) + axis
    return tuple(shape[:a]) + (1,) + tuple(shape[a + 1 :])


def const_bc(frag: Graph, *, name: str, value, target_shape: tuple, dtype) -> str:
    """Add a scalar ConstantOp and broadcast it to ``target_shape``."""
    cid = frag.add_node(
        op=ConstantOp(name=name, value=value),
        inputs=[],
        output=Tensor(name, (1,), dtype),
    )
    return broadcast_to(frag, cid, tuple(target_shape))


def matmul_decompose(frag: Graph, a_id: str, b_id: str, *, name: str, dtype) -> str:
    """Decompose a matmul into unsqueeze → broadcast → multiply → reduce_sum → squeeze.

    Returns the squeezed output node id.
    """
    a_shape = tuple(frag.nodes[a_id].output.shape)
    b_shape = tuple(frag.nodes[b_id].output.shape)
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(a_shape, b_shape)
    a_uid = frag.add_node(op=a_unsq, inputs=[a_id], output=Tensor(f"{name}_a_unsq", a_unsq.out_shape, dtype))
    b_uid = frag.add_node(op=b_unsq, inputs=[b_id], output=Tensor(f"{name}_b_unsq", b_unsq.out_shape, dtype))
    a_bc = broadcast_to(frag, a_uid, mul_shape)
    b_bc = broadcast_to(frag, b_uid, mul_shape)
    ew = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[a_bc, b_bc],
        output=Tensor(f"{name}_ew", mul_shape, dtype),
    )
    red_shape = reduction_shape(mul_shape, k_axis)
    red = frag.add_node(
        op=ReduceOp(op="sum", axis=k_axis),
        inputs=[ew],
        output=Tensor(f"{name}_reduce", red_shape, dtype),
    )
    return squeeze_axis(frag, red, k_axis, out_name=name)


def softmax_decompose(frag: Graph, x_id: str, axis: int, *, name: str, dtype) -> str:
    """Decompose softmax into max → sub → exp → sum → div. Returns the div node id."""
    out_shape = tuple(frag.nodes[x_id].output.shape)
    red_shape = reduction_shape(out_shape, axis) if out_shape else (1,)
    max_id = frag.add_node(
        op=ReduceOp(op="maximum", axis=axis),
        inputs=[x_id],
        output=Tensor(f"{name}_max", red_shape, dtype),
    )
    max_bc = broadcast_to(frag, max_id, out_shape)
    sub_id = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[x_id, max_bc],
        output=Tensor(f"{name}_shifted", out_shape, dtype),
    )
    exp_id = frag.add_node(
        op=ElementwiseOp(op="exp"),
        inputs=[sub_id],
        output=Tensor(f"{name}_exp", out_shape, dtype),
    )
    sum_id = frag.add_node(
        op=ReduceOp(op="sum", axis=axis),
        inputs=[exp_id],
        output=Tensor(f"{name}_sum", red_shape, dtype),
    )
    sum_bc = broadcast_to(frag, sum_id, out_shape)
    return frag.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[exp_id, sum_bc],
        output=Tensor(name, out_shape, dtype),
    )


def gqa_broadcast(frag: Graph, src_id: str, *, target_shape: tuple, head_axis: int, group_size: int, name: str, dtype) -> str:
    """Broadcast a head-axis via integer-divide indexing: out[..., h, ...] = src[..., h // g, ...]."""
    coord_map = []
    for d in range(len(target_shape)):
        p = placeholder(d)
        coord_map.append(BinaryExpr("/", p, Literal(group_size, "int")) if d == head_axis else p)
    return single_indexmap(frag, src_id, out_shape=target_shape, coord_map=coord_map, name=name, dtype=dtype)
