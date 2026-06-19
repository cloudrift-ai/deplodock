"""Higher-level fragment-construction helpers for decomposition rules.

The lower-level primitives (``broadcast_to``, ``squeeze_axis``,
``matmul_unsqueeze``) live in their own modules and are re-exported here so
rules only need a single import.

Helpers take and return ``Node`` values — they read shape/dtype straight off
the node, so callers don't re-look up ``frag.nodes[id].output``. ``Graph.add_node``
accepts ``Node | str`` for ``inputs``, so passing a Node through to a raw
``add_node`` call works too.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, placeholder
from deplodock.compiler.ir.frontend.ir import TransposeOp
from deplodock.compiler.ir.tensor.ir import CastOp, ElementwiseOp, IndexMapOp, IndexSource, ReduceOp
from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to, squeeze_axis
from deplodock.compiler.pipeline.passes.frontend.decomposition._matmul_helpers import matmul_unsqueeze

__all__ = [
    "broadcast_to",
    "const_bc",
    "dequant_decompose",
    "gqa_broadcast",
    "matmul_decompose",
    "matmul_unsqueeze",
    "open_fragment",
    "reduction_shape",
    "single_indexmap",
    "softmax_decompose",
    "squeeze_axis",
]


def _node(frag: Graph, x: Node | str) -> Node:
    """Coerce a Node-or-id to the Node in ``frag``."""
    return x if isinstance(x, Node) else frag.nodes[x]


def open_fragment(graph: Graph, exts: Iterable[Node | str]) -> Graph:
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


def single_indexmap(frag: Graph, x: Node | str, *, out_shape: tuple, coord_map, name: str, dtype: str | None = None) -> Node:
    """Wrap a single-source IndexMapOp with the given coord_map."""
    x = _node(frag, x)
    dtype = dtype or x.output.dtype
    nid = frag.add_node(
        op=IndexMapOp(out_shape=tuple(out_shape), sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[x],
        output=Tensor(name, tuple(out_shape), dtype),
    )
    return frag.nodes[nid]


def reduction_shape(shape: tuple, axis: int) -> tuple:
    """Replace ``shape[axis]`` with 1 (keepdim=True reduction shape)."""
    a = axis if axis >= 0 else len(shape) + axis
    return tuple(shape[:a]) + (1,) + tuple(shape[a + 1 :])


def const_bc(frag: Graph, *, name: str, value, target_shape: tuple, dtype: str) -> Node:
    """Add a scalar ConstantOp and broadcast it to ``target_shape``."""
    cid = frag.add_node(
        op=ConstantOp(name=name, value=value),
        inputs=[],
        output=Tensor(name, (1,), dtype),
    )
    return broadcast_to(frag, cid, tuple(target_shape))


def matmul_decompose(frag: Graph, a: Node | str, b: Node | str, *, name: str, dtype: str | None = None) -> Node:
    """Decompose a matmul into unsqueeze → broadcast → multiply → reduce_sum → squeeze.

    Returns the squeezed output node.
    """
    a, b = _node(frag, a), _node(frag, b)
    dtype = dtype or a.output.dtype
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(a.output.shape, b.output.shape)
    a_uid = frag.add_node(op=a_unsq, inputs=[a], output=Tensor(f"{name}_a_unsq", a_unsq.out_shape, dtype))
    b_uid = frag.add_node(op=b_unsq, inputs=[b], output=Tensor(f"{name}_b_unsq", b_unsq.out_shape, dtype))
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


def softmax_decompose(frag: Graph, x: Node | str, axis: int, *, name: str, dtype: str | None = None) -> Node:
    """Decompose softmax into max → sub → exp → sum → div. Returns the div node."""
    x = _node(frag, x)
    dtype = dtype or x.output.dtype
    out_shape = tuple(x.output.shape)
    red_shape = reduction_shape(out_shape, axis) if out_shape else (1,)
    max_id = frag.add_node(
        op=ReduceOp(op="maximum", axis=axis),
        inputs=[x],
        output=Tensor(f"{name}_max", red_shape, dtype),
    )
    max_bc = broadcast_to(frag, max_id, out_shape)
    sub_id = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[x, max_bc],
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
    div_id = frag.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[exp_id, sum_bc],
        output=Tensor(name, out_shape, dtype),
    )
    return frag.nodes[div_id]


def gqa_broadcast(
    frag: Graph, src: Node | str, *, target_shape: tuple, head_axis: int, group_size: int, name: str, dtype: str | None = None
) -> Node:
    """Broadcast a head-axis via integer-divide indexing: out[..., h, ...] = src[..., h // g, ...]."""
    src = _node(frag, src)
    dtype = dtype or src.output.dtype
    coord_map = []
    for d in range(len(target_shape)):
        p = placeholder(d)
        coord_map.append(BinaryExpr("/", p, Literal(group_size, "int")) if d == head_axis else p)
    return single_indexmap(frag, src, out_shape=target_shape, coord_map=coord_map, name=name, dtype=dtype)


def _unpack_nibbles(frag: Graph, packed: Node | str, *, axis: int, pack_factor: int, num_bits: int, out_shape: tuple, name: str) -> Node:
    """Unpack ``pack_factor`` unsigned nibbles per int32 along ``axis``.

    ``pack_factor`` fixed-shift lanes ``lane_i = (packed >> num_bits*i) & mask``
    (each an i32 tensor the shape of ``packed``), assembled into ``out_shape``
    by a multi-source ``IndexMapOp``: output coord ``j`` along ``axis`` reads
    ``lane_{j % pack_factor}`` at ``j // pack_factor`` (the interleaved
    ``i::pack_factor`` layout). Stays integer end-to-end. The shift / mask
    immediates flow as i32 scalar ``ConstantOp``s (inlined as int literals at
    lowering); no coordinate-dependent shift, so Phase 0 needs no IR gap.
    """
    packed = _node(frag, packed)
    packed_shape = tuple(packed.output.shape)
    mask = (1 << num_bits) - 1

    lanes: list[Node] = []
    for i in range(pack_factor):
        shift_bc = const_bc(frag, name=f"{name}_sh{i}", value=num_bits * i, target_shape=packed_shape, dtype="i32")
        shifted = frag.add_node(
            op=ElementwiseOp(op="right_shift"),
            inputs=[packed, shift_bc],
            output=Tensor(f"{name}_shifted{i}", packed_shape, "i32"),
        )
        mask_bc = const_bc(frag, name=f"{name}_mk{i}", value=mask, target_shape=packed_shape, dtype="i32")
        lane = frag.add_node(
            op=ElementwiseOp(op="bitwise_and"),
            inputs=[shifted, mask_bc],
            output=Tensor(f"{name}_lane{i}", packed_shape, "i32"),
        )
        lanes.append(frag.nodes[lane])

    ndim = len(out_shape)
    sources: list[IndexSource] = []
    for i in range(pack_factor):
        coord_map = tuple(
            BinaryExpr("/", placeholder(d), Literal(pack_factor, "int")) if d == axis else placeholder(d) for d in range(ndim)
        )
        # Last lane is the default (else) branch — no select; earlier lanes fire
        # on ``coord[axis] % pack_factor == i``.
        sel = None
        if i != pack_factor - 1:
            sel = BinaryExpr("==", BinaryExpr("%", placeholder(axis), Literal(pack_factor, "int")), Literal(i, "int"))
        sources.append(IndexSource(input_idx=i, coord_map=coord_map, select=sel))

    nid = frag.add_node(
        op=IndexMapOp(out_shape=tuple(out_shape), sources=tuple(sources)),
        inputs=lanes,
        output=Tensor(name, tuple(out_shape), "i32"),
    )
    return frag.nodes[nid]


def dequant_decompose(
    frag: Graph,
    x: Node | str,
    weight_packed: Node | str,
    scale: Node | str,
    zp: Node | str | None,
    *,
    scheme,
    matmul_name: str,
    out_dtype,
) -> Node:
    """Build the W4A16 unpack → dequant → transpose → matmul cone.

    Mirrors ``matmul_decompose``'s role for ``045_dequant_linear``: returns the
    matmul output node named ``matmul_name`` (the rule adds bias after).

    Asymmetric (``symmetric=False``, the verified format): two independent
    unsigned-nibble unpacks — the weight along the in/K axis, the zero-point
    along OUT — then ``(nibble − zp)`` with zp group-broadcast on K, an
    int→fp16 ``CastOp``, the group-broadcast ``· scale``, a transpose to
    ``[in, out]``, and the matmul. Symmetric subtracts the constant midpoint
    ``2**(num_bits-1)`` instead of the unpacked zp (never both).
    """
    x = _node(frag, x)
    wp = _node(frag, weight_packed)
    sc = _node(frag, scale)
    per = scheme.pack_factor
    num_bits = scheme.num_bits
    g = scheme.group_size
    weight_dtype = sc.output.dtype

    wp_shape = tuple(wp.output.shape)
    out_features = wp_shape[0]
    in_features = wp_shape[1] * per
    weight_int_shape = (out_features, in_features)

    # 1. Unpack the weight nibbles along the in/K axis (packed_dim == 1).
    nibble = _unpack_nibbles(frag, wp, axis=1, pack_factor=per, num_bits=num_bits, out_shape=weight_int_shape, name=f"{matmul_name}_wq")

    # 2. (nibble − zp) [asymmetric]  /  (nibble − 8) [symmetric].
    if scheme.symmetric:
        offset_bc = const_bc(frag, name=f"{matmul_name}_off", value=1 << (num_bits - 1), target_shape=weight_int_shape, dtype="i32")
        diff_b = offset_bc
    else:
        zp_node = _node(frag, zp)
        zp_shape = tuple(zp_node.output.shape)
        zp_groups = zp_shape[1]
        # Zero-point is int4-packed along OUT (axis 0) — a distinct axis from the
        # weight's in-axis pack.
        zp_unpacked = _unpack_nibbles(
            frag, zp_node, axis=0, pack_factor=per, num_bits=num_bits, out_shape=(out_features, zp_groups), name=f"{matmul_name}_zq"
        )
        # Broadcast zp on K (group-indexed): zp_bc[o, k] = zp_unpacked[o, k // g].
        diff_b = single_indexmap(
            frag,
            zp_unpacked,
            out_shape=weight_int_shape,
            coord_map=(placeholder(0), BinaryExpr("/", placeholder(1), Literal(g, "int"))),
            name=f"{matmul_name}_zpbc",
            dtype="i32",
        )
    diff = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[nibble, diff_b],
        output=Tensor(f"{matmul_name}_diff", weight_int_shape, "i32"),
    )

    # 3. int → fp16 cast (the dequant boundary).
    cast = frag.add_node(
        op=CastOp(target_dtype=weight_dtype.name),
        inputs=[diff],
        output=Tensor(f"{matmul_name}_wf", weight_int_shape, weight_dtype),
    )

    # 4. Group-broadcast scale on K: scale_bc[o, k] = scale[o, k // g].
    scale_bc = single_indexmap(
        frag,
        sc,
        out_shape=weight_int_shape,
        coord_map=(placeholder(0), BinaryExpr("/", placeholder(1), Literal(g, "int"))),
        name=f"{matmul_name}_scbc",
        dtype=weight_dtype,
    )

    # 5. w = cast · scale_bc (fp16).
    w = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[cast, scale_bc],
        output=Tensor(f"{matmul_name}_w", weight_int_shape, weight_dtype),
    )

    # 6. Transpose [out, in] → [in, out] (matmul wants A[…,K] @ B[K,N]).
    wt = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[w],
        output=Tensor(f"{matmul_name}_wt", (in_features, out_features), weight_dtype),
    )

    # 7. x @ wt.
    return matmul_decompose(frag, x, wt, name=matmul_name, dtype=out_dtype)
