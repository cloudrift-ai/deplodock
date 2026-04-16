"""Decompose scaled_dot_product_attention into QK^T → scale → softmax → @V.

For GQA (Grouped Query Attention) where Q has more heads than K/V, an
explicit IndexMapOp is inserted on K and V to broadcast the head dim
via integer-divide indexing: ``K[b, q_head // group_size, s, d]``.
"""

import math

from deplodock.compiler.backend.ir.expr import BinOp, Literal
from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    IndexMapOp,
    IndexSource,
    InputOp,
    ReduceOp,
    SdpaOp,
    TransposeOp,
)

GRAMMAR = [Production("root", SdpaOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    """Replace sdpa(Q, K, V, ...) with QK^T → scale → softmax → @V."""
    root = graph.nodes[match.root_node_id]
    q_id = root.inputs[0]
    k_id = root.inputs[1]
    v_id = root.inputs[2]

    q_shape = graph.nodes[q_id].output.shape
    k_shape = graph.nodes[k_id].output.shape
    out_shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    head_dim = q_shape[-1] if len(q_shape) >= 2 else 64
    seq_len = q_shape[-2] if len(q_shape) >= 3 else q_shape[-1]
    q_batch = q_shape[:-2] if len(q_shape) > 2 else ()
    k_batch = k_shape[:-2] if len(k_shape) > 2 else ()
    scores_shape = q_batch + (seq_len, seq_len)

    frag = Graph()

    # InputOp sentinels for all external references.
    # Q, K, V are always referenced.
    ext_ids = {q_id, k_id, v_id}
    for eid in sorted(ext_ids):
        frag.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(graph.nodes[eid].output.name, graph.nodes[eid].output.shape, graph.nodes[eid].output.dtype),
            node_id=eid,
        )

    # K^T: transpose last two dims.
    kt_shape = k_batch + (head_dim, seq_len) if isinstance(head_dim, int) else k_shape
    kt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[k_id],
        output=Tensor(f"{name}_kt", kt_shape, dtype),
    )

    # GQA: when Q has more heads than K, insert an IndexMapOp on K^T to
    # broadcast via integer-divide: K^T[b, q_head // group_size, d, s].
    if q_batch and k_batch and len(q_batch) == len(k_batch):
        q_heads = q_batch[-1] if isinstance(q_batch[-1], int) else None
        k_heads = k_batch[-1] if isinstance(k_batch[-1], int) else None
        if q_heads and k_heads and q_heads > k_heads and q_heads % k_heads == 0:
            group_size = q_heads // k_heads
            gqa_out_shape = q_batch + (head_dim, seq_len)
            ndim = len(gqa_out_shape)
            coord_map = []
            for d in range(ndim):
                p = placeholder(d)
                head_axis = len(q_batch) - 1
                if d == head_axis:
                    coord_map.append(BinOp("/", p, Literal(group_size, "int")))
                else:
                    coord_map.append(p)
            kt_id = frag.add_node(
                op=IndexMapOp(
                    out_shape=gqa_out_shape,
                    sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),),
                ),
                inputs=[kt_id],
                output=Tensor(f"{name}_kt_gqa", gqa_out_shape, dtype),
            )

    # QK^T: matmul(Q, K^T) — unsqueeze for broadcast-compatible mul + reduce.
    from deplodock.compiler.rules.decomposition._matmul_helpers import matmul_unsqueeze

    q_eff_shape = tuple(graph.nodes[q_id].output.shape)
    kt_eff_shape = tuple(frag.nodes[kt_id].output.shape)
    q_unsq, kt_unsq, qk_mul_shape, qk_k_axis = matmul_unsqueeze(q_eff_shape, kt_eff_shape)

    q_unsq_id = frag.add_node(op=q_unsq, inputs=[q_id], output=Tensor(f"{name}_q_unsq", q_unsq.out_shape, dtype))
    kt_unsq_id = frag.add_node(op=kt_unsq, inputs=[kt_id], output=Tensor(f"{name}_kt_unsq", kt_unsq.out_shape, dtype))
    qk_ew_id = frag.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[q_unsq_id, kt_unsq_id],
        output=Tensor(f"{name}_qk_ew", qk_mul_shape, dtype),
    )
    qk_id = frag.add_node(
        op=ReduceOp(fn="sum", axis=qk_k_axis),
        inputs=[qk_ew_id],
        output=Tensor(f"{name}_qk", scores_shape, dtype),
    )

    # Scale constant: 1/sqrt(head_dim)
    scale_value = 1.0 / math.sqrt(head_dim) if isinstance(head_dim, int) else None
    scale_const_id = frag.add_node(
        op=ConstantOp(name=f"{name}_scale", value=scale_value),
        inputs=[],
        output=Tensor(f"{name}_scale", (1,), dtype),
    )
    scaled_id = frag.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[qk_id, scale_const_id],
        output=Tensor(f"{name}_scaled", scores_shape, dtype),
    )

    # Softmax: max → sub → exp → sum → div
    max_id = frag.add_node(
        op=ReduceOp(fn="max", axis=-1),
        inputs=[scaled_id],
        output=Tensor(f"{name}_max", scores_shape[:-1] + (1,) if scores_shape else (1,), dtype),
    )
    sub_id = frag.add_node(
        op=ElementwiseOp(fn="sub"),
        inputs=[scaled_id, max_id],
        output=Tensor(f"{name}_shifted", scores_shape, dtype),
    )
    exp_id = frag.add_node(
        op=ElementwiseOp(fn="exp"),
        inputs=[sub_id],
        output=Tensor(f"{name}_exp", scores_shape, dtype),
    )
    sum_id = frag.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=[exp_id],
        output=Tensor(f"{name}_sum", scores_shape[:-1] + (1,) if scores_shape else (1,), dtype),
    )
    softmax_id = frag.add_node(
        op=ElementwiseOp(fn="div"),
        inputs=[exp_id, sum_id],
        output=Tensor(f"{name}_softmax", scores_shape, dtype),
    )

    # Softmax @ V: matmul(softmax, V) — decomposed as mul + reduce sum.
    # GQA: V needs the same head broadcast as K.
    v_shape = graph.nodes[v_id].output.shape
    v_batch = v_shape[:-2] if len(v_shape) > 2 else ()
    actual_v_id = v_id
    if q_batch and v_batch:
        q_heads_v = q_batch[-1] if isinstance(q_batch[-1], int) else None
        v_heads = v_batch[-1] if isinstance(v_batch[-1], int) else None
        if q_heads_v and v_heads and q_heads_v > v_heads and q_heads_v % v_heads == 0:
            group_size_v = q_heads_v // v_heads
            # Replace V's head dim with Q's head count; keep V's other batch dims.
            head_axis_v = len(v_batch) - 1
            gqa_v_shape = tuple(v_batch[:head_axis_v]) + (q_heads_v,) + tuple(v_shape[-2:])
            coord_map_v = []
            for d in range(len(v_shape)):
                p = placeholder(d)
                if d == head_axis_v:
                    coord_map_v.append(BinOp("/", p, Literal(group_size_v, "int")))
                else:
                    coord_map_v.append(p)
            actual_v_id = frag.add_node(
                op=IndexMapOp(
                    out_shape=gqa_v_shape,
                    sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map_v)),),
                ),
                inputs=[v_id],
                output=Tensor(f"{name}_v_gqa", gqa_v_shape, dtype),
            )

    s_eff_shape = tuple(frag.nodes[softmax_id].output.shape)
    v_eff_shape = tuple(frag.nodes[actual_v_id].output.shape)
    s_unsq, v_unsq, sv_mul_shape, sv_k_axis = matmul_unsqueeze(s_eff_shape, v_eff_shape)

    s_unsq_id = frag.add_node(op=s_unsq, inputs=[softmax_id], output=Tensor(f"{name}_s_unsq", s_unsq.out_shape, dtype))
    v_unsq_id = frag.add_node(op=v_unsq, inputs=[actual_v_id], output=Tensor(f"{name}_v_unsq", v_unsq.out_shape, dtype))
    sv_ew_id = frag.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[s_unsq_id, v_unsq_id],
        output=Tensor(f"{name}_sv_ew", sv_mul_shape, dtype),
    )
    sv_id = frag.add_node(
        op=ReduceOp(fn="sum", axis=sv_k_axis),
        inputs=[sv_ew_id],
        output=Tensor(name, out_shape, dtype),
    )

    frag.outputs = [sv_id]
    return frag
