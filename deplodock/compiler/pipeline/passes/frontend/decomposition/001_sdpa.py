"""Decompose scaled_dot_product_attention into QK^T → scale [→ mask] → softmax → @V.

For GQA (Grouped Query Attention) where Q has more heads than K/V, an
explicit IndexMapOp is inserted on K and V to broadcast the head dim
via integer-divide indexing: ``K[b, q_head // group_size, s, d]``.
"""

import math

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, placeholder
from deplodock.compiler.ir.frontend.ir import SdpaOp, TransposeOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    const_bc,
    gqa_broadcast,
    matmul_decompose,
    open_fragment,
    softmax_decompose,
)

PATTERN = [Pattern("root", SdpaOp)]


def _maybe_gqa(frag: Graph, src_id: str, q_batch: tuple, src_batch: tuple, target_last_dims: tuple, *, name: str, dtype) -> str:
    """Broadcast src's head axis to match q's head count via integer-divide indexing.

    Returns src_id unchanged when there is no GQA mismatch. Head axis is the last
    batch dim on each side; ranks may differ (V's prefix is preserved).
    """
    if not (q_batch and src_batch):
        return src_id
    q_heads = q_batch[-1] if isinstance(q_batch[-1], int) else None
    s_heads = src_batch[-1] if isinstance(src_batch[-1], int) else None
    if not (q_heads and s_heads and q_heads > s_heads and q_heads % s_heads == 0):
        return src_id
    head_axis = len(src_batch) - 1
    target_shape = tuple(src_batch[:head_axis]) + (q_heads,) + tuple(target_last_dims)
    return gqa_broadcast(
        frag,
        src_id,
        target_shape=target_shape,
        head_axis=head_axis,
        group_size=q_heads // s_heads,
        name=name,
        dtype=dtype,
    )


def rewrite(graph: Graph, root: Node) -> Graph | None:
    q_id, k_id, v_id = root.inputs[0], root.inputs[1], root.inputs[2]

    q_shape = graph.nodes[q_id].output.shape
    k_shape = graph.nodes[k_id].output.shape
    v_shape = graph.nodes[v_id].output.shape
    dtype = root.output.dtype
    name = root.output.name

    head_dim = q_shape[-1] if len(q_shape) >= 2 else 64
    seq_len = q_shape[-2] if len(q_shape) >= 3 else q_shape[-1]
    q_batch = q_shape[:-2] if len(q_shape) > 2 else ()
    k_batch = k_shape[:-2] if len(k_shape) > 2 else ()
    v_batch = v_shape[:-2] if len(v_shape) > 2 else ()
    scores_shape = q_batch + (seq_len, seq_len)

    frag = open_fragment(graph, [q_id, k_id, v_id])

    # K^T then GQA broadcast.
    kt_shape = k_batch + (head_dim, seq_len) if isinstance(head_dim, int) else k_shape
    kt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[k_id],
        output=Tensor(f"{name}_kt", kt_shape, dtype),
    )
    kt_id = _maybe_gqa(frag, kt_id, q_batch, k_batch, (head_dim, seq_len), name=f"{name}_kt_gqa", dtype=dtype)

    # QK^T matmul.
    qk_id = matmul_decompose(frag, q_id, kt_id, name=f"{name}_qk", dtype=dtype)

    # Scale by 1/sqrt(head_dim).
    scale_value = 1.0 / math.sqrt(head_dim) if isinstance(head_dim, int) else None
    scale_bc = const_bc(frag, name=f"{name}_scale", value=scale_value, target_shape=scores_shape, dtype=dtype)
    scaled_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[qk_id, scale_bc],
        output=Tensor(f"{name}_scaled", scores_shape, dtype),
    )

    # Causal mask: add -1e9 where key_pos > query_pos.
    if root.op.is_causal:
        ndim_scores = len(scores_shape)
        i_var = placeholder(ndim_scores - 2)
        j_var = placeholder(ndim_scores - 1)
        zero_id = frag.add_node(
            op=ConstantOp(name=f"{name}_mask_zero", value=0.0),
            inputs=[],
            output=Tensor(f"{name}_mask_zero", (1,), dtype),
        )
        mask_fill_id = frag.add_node(
            op=ConstantOp(name=f"{name}_mask_fill", value=-1e9),
            inputs=[],
            output=Tensor(f"{name}_mask_fill", (1,), dtype),
        )
        causal_mask_op = IndexMapOp(
            out_shape=scores_shape,
            sources=(
                IndexSource(input_idx=0, coord_map=(Literal(0, "int"),), select=BinaryExpr("<=", j_var, i_var)),
                IndexSource(input_idx=1, coord_map=(Literal(0, "int"),), select=BinaryExpr(">", j_var, i_var)),
            ),
        )
        mask_id = frag.add_node(
            op=causal_mask_op,
            inputs=[zero_id, mask_fill_id],
            output=Tensor(f"{name}_cmask", scores_shape, dtype),
        )
        scaled_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[scaled_id, mask_id],
            output=Tensor(f"{name}_masked", scores_shape, dtype),
        )

    softmax_id = softmax_decompose(frag, scaled_id, -1, name=f"{name}_softmax", dtype=dtype)

    # Softmax @ V (with GQA on V).
    v_eff_id = _maybe_gqa(frag, v_id, q_batch, v_batch, v_shape[-2:] if len(v_shape) >= 2 else v_shape, name=f"{name}_v_gqa", dtype=dtype)
    sv_id = matmul_decompose(frag, softmax_id, v_eff_id, name=name, dtype=dtype)

    frag.outputs = [sv_id]
    return frag
