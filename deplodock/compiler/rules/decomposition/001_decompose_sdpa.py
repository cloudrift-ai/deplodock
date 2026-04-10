"""Decompose scaled_dot_product_attention into QK^T → scale → softmax → @V."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, ReduceOp, TransposeOp

PATTERN = "Elementwise{sdpa}($Q, $K, $V)"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace sdpa(Q, K, V) with QK^T → scale → softmax → @V."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    q_id = match.bindings["Q"]
    k_id = match.bindings["K"]
    v_id = match.bindings["V"]

    q_shape = g.nodes[q_id].output.shape
    out_shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # Infer head_dim from Q shape: Q is [..., seq_len, head_dim]
    head_dim = q_shape[-1] if len(q_shape) >= 2 else 64
    seq_len = q_shape[-2] if len(q_shape) >= 3 else q_shape[-1]
    batch_heads_shape = q_shape[:-2] if len(q_shape) > 2 else ()
    scores_shape = batch_heads_shape + (seq_len, seq_len)

    # K^T: transpose last two dims
    kt_id = g.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[k_id],
        output=Tensor(f"{name}_kt", batch_heads_shape + (head_dim, seq_len) if isinstance(head_dim, int) else q_shape, dtype),
    )

    # QK^T: matmul(Q, K^T) — decomposed as elementwise mul + reduce sum
    qk_ew_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[q_id, kt_id],
        output=Tensor(f"{name}_qk_ew", scores_shape + (head_dim,) if isinstance(head_dim, int) else scores_shape, dtype),
    )
    qk_id = g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=[qk_ew_id],
        output=Tensor(f"{name}_qk", scores_shape, dtype),
    )

    # Scale
    scale_const_id = g.add_node(
        op=ConstantOp(name=f"{name}_scale"),
        inputs=[],
        output=Tensor(f"{name}_scale", (1,), dtype),
    )
    scaled_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[qk_id, scale_const_id],
        output=Tensor(f"{name}_scaled", scores_shape, dtype),
    )

    # Softmax: max → sub → exp → sum → div
    max_id = g.add_node(
        op=ReduceOp(fn="max", axis=-1),
        inputs=[scaled_id],
        output=Tensor(f"{name}_max", scores_shape[:-1] + (1,) if scores_shape else (1,), dtype),
    )
    sub_id = g.add_node(
        op=ElementwiseOp(fn="sub"),
        inputs=[scaled_id, max_id],
        output=Tensor(f"{name}_shifted", scores_shape, dtype),
    )
    exp_id = g.add_node(
        op=ElementwiseOp(fn="exp"),
        inputs=[sub_id],
        output=Tensor(f"{name}_exp", scores_shape, dtype),
    )
    sum_id = g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=[exp_id],
        output=Tensor(f"{name}_sum", scores_shape[:-1] + (1,) if scores_shape else (1,), dtype),
    )
    softmax_id = g.add_node(
        op=ElementwiseOp(fn="div"),
        inputs=[exp_id, sum_id],
        output=Tensor(f"{name}_softmax", scores_shape, dtype),
    )

    # Softmax @ V: matmul(softmax, V) — decomposed as mul + reduce sum
    sv_ew_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[softmax_id, v_id],
        output=Tensor(f"{name}_sv_ew", out_shape + (seq_len,) if isinstance(seq_len, int) else out_shape, dtype),
    )
    sv_id = g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=[sv_ew_id],
        output=Tensor(name, out_shape, dtype),
    )

    g.replace_node(match.root_node_id, sv_id)
    g.remove_node(match.root_node_id)
    return g
