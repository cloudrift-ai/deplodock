"""Integration test: full Llama-style transformer block through the rewriter."""

from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    FusedReduceElementwiseOp,
    InputOp,
    ReduceOp,
    TransposeOp,
)
from deplodock.compiler.rewriter import Rewriter


def _build_llama_block():
    """Build a simplified Llama 3.1 transformer block in our IR.

    Structure:
        RMSNorm → Q/K/V projections → K^T → QK matmul → scale → softmax →
        attn@V → output projection → residual add →
        RMSNorm → gate/up projections → SiLU+mul → down projection → residual add
    """
    g = Graph()

    # --- Inputs ---
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("S", "D")), node_id="x")
    g.inputs = [x]

    # --- Constants (weights) ---
    def _const(name, shape):
        return g.add_node(op=ConstantOp(name=name), inputs=[], output=Tensor(name, shape), node_id=name)

    rms1_w = _const("rms1_w", ("D",))
    wq = _const("Wq", ("D", "D"))
    wk = _const("Wk", ("D", "D"))
    wv = _const("Wv", ("D", "D"))
    wo = _const("Wo", ("D", "D"))
    rms2_w = _const("rms2_w", ("D",))
    wg = _const("Wg", ("D", "I"))
    wu = _const("Wu", ("D", "I"))
    wd = _const("Wd", ("I", "D"))
    eps = _const("eps", (1,))
    scale = _const("scale", (1,))
    one = _const("one", (1,))

    # --- RMSNorm 1 ---
    def _rmsnorm(prefix, inp_id, w_id):
        sq_norm = g.add_node(
            op=FusedReduceElementwiseOp(reduce_fn="sum", elementwise_fn="mul", axis=1),
            inputs=[inp_id, inp_id],
            output=Tensor(f"{prefix}_sq_norm", ("S",)),
            node_id=f"{prefix}_sq_norm",
        )
        add_eps = g.add_node(
            op=ElementwiseOp(fn="add"), inputs=[sq_norm, eps], output=Tensor(f"{prefix}_var", ("S",)), node_id=f"{prefix}_var"
        )
        rsqrt = g.add_node(
            op=ElementwiseOp(fn="rsqrt"), inputs=[add_eps], output=Tensor(f"{prefix}_rsqrt", ("S",)), node_id=f"{prefix}_rsqrt"
        )
        norm = g.add_node(
            op=ElementwiseOp(fn="mul"), inputs=[inp_id, rsqrt], output=Tensor(f"{prefix}_norm", ("S", "D")), node_id=f"{prefix}_norm"
        )
        out = g.add_node(
            op=ElementwiseOp(fn="mul"), inputs=[norm, w_id], output=Tensor(f"{prefix}_out", ("S", "D")), node_id=f"{prefix}_out"
        )
        return out

    rms1_out = _rmsnorm("rms1", x, rms1_w)

    # --- Linear projections (matmul = reduce_sum(elementwise_mul)) ---
    def _matmul(prefix, inp_id, w_id, out_shape):
        ew = g.add_node(
            op=ElementwiseOp(fn="mul"), inputs=[inp_id, w_id], output=Tensor(f"{prefix}_ew", ("S", "D", "D")), node_id=f"{prefix}_ew"
        )
        red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor(f"{prefix}", out_shape), node_id=prefix)
        return red

    q = _matmul("Q", rms1_out, wq, ("S", "D"))
    k = _matmul("K", rms1_out, wk, ("S", "D"))
    v = _matmul("V", rms1_out, wv, ("S", "D"))

    # --- K^T ---
    kt = g.add_node(op=TransposeOp(axes=(1, 0)), inputs=[k], output=Tensor("Kt", ("D", "S")), node_id="Kt")

    # --- QK matmul → scale ---
    qk_ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[q, kt], output=Tensor("qk_ew", ("S", "D", "S")), node_id="qk_ew")
    qk = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[qk_ew], output=Tensor("qk", ("S", "S")), node_id="qk")
    scaled = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[qk, scale], output=Tensor("scaled", ("S", "S")), node_id="scaled")

    # --- Softmax ---
    mx = g.add_node(op=ReduceOp(fn="max", axis=1), inputs=[scaled], output=Tensor("mx", ("S",)), node_id="mx")
    sub = g.add_node(op=ElementwiseOp(fn="sub"), inputs=[scaled, mx], output=Tensor("shifted", ("S", "S")), node_id="sub")
    exp = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[sub], output=Tensor("exp", ("S", "S")), node_id="exp")
    sum_exp = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[exp], output=Tensor("sum_exp", ("S",)), node_id="sum_exp")
    attn_weights = g.add_node(op=ElementwiseOp(fn="div"), inputs=[exp, sum_exp], output=Tensor("attn_w", ("S", "S")), node_id="attn_w")

    # --- Attention @ V ---
    av_ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[attn_weights, v], output=Tensor("av_ew", ("S", "S", "D")), node_id="av_ew")
    attn_out = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[av_ew], output=Tensor("attn_out", ("S", "D")), node_id="attn_out")

    # --- Output projection ---
    o = _matmul("O", attn_out, wo, ("S", "D"))

    # --- Residual add ---
    res1 = g.add_node(op=ElementwiseOp(fn="add"), inputs=[x, o], output=Tensor("res1", ("S", "D")), node_id="res1")

    # --- RMSNorm 2 ---
    rms2_out = _rmsnorm("rms2", res1, rms2_w)

    # --- FFN: gate + up projections ---
    gate = _matmul("gate", rms2_out, wg, ("S", "I"))
    up = _matmul("up", rms2_out, wu, ("S", "I"))

    # --- SiLU(gate) * up ---
    neg = g.add_node(op=ElementwiseOp(fn="neg"), inputs=[gate], output=Tensor("neg_gate", ("S", "I")), node_id="neg_gate")
    exp_neg = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[neg], output=Tensor("exp_neg", ("S", "I")), node_id="exp_neg")
    add_one = g.add_node(op=ElementwiseOp(fn="add"), inputs=[one, exp_neg], output=Tensor("denom", ("S", "I")), node_id="denom")
    recip = g.add_node(op=ElementwiseOp(fn="recip"), inputs=[add_one], output=Tensor("sigmoid", ("S", "I")), node_id="recip")
    silu = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[gate, recip], output=Tensor("silu", ("S", "I")), node_id="silu")
    silu_mul = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[silu, up], output=Tensor("silu_mul", ("S", "I")), node_id="silu_mul")

    # --- Down projection ---
    down = _matmul("down", silu_mul, wd, ("S", "D"))

    # --- Final residual ---
    res2 = g.add_node(op=ElementwiseOp(fn="add"), inputs=[res1, down], output=Tensor("res2", ("S", "D")), node_id="res2")
    g.outputs = [res2]
    return g


def test_llama_block_full_fusion():
    """Full Llama block → all expected fused ops after rewriting."""
    g = _build_llama_block()
    initial_count = len(g.nodes)

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    result = rewriter.apply(g)

    # Count fused ops by type.
    ops_by_type = {}
    for n in result.nodes.values():
        name = type(n.op).__name__
        ops_by_type[name] = ops_by_type.get(name, 0) + 1

    # Verify expected fused ops.
    assert ops_by_type.get("FusedRMSNormOp", 0) == 2, f"Expected 2 FusedRMSNormOp, got {ops_by_type}"
    # 7 matmuls: Q, K, V, O projections (4) + gate, up, down (3). QK + attn@V consumed by attention.
    assert ops_by_type.get("MatmulOp", 0) == 7, f"Expected 7 MatmulOp, got {ops_by_type}"
    assert ops_by_type.get("FusedAttentionOp", 0) == 1, f"Expected 1 FusedAttentionOp, got {ops_by_type}"
    assert ops_by_type.get("FusedSiLUMulOp", 0) == 1, f"Expected 1 FusedSiLUMulOp, got {ops_by_type}"

    # No leftover ReduceOp.
    assert ops_by_type.get("ReduceOp", 0) == 0, f"Unexpected ReduceOp remaining: {ops_by_type}"

    # Node count should be significantly reduced.
    assert len(result.nodes) < initial_count
