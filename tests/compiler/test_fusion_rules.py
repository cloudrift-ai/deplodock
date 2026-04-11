"""Unit tests for each fusion rule: RMSNorm, softmax, SiLU+mul, matmul."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import (
    ConstantOp,
    ElementwiseOp,
    FusedReduceElementwiseOp,
    FusedRMSNormOp,
    FusedSiLUMulOp,
    FusedSoftmaxOp,
    InputOp,
    MatmulOp,
    ReduceOp,
)
from deplodock.compiler.rewriter import Pass, Rule

# ---- helpers ----


def _load_rule(name: str) -> Rule:
    from pathlib import Path

    rule_path = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion" / name
    return Rule.from_file(rule_path)


def _apply_rule(graph: Graph, rule_name: str) -> Graph:
    rule = _load_rule(rule_name)
    return Pass(name="fusion", rules=[rule]).apply(graph)


# ---- RMSNorm fusion ----


def _make_rmsnorm_graph():
    """Build: (x * rsqrt(sum(x*x) * inv_n + eps)) * weight."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("S", "D")), node_id="x")
    w = g.add_node(op=InputOp(), inputs=[], output=Tensor("w", ("D",)), node_id="w")
    eps = g.add_node(op=ConstantOp(name="eps"), inputs=[], output=Tensor("eps", (1,)), node_id="eps")
    g.inputs = [x]

    # Squared norm: FusedReduceElementwiseOp(sum, mul) — produced by decomposition pass
    sq_norm = g.add_node(
        op=FusedReduceElementwiseOp(reduce_fn="sum", elementwise_fn="mul", axis=1),
        inputs=[x, x],
        output=Tensor("sq_norm", ("S",)),
        node_id="sq_norm",
    )
    # sq_norm + eps
    add_eps = g.add_node(op=ElementwiseOp(fn="add"), inputs=[sq_norm, eps], output=Tensor("var", ("S",)), node_id="add_eps")
    # rsqrt(var)
    rsqrt = g.add_node(op=ElementwiseOp(fn="rsqrt"), inputs=[add_eps], output=Tensor("rsqrt", ("S",)), node_id="rsqrt")
    # x * rsqrt (fan-out: same x node)
    norm = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[x, rsqrt], output=Tensor("norm", ("S", "D")), node_id="norm")
    # norm * weight
    out = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[norm, w], output=Tensor("out", ("S", "D")), node_id="out")
    g.outputs = [out]
    return g


def test_fuse_rmsnorm():
    """RMSNorm chain → FusedRMSNormOp."""
    g = _make_rmsnorm_graph()
    result = _apply_rule(g, "000_fuse_rmsnorm.py")

    op_types = {type(n.op).__name__ for n in result.nodes.values()}
    assert "FusedRMSNormOp" in op_types
    assert "ReduceOp" not in op_types

    fused = [n for n in result.nodes.values() if isinstance(n.op, FusedRMSNormOp)]
    assert len(fused) == 1
    assert set(fused[0].inputs) == {"x", "w"}


def test_rmsnorm_does_not_match_plain_matmul():
    """A regular matmul (A*B, not x*x) should not match RMSNorm pattern."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("K", "N")), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", ("M", "K", "N")), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", ("M", "N")), node_id="red")
    g.outputs = [red]

    result = _apply_rule(g, "000_fuse_rmsnorm.py")
    # Should be unchanged — no RMSNorm pattern.
    assert len(result.nodes) == len(g.nodes)


# ---- Softmax fusion ----


def _make_softmax_graph():
    """Build: exp(x - max(x)) / sum(exp(x - max(x))) with shared exp node."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", ("S", "D")), node_id="x")
    g.inputs = [x]

    # max(x)
    mx = g.add_node(op=ReduceOp(fn="max", axis=1), inputs=[x], output=Tensor("mx", ("S",)), node_id="mx")
    # x - max(x)
    sub = g.add_node(op=ElementwiseOp(fn="sub"), inputs=[x, mx], output=Tensor("shifted", ("S", "D")), node_id="sub")
    # exp(shifted)
    exp = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[sub], output=Tensor("exp", ("S", "D")), node_id="exp")
    # sum(exp) — denominator
    sum_exp = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[exp], output=Tensor("sum_exp", ("S",)), node_id="sum_exp")
    # exp / sum(exp) — fan-out on exp node
    div = g.add_node(op=ElementwiseOp(fn="div"), inputs=[exp, sum_exp], output=Tensor("softmax", ("S", "D")), node_id="div")
    g.outputs = [div]
    return g


def test_fuse_softmax():
    """Softmax chain → FusedSoftmaxOp."""
    g = _make_softmax_graph()
    result = _apply_rule(g, "002_fuse_softmax.py")

    op_types = {type(n.op).__name__ for n in result.nodes.values()}
    assert "FusedSoftmaxOp" in op_types
    assert "ReduceOp" not in op_types

    fused = [n for n in result.nodes.values() if isinstance(n.op, FusedSoftmaxOp)]
    assert len(fused) == 1
    assert fused[0].inputs == ["x"]


def test_softmax_does_not_match_plain_div():
    """A plain div(a, reduce_sum(b)) where a != b should not match."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("a", ("S",)), node_id="a")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("b", ("S",)), node_id="b")
    g.inputs = [a, b]
    red = g.add_node(op=ReduceOp(fn="sum", axis=0), inputs=[b], output=Tensor("s", (1,)), node_id="red")
    div = g.add_node(op=ElementwiseOp(fn="div"), inputs=[a, red], output=Tensor("out", ("S",)), node_id="div")
    g.outputs = [div]

    result = _apply_rule(g, "002_fuse_softmax.py")
    # Pattern requires fan-out (same node for numerator and sum input), so no match.
    assert len(result.nodes) == len(g.nodes)


# ---- SiLU+mul fusion ----


def _make_silu_mul_graph():
    """Build: gate * recip(1 + exp(-gate)) * up."""
    g = Graph()
    gate = g.add_node(op=InputOp(), inputs=[], output=Tensor("gate", ("S", "D")), node_id="gate")
    up = g.add_node(op=InputOp(), inputs=[], output=Tensor("up", ("S", "D")), node_id="up")
    one = g.add_node(op=ConstantOp(name="one"), inputs=[], output=Tensor("one", (1,)), node_id="one")
    g.inputs = [gate, up]

    # -gate
    neg = g.add_node(op=ElementwiseOp(fn="neg"), inputs=[gate], output=Tensor("neg", ("S", "D")), node_id="neg")
    # exp(-gate)
    exp = g.add_node(op=ElementwiseOp(fn="exp"), inputs=[neg], output=Tensor("exp", ("S", "D")), node_id="exp")
    # 1 + exp(-gate)
    add = g.add_node(op=ElementwiseOp(fn="add"), inputs=[one, exp], output=Tensor("denom", ("S", "D")), node_id="add")
    # 1 / (1 + exp(-gate))
    recip = g.add_node(op=ElementwiseOp(fn="recip"), inputs=[add], output=Tensor("sigmoid", ("S", "D")), node_id="recip")
    # gate * sigmoid (fan-out on gate)
    silu = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[gate, recip], output=Tensor("silu", ("S", "D")), node_id="silu")
    # silu * up
    out = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[silu, up], output=Tensor("out", ("S", "D")), node_id="out")
    g.outputs = [out]
    return g


def test_fuse_silu_mul():
    """SiLU+mul chain → FusedSiLUMulOp."""
    g = _make_silu_mul_graph()
    result = _apply_rule(g, "003_fuse_silu_mul.py")

    op_types = {type(n.op).__name__ for n in result.nodes.values()}
    assert "FusedSiLUMulOp" in op_types

    fused = [n for n in result.nodes.values() if isinstance(n.op, FusedSiLUMulOp)]
    assert len(fused) == 1
    assert set(fused[0].inputs) == {"gate", "up"}


# ---- Matmul fusion (updated to produce MatmulOp) ----


def test_matmul_produces_matmul_op():
    """Reduce{sum}(Elementwise{mul}(A, B)) → MatmulOp."""
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", ("M", "K")), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", ("K", "N")), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", ("M", "K", "N")), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", ("M", "N")), node_id="red")
    g.outputs = [red]

    result = _apply_rule(g, "001_fuse_reduce_elementwise.py")

    fused = [n for n in result.nodes.values() if isinstance(n.op, MatmulOp)]
    assert len(fused) == 1
    assert set(fused[0].inputs) == {"A", "B"}
