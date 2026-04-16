"""Tests for individual decomposition rules (SDPA, pow).

Supplements test_rewriter.py which only tests SiLU decomposition.
Each test builds a minimal graph containing the target op and applies
the specific decomposition rule in isolation.
"""

from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, InputOp, ReduceOp, SdpaOp, TransposeOp
from deplodock.compiler.rewriter import Pass, Rule

RULES_DIR = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "decomposition"


# ---------------------------------------------------------------------------
# Pow decomposition: pow(x) → mul(x, x)
# ---------------------------------------------------------------------------


def _make_pow_graph():
    """Build graph with pow(x, 2) for decomposition testing."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 128)), node_id="x")
    exp = g.add_node(op=ConstantOp(name="exp", value=2.0), inputs=[], output=Tensor("exp", (1,)), node_id="exp")
    g.inputs = [x]
    out = g.add_node(op=ElementwiseOp(fn="pow"), inputs=[x, exp], output=Tensor("out", (4, 128)), node_id="out")
    g.outputs = [out]
    return g


def _load_pow_rule():
    return Rule.from_file(RULES_DIR / "003_decompose_pow.py")


def test_pow_decomposes_to_self_mul():
    """pow(x) → mul(x, x)."""
    g = _make_pow_graph()
    result = Pass(name="decomp", rules=[_load_pow_rule()]).apply(g)

    fns = [n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)]
    assert "pow" not in fns, "pow should be eliminated"
    assert "mul" in fns, "pow should decompose to mul"


def test_pow_output_uses_self_multiply():
    """The decomposed mul node uses the same input for both operands."""
    g = _make_pow_graph()
    result = Pass(name="decomp", rules=[_load_pow_rule()]).apply(g)

    out_node = result.nodes[result.outputs[0]]
    assert isinstance(out_node.op, ElementwiseOp)
    assert out_node.op.fn == "mul"
    assert len(out_node.inputs) == 2
    assert out_node.inputs[0] == out_node.inputs[1], "pow(x) should become mul(x, x)"


def test_pow_preserves_shape_and_dtype():
    """Decomposition preserves the output shape and dtype."""
    g = _make_pow_graph()
    original_shape = g.nodes[g.outputs[0]].output.shape
    original_dtype = g.nodes[g.outputs[0]].output.dtype

    result = Pass(name="decomp", rules=[_load_pow_rule()]).apply(g)

    out_node = result.nodes[result.outputs[0]]
    assert out_node.output.shape == original_shape
    assert out_node.output.dtype == original_dtype


def test_pow_idempotent():
    """Applying pow decomposition twice has no additional effect."""
    g = _make_pow_graph()
    p = Pass(name="decomp", rules=[_load_pow_rule()])
    once = p.apply(g)
    twice = p.apply(once)
    assert len(twice.nodes) == len(once.nodes)


# ---------------------------------------------------------------------------
# SDPA decomposition: sdpa(Q, K, V) → QK^T → scale → softmax → @V
# ---------------------------------------------------------------------------


def _make_sdpa_graph(seq_len=32, head_dim=64, num_heads=1):
    """Build a graph with sdpa(Q, K, V)."""
    g = Graph()
    q_shape = (num_heads, seq_len, head_dim)
    k_shape = (num_heads, seq_len, head_dim)
    v_shape = (num_heads, seq_len, head_dim)
    out_shape = (num_heads, seq_len, head_dim)

    q = g.add_node(op=InputOp(), inputs=[], output=Tensor("Q", q_shape), node_id="Q")
    k = g.add_node(op=InputOp(), inputs=[], output=Tensor("K", k_shape), node_id="K")
    v = g.add_node(op=InputOp(), inputs=[], output=Tensor("V", v_shape), node_id="V")
    g.inputs = [q, k, v]

    out = g.add_node(
        op=SdpaOp(),
        inputs=[q, k, v],
        output=Tensor("out", out_shape),
        node_id="sdpa_out",
    )
    g.outputs = [out]
    return g


def _load_sdpa_rule():
    return Rule.from_file(RULES_DIR / "001_decompose_sdpa.py")


def test_sdpa_decomposes():
    """sdpa is fully replaced by primitive ops."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "sdpa" not in fns, "sdpa should be fully decomposed"


def test_sdpa_produces_transpose():
    """SDPA decomposition produces a K^T transpose."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    transposes = [n for n in result.nodes.values() if isinstance(n.op, TransposeOp)]
    assert len(transposes) >= 1, "Should have K^T transpose"
    # Transpose of last two dims.
    assert transposes[0].op.axes == (-2, -1)


def test_sdpa_produces_two_matmuls():
    """SDPA decomposes into two matmuls: QK^T and softmax@V."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    # Each matmul = mul + sum pair.
    muls = [n for n in result.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.fn == "mul"]
    sums = [n for n in result.nodes.values() if isinstance(n.op, ReduceOp) and n.op.fn == "sum"]
    # scale constant mul also counts as a mul, so >= 3 muls (2 matmul + 1 scale).
    assert len(muls) >= 3, f"Expected >= 3 mul ops (2 matmul + scale), got {len(muls)}"
    assert len(sums) >= 2, f"Expected >= 2 sum ops (2 matmul reductions), got {len(sums)}"


def test_sdpa_produces_softmax_pattern():
    """SDPA decomposition includes max, sub, exp, sum, div (softmax)."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    reduce_fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ReduceOp)}

    assert "sub" in fns, "Softmax needs sub (x - max)"
    assert "exp" in fns, "Softmax needs exp"
    assert "div" in fns, "Softmax needs div (exp / sum)"
    assert "max" in reduce_fns, "Softmax needs max reduction"


def test_sdpa_produces_scale_constant():
    """SDPA decomposition includes a scale constant (1/sqrt(d_k))."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    constants = [n for n in result.nodes.values() if isinstance(n.op, ConstantOp)]
    scale_constants = [c for c in constants if "scale" in c.op.name]
    assert len(scale_constants) >= 1, "Should have a scale constant"


def test_sdpa_preserves_io_count():
    """SDPA decomposition preserves 3 inputs and 1 output."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    assert len(result.inputs) == 3, f"Should have 3 inputs (Q, K, V), got {len(result.inputs)}"
    assert len(result.outputs) == 1, f"Should have 1 output, got {len(result.outputs)}"


def test_sdpa_idempotent():
    """Applying SDPA decomposition twice has no additional effect."""
    g = _make_sdpa_graph()
    p = Pass(name="decomp", rules=[_load_sdpa_rule()])
    once = p.apply(g)
    twice = p.apply(once)
    assert len(twice.nodes) == len(once.nodes)


def test_sdpa_output_is_valid():
    """The decomposed graph's output node is an ElementwiseOp or ReduceOp."""
    g = _make_sdpa_graph()
    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    out_node = result.nodes[result.outputs[0]]
    assert isinstance(out_node.op, (ElementwiseOp, ReduceOp))


def test_sdpa_with_extra_args_not_matched():
    """SDPA with extra args (dropout_p) does NOT match the 3-arg pattern.

    The matcher is strict about arity: Elementwise{sdpa}($Q, $K, $V)
    requires exactly 3 inputs. Nodes with extra args (dropout_p, is_causal)
    must be stripped to 3 inputs by the tracer before decomposition.
    """
    g = Graph()
    q = g.add_node(op=InputOp(), inputs=[], output=Tensor("Q", (1, 32, 64)), node_id="Q")
    k = g.add_node(op=InputOp(), inputs=[], output=Tensor("K", (1, 32, 64)), node_id="K")
    v = g.add_node(op=InputOp(), inputs=[], output=Tensor("V", (1, 32, 64)), node_id="V")
    dp = g.add_node(op=ConstantOp(name="dropout_p"), inputs=[], output=Tensor("dp", (1,)), node_id="dp")
    g.inputs = [q, k, v]

    out = g.add_node(
        op=SdpaOp(),
        inputs=[q, k, v, dp],
        output=Tensor("out", (1, 32, 64)),
        node_id="sdpa_out",
    )
    g.outputs = [out]

    result = Pass(name="decomp", rules=[_load_sdpa_rule()]).apply(g)

    # Pattern doesn't match — sdpa survives (graph unchanged).
    has_sdpa = any(isinstance(n.op, SdpaOp) for n in result.nodes.values())
    assert has_sdpa, "SdpaOp should survive when pattern doesn't match"
    assert len(result.nodes) == len(g.nodes)
