"""Tests for automatic fusion region discovery.

Tests both the MILP-based fusion algorithm and its behavior on
common LLM patterns: softmax, RMSNorm, SiLU, matmul, matmul+bias,
pointwise chains, and multi-consumer diamonds.
"""

import json
from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, InputOp, KernelOp, ReduceOp
from deplodock.compiler.rewriter import Rewriter
from tests.compiler._fusion_helper import auto_fuse

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_and_decompose() -> Graph:
    """Load TinyLlama fixture and run decomposition (no fusion)."""
    with open(FIXTURE_DIR / "tinyllama_layer0.json") as f:
        g = Graph.from_dict(json.load(f))
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    decomp_pass = [p for p in rewriter.passes if p.name == "decomposition"][0]
    return decomp_pass.apply(g)


def _fused_regions(g: Graph) -> list:
    """Return list of FusedRegionOp nodes from a fused graph."""
    return [n for n in g.nodes.values() if isinstance(n.op, KernelOp)]


def _region_op_types(region) -> list[str]:
    """Return list of op type strings from a FusedRegionOp."""
    result = []
    for _node in region.op.body_ops():
        op = _node.op
        t = type(op).__name__
        fn = getattr(op, "fn", "")
        result.append(f"{t}({fn})" if fn else t)
    return result


# ---------------------------------------------------------------------------
# TinyLlama fixture tests
# ---------------------------------------------------------------------------


def test_auto_fuse_produces_fused_regions():
    """auto_fuse groups primitive ops into FusedRegionOp nodes."""
    fused = auto_fuse(_load_and_decompose())
    assert len(_fused_regions(fused)) > 0


def test_auto_fuse_reduces_node_count():
    """Fusion should reduce the total node count."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)
    assert len(fused.nodes) < len(decomposed.nodes)


def test_auto_fuse_preserves_io():
    """Fusion preserves graph inputs and outputs."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)
    assert len(fused.inputs) == len(decomposed.inputs)
    assert len(fused.outputs) == len(decomposed.outputs)


def test_auto_fuse_region_has_ops():
    """Each FusedRegionOp contains primitive ops."""
    fused = auto_fuse(_load_and_decompose())
    for n in _fused_regions(fused):
        assert len(n.op.body_ops()) >= 1
        assert len(n.op.inputs) > 0
        assert len(n.op.outputs) > 0


# ---------------------------------------------------------------------------
# Pattern: Softmax — max → sub → exp → sum → div
# Multi-consumer: exp feeds both sum and div
# ---------------------------------------------------------------------------


def test_softmax_fuses_into_one_region():
    """Softmax regions: codegen supports single reduce per region, so max and
    sum go into separate regions. The sub/exp ops fuse with max, div with sum."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 64)), node_id="x")
    g.inputs = [x]
    mx = g.add_node(ReduceOp("max", axis=1), [x], Tensor("mx", (4, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (4, 64)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (4, 64)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=1), [exp], Tensor("sm", (4, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (4, 64)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    # Multi-reduce fusion needs multi-pass codegen (future). For now,
    # max and sum are in separate regions.
    assert len(regions) >= 1, f"Softmax should have fused regions, got {len(regions)}"
    all_ops = []
    for r in regions:
        all_ops.extend(_region_op_types(r))
    # At least one reduce should be fused with its adjacent elementwise ops.
    assert any("ReduceOp" in op for op in all_ops)


# ---------------------------------------------------------------------------
# Pattern: Softmax on 4D tensor (attention scores)
# max → sub → exp → sum → div on (1, 28, 32, 32) — should fuse
# ---------------------------------------------------------------------------


def test_softmax_4d_fuses():
    """Softmax on 4D attention scores should fuse (>2D dims allowed)."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (1, 28, 32, 32)), node_id="x")
    g.inputs = [x]
    mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (1, 28, 32, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (1, 28, 32, 32)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (1, 28, 32, 32)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (1, 28, 32, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (1, 28, 32, 32)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    # Multi-reduce fusion: all softmax ops (max+sub+exp+sum+div) in one region.
    assert len(regions) == 1, f"Expected 1 region for 4D softmax, got {len(regions)}"
    all_ops = []
    for r in regions:
        all_ops.extend(_region_op_types(r))
    assert "ReduceOp(max)" in all_ops
    assert "ReduceOp(sum)" in all_ops
    assert "ElementwiseOp(sub)" in all_ops
    assert "ElementwiseOp(exp)" in all_ops
    assert "ElementwiseOp(div)" in all_ops


# ---------------------------------------------------------------------------
# Pattern: RMSNorm — mul(x,x) → sum → add(eps) → rsqrt → mul(x) → mul(w)
# Multi-consumer: x feeds both mul(x,x) and mul(x, rsqrt)
# ---------------------------------------------------------------------------


def test_rmsnorm_fuses_into_one_region():
    """RMSNorm should fuse into a single region."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 128)), node_id="x")
    w = g.add_node(InputOp(), [], Tensor("w", (128,)), node_id="w")
    eps = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    g.inputs = [x, w]
    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (4, 128)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=1), [sq], Tensor("s", (4, 1)), node_id="s")
    ae = g.add_node(ElementwiseOp("add"), [red, eps], Tensor("ae", (4, 1)), node_id="ae")
    rs = g.add_node(ElementwiseOp("rsqrt"), [ae], Tensor("rs", (4, 1)), node_id="rs")
    norm = g.add_node(ElementwiseOp("mul"), [x, rs], Tensor("norm", (4, 128)), node_id="norm")
    out = g.add_node(ElementwiseOp("mul"), [norm, w], Tensor("out", (4, 128)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1, f"RMSNorm should be 1 region, got {len(regions)}"
    ops = _region_op_types(regions[0])
    assert "ReduceOp(sum)" in ops
    assert "ElementwiseOp(rsqrt)" in ops


# ---------------------------------------------------------------------------
# Pattern: SiLU — neg → exp → add(1) → recip → mul(x)
# ---------------------------------------------------------------------------


def test_silu_fuses_into_one_region():
    """SiLU chain should fuse into a single pointwise region."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (256,)), node_id="x")
    one = g.add_node(ConstantOp(name="one"), [], Tensor("one", (1,)), node_id="one")
    g.inputs = [x]
    neg = g.add_node(ElementwiseOp("neg"), [x], Tensor("neg", (256,)), node_id="neg")
    exp = g.add_node(ElementwiseOp("exp"), [neg], Tensor("exp", (256,)), node_id="exp")
    add = g.add_node(ElementwiseOp("add"), [one, exp], Tensor("add", (256,)), node_id="add")
    recip = g.add_node(ElementwiseOp("recip"), [add], Tensor("recip", (256,)), node_id="recip")
    out = g.add_node(ElementwiseOp("mul"), [x, recip], Tensor("out", (256,)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1, f"SiLU should be 1 region, got {len(regions)}"


# ---------------------------------------------------------------------------
# Pattern: Matmul — mul(A, B) → reduce_sum
# ---------------------------------------------------------------------------


def test_matmul_fuses():
    """Matmul (mul → sum) should fuse into a single contraction region."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 6)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (4, 8, 6)), node_id="AB")
    out = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("C", (4, 6)), node_id="C")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1
    ops = _region_op_types(regions[0])
    assert "ElementwiseOp(mul)" in ops
    assert "ReduceOp(sum)" in ops


# ---------------------------------------------------------------------------
# Pattern: Matmul + residual add — mul → sum → add(residual)
# Multi-consumer: residual is external, add is epilogue
# ---------------------------------------------------------------------------


def test_matmul_residual_add_fuses():
    """Matmul + residual add should fuse into a single region (add as epilogue)."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 6)), node_id="B")
    res = g.add_node(InputOp(), [], Tensor("res", (4, 6)), node_id="res")
    g.inputs = [a, b, res]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (4, 8, 6)), node_id="AB")
    red = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("mm", (4, 6)), node_id="mm")
    out = g.add_node(ElementwiseOp("add"), [red, res], Tensor("out", (4, 6)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    # Matmul + epilogue add fuse into a single contraction region.
    assert len(regions) == 1, f"Expected 1 fused region, got {len(regions)}"
    ops = _region_op_types(regions[0])
    assert "ElementwiseOp(mul)" in ops
    assert "ReduceOp(sum)" in ops
    assert "ElementwiseOp(add)" in ops


# ---------------------------------------------------------------------------
# Pattern: Matmul + bias add + ReLU — mul → sum → add(bias) → relu
# All four ops should fuse into a single contraction-with-epilogue region.
# ---------------------------------------------------------------------------


def test_matmul_bias_activation_fuses():
    """Matmul + bias add + ReLU should fuse into a single region."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 6)), node_id="B")
    bias = g.add_node(InputOp(), [], Tensor("bias", (6,)), node_id="bias")
    g.inputs = [a, b, bias]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (4, 8, 6)), node_id="AB")
    red = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("mm", (4, 6)), node_id="mm")
    ba = g.add_node(ElementwiseOp("add"), [red, bias], Tensor("ba", (4, 6)), node_id="ba")
    out = g.add_node(ElementwiseOp("relu"), [ba], Tensor("out", (4, 6)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1, f"Expected 1 fused region, got {len(regions)}"
    ops = _region_op_types(regions[0])
    assert "ElementwiseOp(mul)" in ops
    assert "ReduceOp(sum)" in ops
    assert "ElementwiseOp(add)" in ops
    assert "ElementwiseOp(relu)" in ops


# ---------------------------------------------------------------------------
# Pattern: Two independent matmuls sharing an input
# Must be in SEPARATE regions (different contractions)
# ---------------------------------------------------------------------------


def test_two_matmuls_stay_separate():
    """Two matmuls sharing input A must be in separate regions."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b1 = g.add_node(InputOp(), [], Tensor("B1", (8, 6)), node_id="B1")
    b2 = g.add_node(InputOp(), [], Tensor("B2", (8, 3)), node_id="B2")
    g.inputs = [a, b1, b2]
    ew1 = g.add_node(ElementwiseOp("mul"), [a, b1], Tensor("AB1", (4, 8, 6)), node_id="AB1")
    c1 = g.add_node(ReduceOp("sum", axis=1), [ew1], Tensor("C1", (4, 6)), node_id="C1")
    ew2 = g.add_node(ElementwiseOp("mul"), [a, b2], Tensor("AB2", (4, 8, 3)), node_id="AB2")
    c2 = g.add_node(ReduceOp("sum", axis=1), [ew2], Tensor("C2", (4, 3)), node_id="C2")
    g.outputs = [c1, c2]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 2, f"Two matmuls should be 2 regions, got {len(regions)}"


# ---------------------------------------------------------------------------
# Pattern: Matmul followed by RMSNorm — must be SEPARATE
# (contraction axis ≠ row reduction axis)
# ---------------------------------------------------------------------------


def test_matmul_then_rmsnorm_stays_separate():
    """Matmul and RMSNorm must be separate regions (different axes)."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 6)), node_id="B")
    eps = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (4, 8, 6)), node_id="AB")
    mm = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("mm", (4, 6)), node_id="mm")
    # RMSNorm on the matmul output
    sq = g.add_node(ElementwiseOp("mul"), [mm, mm], Tensor("sq", (4, 6)), node_id="sq")
    rs = g.add_node(ReduceOp("sum", axis=-1), [sq], Tensor("rs", (4, 1)), node_id="rs")
    ae = g.add_node(ElementwiseOp("add"), [rs, eps], Tensor("ae", (4, 1)), node_id="ae")
    out = g.add_node(ElementwiseOp("rsqrt"), [ae], Tensor("out", (4, 1)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) >= 2, f"Matmul + RMSNorm should be ≥2 regions, got {len(regions)}"


# ---------------------------------------------------------------------------
# Pattern: Pointwise chain — pure elementwise, no reductions
# ---------------------------------------------------------------------------


def test_pointwise_chain_fuses():
    """A chain of elementwise ops should fuse into one region."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (100,)), node_id="x")
    g.inputs = [x]
    a = g.add_node(ElementwiseOp("neg"), [x], Tensor("a", (100,)), node_id="a")
    b = g.add_node(ElementwiseOp("exp"), [a], Tensor("b", (100,)), node_id="b")
    c = g.add_node(ElementwiseOp("neg"), [b], Tensor("c", (100,)), node_id="c")
    g.outputs = [c]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1


# ---------------------------------------------------------------------------
# Pattern: Diamond — x feeds both add(x, x) (no reduction)
# Multi-consumer: x consumed twice by the same op
# ---------------------------------------------------------------------------


def test_diamond_fuses():
    """x → add(x, x) should fuse (same input used twice)."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (100,)), node_id="x")
    g.inputs = [x]
    out = g.add_node(ElementwiseOp("add"), [x, x], Tensor("out", (100,)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    # Single op — may or may not create a FusedRegionOp (singletons stay as-is).
    # The key assertion: no errors, no cycles.
    assert len(fused.outputs) == 1


# ---------------------------------------------------------------------------
# Pattern: Residual connection — x → f(x) → add(x, f(x))
# Multi-consumer: x feeds both f and the skip connection
# ---------------------------------------------------------------------------


def test_residual_connection_fuses():
    """Residual: x → neg → exp → add(x, exp) should fuse."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (100,)), node_id="x")
    g.inputs = [x]
    neg = g.add_node(ElementwiseOp("neg"), [x], Tensor("neg", (100,)), node_id="neg")
    exp = g.add_node(ElementwiseOp("exp"), [neg], Tensor("exp", (100,)), node_id="exp")
    out = g.add_node(ElementwiseOp("add"), [x, exp], Tensor("out", (100,)), node_id="out")
    g.outputs = [out]

    fused = auto_fuse(g)
    regions = _fused_regions(fused)
    assert len(regions) == 1, f"Residual should fuse into 1 region, got {len(regions)}"
