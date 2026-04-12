"""Tests for TileAnalysis pattern classification.

Verifies that analyze() correctly classifies FusedRegionOp patterns
as pointwise, row_reduce, reduce_broadcast, or contraction.
"""

from deplodock.compiler.backend.cuda.generators.analysis import analyze
from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, FusedRegionOp, InputOp, ReduceOp


def _fuse_and_analyze(g: Graph):
    """Auto-fuse a graph and return TileAnalysis for the first fused region."""
    fused = auto_fuse(g)
    for nd in fused.nodes.values():
        if isinstance(nd.op, FusedRegionOp):
            shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
            for nid in fused.nodes:
                shapes[nid] = fused.nodes[nid].output.shape
            return analyze(nd.op, shapes)
    raise AssertionError("No FusedRegionOp found")


def _analyze_manual(region: FusedRegionOp, shapes: dict):
    """Analyze a manually-constructed FusedRegionOp."""
    return analyze(region, shapes)


def test_pointwise_silu():
    """SiLU chain: neg → exp → add → recip → mul is pointwise."""
    g = Graph()
    gate = g.add_node(InputOp(), [], Tensor("gate", (256,)), node_id="gate")
    one = g.add_node(ConstantOp(name="one"), [], Tensor("one", (1,)), node_id="one")
    g.inputs = [gate]
    neg = g.add_node(ElementwiseOp("neg"), [gate], Tensor("neg", (256,)), node_id="neg")
    exp = g.add_node(ElementwiseOp("exp"), [neg], Tensor("exp", (256,)), node_id="exp")
    add = g.add_node(ElementwiseOp("add"), [one, exp], Tensor("add", (256,)), node_id="add")
    recip = g.add_node(ElementwiseOp("recip"), [add], Tensor("recip", (256,)), node_id="recip")
    out = g.add_node(ElementwiseOp("mul"), [gate, recip], Tensor("out", (256,)), node_id="out")
    g.outputs = [out]

    analysis = _fuse_and_analyze(g)
    assert analysis.pattern == "pointwise"
    assert len(analysis.op_phases.reduces) == 0
    assert len(analysis.op_phases.prologue) > 0


def test_row_reduce_sum():
    """Simple row sum is row_reduce."""
    region = FusedRegionOp(
        region_ops=[("out", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["out"],
    )
    shapes = {"x": (8, 64), "out": (8,)}

    analysis = _analyze_manual(region, shapes)
    assert analysis.pattern == "row_reduce"
    assert analysis.rows == 8
    assert analysis.cols == 64
    assert analysis.reduce_fns == ["sum"]
    assert not analysis.epilogue_needs_per_element


def test_reduce_broadcast_rmsnorm():
    """RMSNorm: mul(x,x) → sum → add(eps) → rsqrt → mul(x) → mul(w) is reduce_broadcast."""
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

    analysis = _fuse_and_analyze(g)
    assert analysis.pattern == "reduce_broadcast"
    assert analysis.rows == 4
    assert analysis.cols == 128
    assert analysis.epilogue_needs_per_element


def test_reduce_broadcast_softmax():
    """Softmax region with max reduce + epilogue is reduce_broadcast."""
    # Softmax may split into multiple fused regions. Test the max-reduce region.
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
    # Collect all fused regions and their analyses.
    analyses = []
    for nd in fused.nodes.values():
        if isinstance(nd.op, FusedRegionOp):
            shapes = {nid: g.nodes[nid].output.shape for nid in g.nodes}
            for nid in fused.nodes:
                shapes[nid] = fused.nodes[nid].output.shape
            analyses.append(analyze(nd.op, shapes))

    # At least one region should be reduce_broadcast with epilogue.
    patterns = [a.pattern for a in analyses]
    assert "reduce_broadcast" in patterns
    reduce_analyses = [a for a in analyses if a.pattern in ("reduce_broadcast", "row_reduce")]
    assert len(reduce_analyses) >= 1
    assert any("max" in a.reduce_fns or "sum" in a.reduce_fns for a in reduce_analyses)


def test_contraction_matmul():
    """Pure matmul: mul(A, B) → sum is contraction."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 6)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("AB", (4, 8, 6)), node_id="AB")
    out = g.add_node(ReduceOp("sum", axis=1), [ew], Tensor("C", (4, 6)), node_id="C")
    g.outputs = [out]

    analysis = _fuse_and_analyze(g)
    assert analysis.pattern == "contraction"
    assert analysis.rows == 4
    assert analysis.cols == 6
    assert analysis.k_dim == 8
    assert analysis.contraction_a == "A"
    assert analysis.contraction_b == "B"


def test_input_access_patterns():
    """Verify AccessPattern classification for different input shapes."""
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

    analysis = _fuse_and_analyze(g)
    assert analysis.input_access["x"].is_2d
    assert analysis.input_access["w"].is_row_vector
    assert analysis.input_access["eps"].is_scalar
