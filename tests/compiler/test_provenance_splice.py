"""Provenance propagation through real passes (decompose / lift / fuse).

Exercises the ``Graph.splice`` hook end to end: decomposition mints fresh
pieces under one origin, fusion aggregates pieces across origins, and the
piece count survives recursive decomposition.
"""

from deplodock.compiler import provenance as prov
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def _rms_graph() -> Graph:
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)))
    w = g.add_node(InputOp(), [], Tensor("w", (8,)))
    g.inputs = [x, w]
    rms = g.add_node(RmsNormOp(), [x, w], Tensor("rms_norm_0", (1, 4, 8)))
    g.outputs = [rms]
    return g


def _matmul_graph() -> tuple[Graph, str, str]:
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8, 4)))
    b = g.add_node(InputOp(), [], Tensor("B", (4, 8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(op="multiply"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(op="sum", axis=1), [ew], Tensor("C", (4, 1, 4)))
    g.outputs = [c]
    return g, ew, c


def test_decomposition_mints_pieces_under_one_origin():
    """RmsNormOp decomposes (recursively, incl. its inner mean) into many
    primitives, all distinct pieces of the single ``rms_norm_0`` origin."""
    from deplodock.compiler.pipeline import Pipeline

    out = Pipeline.build(["frontend/decomposition"]).run(_rms_graph())

    pieces: set[str] = set()
    for node in out.nodes.values():
        p = prov.get(node)
        if not p:
            continue  # InputOp / ConstantOp boundary
        assert set(p) == {"rms_norm_0"}
        assert p["rms_norm_0"]["kind"] == "RmsNormOp"
        pieces.update(p["rms_norm_0"]["pieces"])
    # rms_norm fans out into well more than one primitive (mul, mean→sum/div,
    # add-eps, rsqrt, two scales, broadcasts).
    assert len(pieces) >= 5


def test_fusion_aggregates_origins():
    """mul + sum fuse into one LoopOp whose prov covers both source origins."""
    from deplodock.compiler.ir.loop import LoopOp
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline

    g, ew, c = _matmul_graph()
    fused = Pipeline.build(LOOP_PASSES).run(g)

    loops = [n for n in fused.nodes.values() if isinstance(n.op, LoopOp)]
    assert len(loops) == 1
    merged = prov.get(loops[0])
    # both the elementwise and the reduce origins survive the fusion
    assert set(merged) == {ew, c}
    assert merged[ew]["kind"] == "ElementwiseOp"
    assert merged[c]["kind"] == "ReduceOp"


def test_rms_norm_fully_covered_after_full_pipeline():
    """Through decompose→lift→fuse, the rms_norm origin's pieces survive; the
    kernels that carry it collectively realize every piece (full coverage)."""
    from deplodock.compiler.ir.loop import LoopOp
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline

    fused = Pipeline.build(LOOP_PASSES).run(_rms_graph())

    totals = prov.totals(fused)
    assert "rms_norm_0" in totals
    # Union of pieces across all LoopOp kernels == the origin's total pieces:
    # nothing was dropped crossing the fusion boundaries.
    covered: set[str] = set()
    for n in fused.nodes.values():
        if isinstance(n.op, LoopOp):
            covered.update(prov.get(n).get("rms_norm_0", {}).get("pieces", []))
    assert covered == totals["rms_norm_0"]
