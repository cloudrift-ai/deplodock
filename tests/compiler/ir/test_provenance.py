"""Tests for op provenance (``deplodock.compiler.provenance``).

Two surfaces in one file: the data model + seeding (``seed`` / ``put`` /
``get`` / ``union`` / ``mint`` / ``totals``), and propagation through the
real passes (decomposition mints pieces under one origin; fusion
aggregates pieces across origins via the ``Graph.splice`` hook). Same
``_graph`` fixture serves both.
"""

from deplodock.compiler import provenance as prov
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def _graph() -> tuple[Graph, str, str, str, str]:
    """C = Reduce{sum}(Elementwise{mul}(A, B)) — two compute nodes, two inputs."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8, 4)))
    b = g.add_node(InputOp(), [], Tensor("B", (4, 8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(op="multiply"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(op="sum", axis=1), [ew], Tensor("C", (4, 1, 4)))
    g.outputs = [c]
    return g, a, b, ew, c


# --- seed ---


def test_seed_stamps_compute_skips_boundary():
    g, a, b, ew, c = _graph()
    prov.seed(g)
    assert prov.get(g.nodes[ew]) == {ew: {"kind": "ElementwiseOp", "pieces": [ew]}}
    assert prov.get(g.nodes[c]) == {c: {"kind": "ReduceOp", "pieces": [c]}}
    # InputOp sentinels compute nothing — never seeded.
    assert prov.get(g.nodes[a]) == {}
    assert prov.get(g.nodes[b]) == {}


def test_put_skips_boundary_and_clears_stale_prov():
    """Boundary sentinels never carry prov: ``put`` onto one is a no-op, and it
    clears a stale hint (e.g. prov copied by ``Graph.splice``'s generic hint
    merge when a fold collapses a compute op into a constant)."""
    g, a, _, _, _ = _graph()
    prov.put(g.nodes[a], {"o": {"kind": "LinearOp", "pieces": ["p"]}})
    assert prov.get(g.nodes[a]) == {}
    # A stale hint planted behind put's back is scrubbed by the next put.
    g.nodes[a].hints.set(prov.PROV, {"o": {"kind": "LinearOp", "pieces": ["p"]}})
    prov.put(g.nodes[a], {"o": {"kind": "LinearOp", "pieces": ["q"]}})
    assert prov.get(g.nodes[a]) == {}


def test_seed_idempotent():
    g, _, _, ew, _ = _graph()
    prov.seed(g)
    prov.put(g.nodes[ew], {"custom": {"kind": "X", "pieces": ["p"]}})
    prov.seed(g)  # must not overwrite an already-seeded node
    assert prov.get(g.nodes[ew]) == {"custom": {"kind": "X", "pieces": ["p"]}}


# --- helpers ---


def test_union_merges_pieces_keeps_kind():
    a = {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2"]}}
    b = {"o": {"kind": "RmsNormOp", "pieces": ["p2", "p3"]}, "q": {"kind": "LinearOp", "pieces": ["x"]}}
    u = prov.union(a, b)
    assert u["o"] == {"kind": "RmsNormOp", "pieces": ["p1", "p2", "p3"]}
    assert u["q"] == {"kind": "LinearOp", "pieces": ["x"]}


def test_mint_fresh_piece_per_origin():
    m = prov.mint({"o": "RmsNormOp", "q": "LinearOp"}, "newnode")
    assert m == {
        "o": {"kind": "RmsNormOp", "pieces": ["newnode"]},
        "q": {"kind": "LinearOp", "pieces": ["newnode"]},
    }


def test_origins():
    p = {"o": {"kind": "RmsNormOp", "pieces": ["p1"]}, "q": {"kind": "LinearOp", "pieces": ["x"]}}
    assert prov.origins(p) == {"o": "RmsNormOp", "q": "LinearOp"}


def test_totals_and_coverage():
    g, _, _, ew, c = _graph()
    prov.put(g.nodes[ew], {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2"]}})
    prov.put(g.nodes[c], {"o": {"kind": "RmsNormOp", "pieces": ["p3"]}})
    t = prov.totals(g)
    assert t["o"] == {"p1", "p2", "p3"}
    assert prov.coverage(prov.get(g.nodes[ew]), t)["o"] == (2, 3, False)

    # A node holding every piece is "full".
    prov.put(g.nodes[c], {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2", "p3"]}})
    t = prov.totals(g)
    assert prov.coverage(prov.get(g.nodes[c]), t)["o"] == (3, 3, True)


# --- serialization ---


def test_prov_roundtrips_through_serialization():
    g, _, _, ew, _ = _graph()
    prov.put(g.nodes[ew], {"rms_0": {"kind": "RmsNormOp", "pieces": ["sq", "mean"]}})
    g2 = Graph.from_dict(g.to_dict())
    got = prov.get(g2.nodes[ew])
    assert got == {"rms_0": {"kind": "RmsNormOp", "pieces": ["sq", "mean"]}}
    assert isinstance(got["rms_0"]["pieces"], list)  # stays a list, not stringified


# --- propagation through real passes ---


def _rms_graph() -> Graph:
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)))
    w = g.add_node(InputOp(), [], Tensor("w", (8,)))
    g.inputs = [x, w]
    rms = g.add_node(RmsNormOp(), [x, w], Tensor("rms_norm_0", (1, 4, 8)))
    g.outputs = [rms]
    return g


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

    g, _, _, ew, c = _graph()
    fused = Pipeline.build(LOOP_PASSES).run(g)

    loops = [n for n in fused.nodes.values() if isinstance(n.op, LoopOp)]
    assert len(loops) == 1
    merged = prov.get(loops[0])
    # both the elementwise and the reduce origins survive the fusion
    assert set(merged) == {ew, c}
    assert merged[ew]["kind"] == "ElementwiseOp"
    assert merged[c]["kind"] == "ReduceOp"


def test_constant_fold_strands_no_prov_on_boundary():
    """``050_fold_into_constant`` (sm_90+ TMA path) collapses the weight
    transpose minted by the Linear decomposition into a parameter ConstantOp.
    The splice's generic hint merge used to copy the transpose's prov onto that
    constant, inflating ``totals`` so the matmul kernel read partial coverage
    (the spurious ``k_linear_reduce`` instead of ``k_linear``). No boundary node
    may carry prov, and ``totals`` must equal the union over compute nodes."""
    from deplodock.compiler import target as target_mod
    from deplodock.compiler.ir.base import ConstantOp
    from deplodock.compiler.ir.frontend.ir import LinearOp
    from deplodock.compiler.pipeline import Pipeline

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (8, 16)))
    w = g.add_node(
        ConstantOp(name="w", source_path="linear.weight", source_shape=(16, 16), source_dtype="float32"),
        [],
        Tensor("w", (16, 16)),
    )
    g.inputs = [x]
    lin = g.add_node(LinearOp(), [x, w], Tensor("linear_0", (8, 16)))
    g.outputs = [lin]

    target_mod.set_target((9, 0))
    try:
        out = Pipeline.build(["frontend/decomposition"]).run(g)
    finally:
        target_mod.set_target(None)

    folded = [n for n in out.nodes.values() if isinstance(n.op, ConstantOp) and n.op.load_ops]
    assert folded, "the weight transpose should have been folded into the ConstantOp"
    for node in out.nodes.values():
        if prov.is_boundary(node.op):
            assert prov.get(node) == {}, f"boundary node carries prov: {prov.get(node)}"
    compute_pieces: set[str] = set()
    for node in out.nodes.values():
        compute_pieces.update(prov.get(node).get("linear_0", {}).get("pieces", []))
    assert prov.totals(out)["linear_0"] == compute_pieces


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
