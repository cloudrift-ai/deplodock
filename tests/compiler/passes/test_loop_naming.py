"""Loop-dialect naming rule (``loop/fusion/030_stamp_loop_names``).

Companion to ``test_tile_naming``: that file checks the name lands on the
final TileOp; this one checks the stamping rule itself — every LoopOp gets
a non-empty ``name`` after the loop dialect runs, the rule is idempotent,
and the unit naming function (:func:`provenance.name_for`) covers all
three branches (glue-only fallback, fully-covered single op, multi-op /
partial coverage)."""

from __future__ import annotations

import importlib

from deplodock.compiler import provenance as prov
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.frontend.ir import RmsNormOp
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline

_STAMP_MODULE = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.030_stamp_loop_names")


def _pointwise_loop() -> LoopOp:
    i, j = Axis("i", 4), Axis("j", 8)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
                        ),
                    ),
                ),
            ),
        )
    )


# ---------------------------------------------------------------------------
# End-to-end: every LoopOp gets stamped by the loop dialect
# ---------------------------------------------------------------------------


def test_every_loop_op_has_name_after_loop_passes():
    """After ``LOOP_PASSES`` runs, every surviving LoopOp carries a non-empty
    ``name`` shaped ``k_…``. ``RmsNormOp`` decomposes + fuses into multiple
    LoopOps; all of them must be named."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("rms_norm_0", (1, 4, 8)), node_id="rms_norm_0")
    g.inputs, g.outputs = ["x", "w"], ["rms_norm_0"]

    out = Pipeline.build(LOOP_PASSES).run(g)
    loops = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)]
    assert loops, "rms_norm should lower to at least one LoopOp"
    assert all(n.op.name and n.op.name.startswith("k_") for n in loops), [n.op.name for n in loops]


# ---------------------------------------------------------------------------
# Rule idempotence
# ---------------------------------------------------------------------------


def test_stamp_rule_is_idempotent():
    """Once a LoopOp has a name, the rule skips on a second invocation."""

    class _StubMatch:
        def __init__(self, graph: Graph) -> None:
            self.graph = graph

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(_pointwise_loop(), ["x"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    prov.seed(g)

    root = g.nodes["o"]
    stamped = _STAMP_MODULE.rewrite(match=_StubMatch(g), root=root)
    assert stamped is not None and stamped.name

    # Second pass over the now-named LoopOp must skip (RuleSkipped). The
    # rewrite engine catches RuleSkipped, so importing it here keeps the
    # assertion explicit about which exit the rule takes.
    from deplodock.compiler.pipeline import RuleSkipped

    g.nodes["o"].op = stamped
    try:
        _STAMP_MODULE.rewrite(match=_StubMatch(g), root=g.nodes["o"])
    except RuleSkipped:
        return
    raise AssertionError("rewrite must raise RuleSkipped on an already-named LoopOp")


# ---------------------------------------------------------------------------
# Unit: provenance.name_for branches
# ---------------------------------------------------------------------------


def test_name_for_glue_only_falls_back_to_base_name():
    """A kernel whose only origins are generic glue ops (Elementwise / Reduce
    / IndexMap / …) keeps the node-id base name + reduce|pointwise qualifier."""
    loop = _pointwise_loop()
    prov_map = {"mul_0": {"kind": "ElementwiseOp", "pieces": ["o"]}}
    totals = {"mul_0": {"o"}}
    assert prov.name_for(loop, "o", prov_map, totals) == "k_o_pointwise"


def test_name_for_full_single_op_drops_qualifier():
    """A kernel that fully realizes one meaningful op: ``k_<op>_<h>`` — no
    reduce|pointwise qualifier, just the structural-body hash."""
    loop = _pointwise_loop()
    prov_map = {"rms_norm_0": {"kind": "RmsNormOp", "pieces": ["o"]}}
    totals = {"rms_norm_0": {"o"}}
    name = prov.name_for(loop, "o", prov_map, totals)
    assert name.startswith("k_rms_norm_") and "pointwise" not in name and "reduce" not in name
    # Hash is 6 hex chars; total length is len("k_rms_norm_") + 6 = 17.
    assert len(name) == len("k_rms_norm_") + 6


def test_name_for_partial_coverage_keeps_qualifier():
    """A kernel that holds only some pieces of an origin gets the
    ``_<reduce|pointwise>_<h>`` qualifier — the reduce half is told apart
    from the pointwise tail."""
    loop = _pointwise_loop()
    prov_map = {"rms_norm_0": {"kind": "RmsNormOp", "pieces": ["o"]}}
    # ``totals`` says rms_norm_0 has two pieces graph-wide; this loop only
    # carries one — so the kernel is a partial covering.
    totals = {"rms_norm_0": {"o", "other_piece"}}
    name = prov.name_for(loop, "o", prov_map, totals)
    assert name.startswith("k_rms_norm_pointwise_")


def test_name_for_dedups_repeated_labels():
    """Two distinct origins with the same op kind collapse to one label
    (``_dedup_tokens`` drops adjacent duplicates)."""
    loop = _pointwise_loop()
    prov_map = {
        "rms_norm_0": {"kind": "RmsNormOp", "pieces": ["o"]},
        "rms_norm_1": {"kind": "RmsNormOp", "pieces": ["o"]},
    }
    totals = {"rms_norm_0": {"o"}, "rms_norm_1": {"o"}}
    name = prov.name_for(loop, "o", prov_map, totals)
    # ``rms_norm_rms_norm`` would be the un-deduped joined label; the dedup
    # collapses adjacent duplicates so it stays a single ``rms_norm``.
    assert "rms_norm_rms_norm" not in name
    assert name.startswith("k_rms_norm_")
