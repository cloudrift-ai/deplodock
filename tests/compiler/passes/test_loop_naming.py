"""Loop-dialect naming rule (``loop/stamp/010_stamp_loop_names``).

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

_STAMP_MODULE = importlib.import_module("deplodock.compiler.pipeline.passes.loop.stamp.010_stamp_loop_names")


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


def test_single_kernel_linear_has_no_qualifier_under_sm90_fold():
    """A bias-free Linear lowers to one kernel that IS the whole op, so the
    name is ``k_linear_<6hex>`` — no ``_reduce``. Regression: the sm_90+
    weight-transpose fold (``050_fold_into_constant``) used to strand a prov
    piece on the folded ConstantOp, inflating ``totals`` so the kernel read
    partial coverage and kept the qualifier."""
    import re

    from deplodock.compiler import target as target_mod
    from deplodock.compiler.ir.base import ConstantOp
    from deplodock.compiler.ir.frontend.ir import LinearOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8, 16)), node_id="x")
    g.add_node(
        ConstantOp(name="w", source_path="linear.weight", source_shape=(16, 16), source_dtype="float32"),
        [],
        Tensor("w", (16, 16)),
        node_id="w",
    )
    g.add_node(LinearOp(), ["x", "w"], Tensor("linear_0", (8, 16)), node_id="linear_0")
    g.inputs, g.outputs = ["x"], ["linear_0"]

    target_mod.set_target((9, 0))
    try:
        out = Pipeline.build(LOOP_PASSES).run(g)
    finally:
        target_mod.set_target(None)

    names = [n.op.name for n in out.nodes.values() if isinstance(n.op, LoopOp)]
    assert len(names) == 1, names
    assert re.fullmatch(r"k_linear_[0-9a-f]{6}", names[0]), names[0]


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


def test_name_for_drops_weak_kinds_when_strong_present():
    """Layout/plumbing origins (transpose / reshape / unsqueeze / cat / slice)
    never label a kernel that also carries a strong op — RoPE plumbing fused
    into attention stays ``k_sdpa_…``, not ``k_sdpa_cat_slice_…``."""
    loop = _pointwise_loop()
    prov_map = {
        "sdpa_0": {"kind": "SdpaOp", "pieces": ["p1", "p2"]},
        "cat_0": {"kind": "CatOp", "pieces": ["p3"]},
        "slice_0": {"kind": "SliceOp", "pieces": ["p4"]},
        "transpose_0": {"kind": "TransposeOp", "pieces": ["p5"]},
    }
    totals = {oid: set(e["pieces"]) for oid, e in prov_map.items()}
    name = prov.name_for(loop, "o", prov_map, totals)
    assert name.startswith("k_sdpa_") and "cat" not in name and "slice" not in name and "transpose" not in name


def test_name_for_pure_weak_kernel_uses_weak_label():
    """A kernel whose only origins are layout ops (e.g. a standalone cat copy)
    still gets the descriptive weak label, not the node-id fallback."""
    loop = _pointwise_loop()
    prov_map = {"cat_0": {"kind": "CatOp", "pieces": ["o"]}}
    totals = {"cat_0": {"o"}}
    name = prov.name_for(loop, "merged_lift_n42", prov_map, totals)
    assert name.startswith("k_cat_")


def test_name_for_order_is_dominant_first_and_merge_order_independent():
    """Labels sort by descending piece count (the op the kernel mostly
    implements leads), so the name is the same whatever order fusion merged
    the origins in."""
    loop = _pointwise_loop()
    sdpa = {"kind": "SdpaOp", "pieces": ["p1", "p2", "p3"]}
    linear = {"kind": "LinearOp", "pieces": ["p4"]}
    totals = {"sdpa_0": {"p1", "p2", "p3"}, "linear_0": {"p4", "elsewhere"}}
    a = prov.name_for(loop, "o", {"sdpa_0": dict(sdpa), "linear_0": dict(linear)}, totals)
    b = prov.name_for(loop, "o", {"linear_0": dict(linear), "sdpa_0": dict(sdpa)}, totals)
    assert a == b
    assert a.startswith("k_sdpa_linear_")


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
