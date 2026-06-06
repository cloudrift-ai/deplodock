"""The knob-stamp invariant: every knob-owning lowering pass records its decision
on every non-idempotency path — the chosen value when it acts, an explicit
off/default when it declines or is gated off — so a realized kernel config never
leaves a knob absent.

Why it matters: the learned ``CatBoostPrior`` 0-fills absent feature columns, so a
knob a pass never stamped is indistinguishable from one explicitly set to off.
The optimistic value-of-position branch rows then make that "absent-knob" region
look fast, and the greedy pick collapses onto a degenerate config (the
``square.512.fp16`` tune that dragged the fp32 ``STAGE=00`` pick to 36.9 µs). With
every pass stamping, realized leaves carry a complete, uniform knob set and are
featurized in the same region as their own truthful training rows.

Per-pass unit tests cover the transformational / fallback passes that previously
``RuleSkipped`` without stamping; the holistic test checks a compiled matmul leaf
carries the full staging / transport / structural / kernel-marker knob set.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import ThreadTile, TileOp
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline, RuleSkipped

# Pass modules start with a digit — import via importlib like the pipeline loader.
_async = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.060_use_async_copy")
_pipe = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.080_pipeline_stages")
_ring = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.040_use_ring_buffers")


class _Node:
    """Minimal node stub exposing ``.op`` (all the passes here read)."""

    def __init__(self, op: TileOp) -> None:
        self.op = op


def _bare_tile(knobs: dict | None = None) -> TileOp:
    """A TileOp with no SerialTile(serial_outer) / StageBundle — every transport
    pass declines on it (nothing to promote / pipeline / buffer)."""
    body = Body((ThreadTile(axes=(Axis("t", 32),), body=Body(())),))
    return TileOp(body=body, name="k", knobs=knobs or {})


# --- 060_use_async_copy ------------------------------------------------------


def test_async_copy_off_stamped_pre_sm80():
    """cp.async needs sm_80+; below that ASYNC_COPY defaults off → records False."""
    res = _async.rewrite(Context(compute_capability=(7, 0)), _Node(_bare_tile()))
    assert isinstance(res, TileOp)
    assert res.knobs[_async.ASYNC_COPY.name] is False


def test_async_copy_off_when_no_buffered_bundle():
    """sm_80+, but no BUFFERED bundle to promote → records ASYNC_COPY=False."""
    res = _async.rewrite(Context(compute_capability=(8, 0)), _Node(_bare_tile()))
    assert res.knobs[_async.ASYNC_COPY.name] is False


def test_async_copy_idempotent():
    op = _bare_tile({_async.ASYNC_COPY.name: False})
    with pytest.raises(RuleSkipped, match="already decided"):
        _async.rewrite(Context(compute_capability=(8, 0)), _Node(op))


# --- 080_pipeline_stages -----------------------------------------------------


def test_pipeline_stages_off_when_nothing_to_pipeline():
    res = _pipe.rewrite(Context(compute_capability=(9, 0)), _Node(_bare_tile()))
    assert isinstance(res, TileOp)
    assert res.knobs[_pipe.PIPELINE_STAGES.name] is False


def test_pipeline_stages_idempotent():
    op = _bare_tile({_pipe.PIPELINE_STAGES.name: True})
    with pytest.raises(RuleSkipped, match="already decided"):
        _pipe.rewrite(Context(compute_capability=(9, 0)), _Node(op))


# --- 040_use_ring_buffers ----------------------------------------------------


def test_buffer_count_fallback_when_nothing_fits():
    """No promotable SYNC ring → fall back to BUFFER_COUNT=1 (single buffer / no
    ring), a single non-fork variant, instead of RuleSkipping."""
    res = _ring.rewrite(Context(compute_capability=(8, 0)), _Node(_bare_tile()))
    assert isinstance(res, list) and len(res) == 1
    assert res[0].knobs[_ring.BUFFER_COUNT.name] == 1


# --- holistic: a compiled matmul leaf carries every knob ---------------------

# Knobs every matmul TileOp must carry after lowering — the planner's tile
# geometry plus every staging / transport / structural decision. (MMA / WM / WN
# are tensor-core-path only, so they're not in this scalar-fp32 universal set.)
_TILE_KNOBS = {
    "BM", "BN", "BK", "BR", "FM", "FN", "FK", "SPLITK",
    "STAGE", "GROUP_M", "ATOMIC_FREE_SPLITK", "HOIST_COMPUTE",
    "BUFFER_COUNT", "TMA", "ASYNC_COPY", "PAD_SMEM", "PIPELINE_STAGES", "WARP_SPECIALIZE",
}  # fmt: skip
_KERNEL_MARKERS = {"VECTORIZE_LOADS", "INTERLEAVE_LOADS", "PERMUTE_LANES"}


def _matmul_kernel_knobs(g: Graph) -> dict:
    """The knob dict of the lowered matmul kernel — the kernel-bearing op whose
    knobs carry the planner's ``BK`` (uniquely the matmul, not a sibling reduce)."""
    for node in g.nodes.values():
        knobs = getattr(node.op, "knobs", None)
        if knobs and "BK" in knobs:
            return knobs
    raise AssertionError("no matmul kernel op found in lowered graph")


def test_compiled_matmul_leaf_has_complete_knob_set():
    """End-to-end: a lowered matmul leaf carries a complete, uniform knob set —
    every knob-owning pass recorded its decision. Pinned ``--target`` (sm_80) so
    the greedy pick is deterministic GPU-less; the knob *set* is pick- and
    cc-independent (each fork branch stamps its knob), only the values vary."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (256, 256)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (256, 256)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (256, 256)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    out = Pipeline.build(KERNEL_PASSES).run(g, ctx=Context(compute_capability=(8, 0)))
    knobs = _matmul_kernel_knobs(out)
    missing = (_TILE_KNOBS | _KERNEL_MARKERS) - set(knobs)
    assert not missing, f"matmul leaf missing knobs: {sorted(missing)} (has {sorted(knobs)})"
