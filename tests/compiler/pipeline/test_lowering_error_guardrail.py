"""Loud-error guardrail for the silent ``validate(ctx)`` lowering drop.

When a deterministic (greedy) compile reaches the final tile→kernel
lowering and the produced ``KernelOp`` fails ``validate(ctx)`` (e.g. the
chosen tile shape's materialized smem exceeds ``ctx.max_dynamic_smem``),
``Candidate.try_rewrite`` filters the only option away. Historically this
was silent (DEBUG-only) and the un-lowered ``TileOp`` leaked all the way
to ``CudaBackend``, which raised a cryptic ``non-CudaOp`` ``TypeError``.

``Pipeline.run`` now installs a rejection sink and, after the single
terminal settles, raises a loud :class:`LoweringError` naming the node,
the pass that declined it, and the validate reason. The tuning path
(``Pipeline.tune_async`` / ``TuningSearch``) installs no sink, so the
fork-pruning drop stays silent and a dropped branch is a graceful dead
end — sibling branches carry other tile shapes.

This is the SDPA "silent TileOp leak" failure mode: a scoring change can
nudge the planner into an over-budget QK^T / P@V tile, and without this
guardrail the only symptom was the downstream ``CudaBackend`` mystery.
"""

from __future__ import annotations

import inspect

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.kernel.ir import KernelOp, Smem
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import LoweringError
from deplodock.compiler.pipeline.pipeline import Pass, Pattern, Pipeline, Rule, _raise_on_unlowered
from tests.compiler.conftest import drain_tune


def _small_smem_ctx() -> Context:
    """A ctx whose dynamic-smem cap (2 KiB) is far below the test
    kernel's 16 KiB slab, so ``KernelOp.validate`` rejects it."""
    return Context(compute_capability=(9, 0), max_dynamic_smem=2048)


def _graph_with_tile() -> Graph:
    """``x -> y`` where ``y`` holds a (placeholder) ``TileOp``. The rule's
    rewrite ignores the body, so an empty one is fine."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,), "f32"), node_id="x")
    g.add_node(op=TileOp(body=Body(()), name="k_test"), inputs=["x"], output=Tensor("y", (4,), "f32"), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _over_budget_kernel() -> KernelOp:
    """A ``KernelOp`` whose 16 KiB smem slab exceeds the 2 KiB test cap."""
    return KernelOp(body=[Smem(name="buf", extents=(4096,), dtype="float")], name="k_test")


def _build_pipeline() -> Pipeline:
    """One pass, one rule matching the ``TileOp`` at ``y`` whose rewrite
    always returns the over-budget kernel (so its only option is filtered
    by ``validate(ctx)``)."""

    def rewrite(root):  # noqa: ARG001 — the over-budget kernel is fixed
        return _over_budget_kernel()

    rule = Rule(
        name="__over_budget_lower__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="__test_lower__", rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


# ---------------------------------------------------------------------------
# Unit: _raise_on_unlowered
# ---------------------------------------------------------------------------


def test_raise_on_unlowered_fires_for_stuck_tileop():
    g = _graph_with_tile()
    rejections = [("y", "k:100_materialize_tile", "smem 104960 > max_dynamic_smem 101376")]
    with pytest.raises(LoweringError) as exc:
        _raise_on_unlowered(g, rejections, _small_smem_ctx())
    msg = str(exc.value)
    assert "'y'" in msg
    assert "k:100_materialize_tile" in msg
    assert "smem 104960 > max_dynamic_smem 101376" in msg


def test_no_raise_when_no_rejections():
    # Empty rejection list — even with an un-lowered TileOp present, the
    # guardrail only fires for a *recorded* validate drop.
    _raise_on_unlowered(_graph_with_tile(), [], _small_smem_ctx())


def test_no_raise_when_node_lowered_despite_rejection():
    # A rejection was recorded, but a later rule lowered the node anyway
    # (its terminal op is no longer a TileOp/LoopOp) → stay silent.
    g = _graph_with_tile()
    g.nodes["y"].op = _over_budget_kernel()  # now a KernelOp, i.e. lowered
    _raise_on_unlowered(g, [("y", "k:100_materialize_tile", "smem ...")], _small_smem_ctx())


def test_no_raise_when_rejection_node_absent():
    _raise_on_unlowered(_graph_with_tile(), [("ghost", "k:x", "smem ...")], _small_smem_ctx())


# ---------------------------------------------------------------------------
# Integration: greedy raises, tuning prunes gracefully
# ---------------------------------------------------------------------------


def test_greedy_run_raises_lowering_error():
    pipeline = _build_pipeline()
    with pytest.raises(LoweringError) as exc:
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    msg = str(exc.value)
    assert "'y'" in msg
    # The reason is derived from KernelOp.validate via _validate_reason.
    assert "smem 16384 > max_dynamic_smem 2048" in msg


def test_tuning_does_not_raise_and_prunes_branch():
    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.search.db import SearchDB

    pipeline = _build_pipeline()
    terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    # Tuning yields the (dead) terminal without raising; the node stays a
    # TileOp because its only lowering option was validate-filtered.
    assert terminals, "tuning should still yield the dead terminal"
    assert isinstance(terminals[0].graph.nodes["y"].op, TileOp)


def test_run_leaves_no_state_on_pipeline():
    # The rejection sink is Run-scoped (``Run.rejections``), never stashed on
    # the shared frozen Pipeline — a subsequent tune on the same Pipeline sees
    # no sink (silent fork-pruning preserved), and concurrent runs can't
    # clobber each other.
    pipeline = _build_pipeline()
    with pytest.raises(LoweringError):
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert not hasattr(pipeline, "_lowering_rejections")
