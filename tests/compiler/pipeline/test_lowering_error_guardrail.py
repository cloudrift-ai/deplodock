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
from deplodock.compiler.pipeline import LoweringError

# tile IR demolished — pending rebuild; guarded so the
# module collects and its tests register as xfail rather than a collection error.
try:
    from deplodock.compiler.ir.tile.ir import TileOp
except ModuleNotFoundError:
    TileOp = None
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
    g.add_node(op=TileOp(name="k_test"), inputs=["x"], output=Tensor("y", (4,), "f32"), node_id="y")
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


# ---------------------------------------------------------------------------
# Option-0 fallback: a prior that over-extrapolates large (over-budget) tiles
# onto a small shape must not abort the greedy compile. The retry blocklist
# exhausts on the prior-ranked over-budget tiles, then the conservative
# emission-order pick (option-0) recovers the in-budget tile. This is the
# tune-time golden-sweep crash: a prior trained on big square matmuls picked a
# >smem-cap tile for the tiny ``qwen3_06b.q_proj`` projection and the assemble
# raised instead of falling back.
# ---------------------------------------------------------------------------


class _BiggestBNFirstPrior:
    """Stub global prior that ranks leaves by ``BN`` descending — i.e. always
    prefers the largest (over-budget) tile, the way a prior trained on big
    square matmuls extrapolates onto a tiny shape. ``pick`` returns the
    argmax-BN row, so greedy keeps choosing over-budget tiles until the
    blocklist retry budget is exhausted."""

    fitted = True

    def pick(self, rows: list[dict]) -> tuple[int, float]:
        best_i = max(range(len(rows)), key=lambda i: rows[i].get("BN", 0))
        return best_i, 0.0


def _two_pass_tile_pipeline(n_over_budget: int) -> Pipeline:
    """Mirror the real lowering shape: pass 0 (partition → tile ``Fork``) emits
    an in-budget option-0 (``BN=8``, emitted first) followed by
    ``n_over_budget`` over-budget tile leaves (``BN=16, 24, …``); pass 1
    (``100_materialize_tile``) materializes the chosen tile into a ``KernelOp``
    and lets ``validate(ctx)`` filter it. Over-budget tiles only fail at the
    materialize pass (like the real planner emitting tile leaves that pass
    through ``Candidate.try_rewrite``'s validate filter unchecked), so the
    prior can rank them top and the blocklist retry engages per tile identity
    (``BN``). Pass 0 ``RuleSkipped``-guards on the BN marker so it never
    re-fires on its own (already-tiled) output."""
    from deplodock.compiler.pipeline.fork import OptionFork, ThunkFork
    from deplodock.compiler.pipeline.pipeline import RuleSkipped

    def _tile_leaf(bn: int) -> TileOp:
        return TileOp(name="k_test", knobs={"BN": bn})

    def emit_tiles(root):
        if "BN" in root.op.knobs:  # already tiled (our own output) → don't re-fork
            raise RuleSkipped("already tiled")
        leaves = [OptionFork(option=_tile_leaf(8), knobs={"BN": 8})]
        leaves += [OptionFork(option=_tile_leaf(16 + 8 * i), knobs={"BN": 16 + 8 * i}) for i in range(n_over_budget)]
        return ThunkFork(knobs={}, expand_fn=lambda _k: leaves)

    def materialize(root):
        bn = root.op.knobs.get("BN", 0)
        extents = (64,) if bn <= 8 else (4096,)  # 256 B fits the 2 KiB cap; 16 KiB overflows
        return KernelOp(body=[Smem(name="buf", extents=extents, dtype="float")], name="k_test", knobs={"BN": bn})

    emit_rule = Rule(
        name="__emit_tiles__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=emit_tiles,
        param_names=tuple(inspect.signature(emit_tiles).parameters.keys()),
    )
    mat_rule = Rule(
        name="100_materialize_tile",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=materialize,
        param_names=tuple(inspect.signature(materialize).parameters.keys()),
    )
    p0 = Pass(name="__partition__", rules=[emit_rule], index=0)
    p1 = Pass(name="__materialize__", rules=[mat_rule], index=1)
    emit_rule.pass_ = p0
    mat_rule.pass_ = p1
    return Pipeline(passes=[p0, p1])


def test_greedy_run_falls_back_to_option0_when_prior_overflows(monkeypatch):
    # The prior ranks 12 over-budget tiles above the lone in-budget tile, so
    # the blocklist retry can never reach it within its budget. Before the
    # option-0 fallback this raised LoweringError; now ``Pipeline.run`` takes
    # the conservative emission-order pick (option-0 = the in-budget tile).
    import deplodock.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    terminal = _two_pass_tile_pipeline(n_over_budget=12).run(_graph_with_tile(), ctx=_small_smem_ctx())
    op = terminal.nodes["y"].op
    assert isinstance(op, KernelOp), "the in-budget option-0 tile must lower (no LoweringError)"
    assert op.knobs.get("BN") == 8, "the recovered tile is the budget-safe emission-order leaf"


def test_greedy_run_still_raises_when_no_in_budget_option(monkeypatch):
    # The fallback must not paper over a genuinely un-lowerable shape: when
    # EVERY tile is over-budget, option-0 overflows too and the loud
    # LoweringError still fires (no in-budget leaf exists to recover).
    import deplodock.compiler.pipeline.search.policy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_load_prior_safe", lambda: _BiggestBNFirstPrior())
    # Drop the in-budget option-0: shift all leaves over budget by tuning the
    # materializer to overflow for every BN (handled by the all-over-budget
    # single-option pipeline already covered by ``_build_pipeline``).
    with pytest.raises(LoweringError):
        _build_pipeline().run(_graph_with_tile(), ctx=_small_smem_ctx())


def test_run_leaves_no_state_on_pipeline():
    # The rejection sink is Run-scoped (``Run.rejections``), never stashed on
    # the shared frozen Pipeline — a subsequent tune on the same Pipeline sees
    # no sink (silent fork-pruning preserved), and concurrent runs can't
    # clobber each other.
    pipeline = _build_pipeline()
    with pytest.raises(LoweringError):
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())
    assert not hasattr(pipeline, "_lowering_rejections")


# ---------------------------------------------------------------------------
# Per-variant containment: a lowering pass that *raises* (not a validate
# filter) aborts a greedy compile loudly, but under tune is a dropped dead
# end so one un-lowerable search fork can't abort the whole tune. This is the
# stacked defect the static tune-findings report flagged: an un-handled
# fused-cell slab shape (compute_phase_info LoweringError) / an orphan AtomTile
# at render would crash mid-tune with no per-variant containment.
# ---------------------------------------------------------------------------


def _build_raising_pipeline() -> Pipeline:
    """One pass, one rule matching the ``TileOp`` at ``y`` whose rewrite
    *raises* a ``LoweringError`` (an un-lowerable shape a deterministic pass
    chokes on), rather than returning a validate-filtered option."""

    def rewrite(root):  # noqa: ARG001
        raise LoweringError("synthetic un-lowerable shape")

    rule = Rule(
        name="__raising_lower__",
        pattern=[Pattern(name="root", op_type=TileOp)],
        rewrite=rewrite,
        param_names=tuple(inspect.signature(rewrite).parameters.keys()),
    )
    pass_ = Pass(name="__test_raise__", rules=[rule], index=0)
    rule.pass_ = pass_
    return Pipeline(passes=[pass_])


def test_greedy_run_propagates_lowering_exception():
    # Greedy uses ``Run.resolve`` (no containment) — a raising lowering pass
    # propagates loudly, exactly as before.
    pipeline = _build_raising_pipeline()
    with pytest.raises(LoweringError, match="synthetic un-lowerable shape"):
        pipeline.run(_graph_with_tile(), ctx=_small_smem_ctx())


def test_tuning_contains_raising_lowering_pass(caplog):
    # Under tune, ``Run.drive`` catches the lowering exception, drops the
    # candidate's subtree, logs a warning, and finishes without raising —
    # so a single un-lowerable fork can't abort the whole tune.
    import logging

    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.search.db import SearchDB

    pipeline = _build_raising_pipeline()
    with caplog.at_level(logging.WARNING, logger="deplodock.compiler.pipeline"):
        terminals = drain_tune(pipeline, _graph_with_tile(), search=TuningSearch(patience=10**6), ctx=_small_smem_ctx(), db=SearchDB())
    # The only lowering option raised, so no terminal is benchable — the search
    # ends cleanly with zero terminals instead of crashing.
    assert terminals == []
    assert any("dropped un-lowerable candidate" in r.message for r in caplog.records)
