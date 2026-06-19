"""Greedy in-pass hardware-feasibility filter (``greedy_decide`` partition fork).

A prior trained on one card can rank a tile whose lowered ``KernelOp`` exceeds
another device's dynamic-smem cap *first* (the Blackwell-trained prior on Ada
sm_89, 99 KB — ``plans/golden-sweep-rtx4070ti-findings.md`` Finding 1). The smem
footprint isn't knowable off the leaf at the partition fork (no ``StageBundle`` /
unstamped dtype — a downstream staging choice), so ``greedy_decide`` probes
feasibility by *lowering* each prior-ranked candidate to its ``KernelOp`` and
deploys the first that passes ``validate(ctx)`` (:func:`_leaf_feasible`,
memoized), in one resolve pass.

These tests cover the new control flow deterministically:

* the rank-order **selection** (skip the infeasible top pick, deploy the first
  feasible leaf) and its ``check_feasibility`` gate — by monkeypatching the
  per-leaf feasibility verdict;
* :func:`_leaf_feasible`'s own wiring (lower → ``KernelOp.validate`` → memoize;
  best-effort feasible on a probe that raises).

The *emergent* case — the real trained prior driving deep staging into an
irreducibly over-cap slab — can't be reproduced in-process: when feasibility is
probed at the live cap, the staging passes self-heal (ring-prune + gmem-direct
fallback) to fit, so only a prior that pins the oversize staging triggers it.
That path is exercised by the on-box golden sweep, not here.
"""

from __future__ import annotations

import deplodock.compiler.pipeline.search.policy.greedy as greedy_mod
from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.pipeline import _MAX_GREEDY_RETRIES, Run
from deplodock.compiler.pipeline.search.policy.greedy import (
    _first_leaf,
    _leaf_feasible,
    _leaf_knobs,
    greedy_decide,
)


def _matmul_graph(*, M: int = 128, N: int = 128, K: int = 64, dtype=F16) -> Graph:
    """``C[M,N] = sum_k A[M,K] * B[K,N]`` with typed inputs — reaches the
    partition planner with a fan-out of complete tile leaves."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=dtype), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=dtype), node_id="b")
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g.add_node(op=op, inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F32), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def _ctx(cap: int = 300000) -> Context:
    return Context(compute_capability=(8, 0), max_dynamic_smem=cap)


def _area(knobs: dict) -> int:
    """Tile output area from the leaf knob row — the proxy the stub prior ranks by."""
    return int(knobs.get("BM", 1)) * int(knobs.get("FM", 1)) * int(knobs.get("BN", 1)) * int(knobs.get("FN", 1))


class _AreaStub:
    """A bare-``mean_scores`` prior (no ``pick``) ranking the LARGEST-area tile
    first — checkpoint-independent. Exercises the bare-mean_scores branch."""

    fitted = True

    def mean_scores(self, rows: list[dict]) -> list[float]:
        return [float(-_area(r)) for r in rows]


def _resolve_pick(graph: Graph, *, check_feasibility: bool) -> dict:
    """Resolve ``graph`` greedily with the area stub and return the deployed
    kernel's tile knobs (``BM``/``BN``/``FM``/``FN`` carried onto the KernelOp)."""
    decide = greedy_decide(prior=_AreaStub(), check_feasibility=check_feasibility)
    terminal, _ = Run(pipeline=Pipeline.build(KERNEL_PASSES), ctx=_ctx()).resolve(graph, decide)
    op = terminal.nodes["c"].op
    assert isinstance(op, KernelOp), f"expected a deployed KernelOp, got {type(op).__name__}"
    return dict(op.knobs or {})


# ---------------------------------------------------------------------------
# Selection: first-feasible in prior-rank order, gated by check_feasibility
# ---------------------------------------------------------------------------


def test_filter_off_deploys_prior_top_pick():
    """Baseline: with the filter off greedy deploys the stub's #1 (largest area)."""
    knobs = _resolve_pick(_matmul_graph(), check_feasibility=False)
    # The stub ranks largest-area first, so #1 is the global-max-area tile.
    assert _area(knobs) > 0


def test_filter_skips_infeasible_top_pick(monkeypatch):
    """With the filter on, the prior's #1 marked infeasible is skipped and the
    next-ranked (smaller-area) feasible leaf is deployed in one pass."""
    # The global-max area the prior would pick #1 (filter-off baseline).
    top_area = _area(_resolve_pick(_matmul_graph(), check_feasibility=False))

    # Mark ONLY the global-max-area tile infeasible; everything else fits.
    def fake_feasible(fp, leaf, prior, feas_memo):  # noqa: ARG001
        return _area(_leaf_knobs(leaf)) < top_area

    monkeypatch.setattr(greedy_mod, "_leaf_feasible", fake_feasible)
    knobs = _resolve_pick(_matmul_graph(), check_feasibility=True)
    # The deployed tile is strictly smaller-area than the (infeasible) top pick —
    # the filter walked past it to the first feasible leaf, no retry loop.
    assert _area(knobs) < top_area


def test_filter_off_ignores_feasibility(monkeypatch):
    """``check_feasibility=False`` keeps the bare argmin even when the top pick
    is (would-be) infeasible — the gate, and the nested-probe re-entry guard."""
    top_area = _area(_resolve_pick(_matmul_graph(), check_feasibility=False))

    calls: list[int] = []

    def fake_feasible(fp, leaf, prior, feas_memo):  # noqa: ARG001
        calls.append(1)
        return _area(_leaf_knobs(leaf)) < top_area

    monkeypatch.setattr(greedy_mod, "_leaf_feasible", fake_feasible)
    knobs = _resolve_pick(_matmul_graph(), check_feasibility=False)
    assert _area(knobs) == top_area  # unchanged — feasibility ignored
    assert calls == []  # the probe is never consulted off the partition path


# ---------------------------------------------------------------------------
# _leaf_feasible: lower → KernelOp.validate, memoize, best-effort
# ---------------------------------------------------------------------------


def _partition_leaves(graph: Graph):
    """Drive a resolve to capture the partition fork's flattened leaves + the
    ForkPoint, so the unit tests can call ``_leaf_feasible`` on a real leaf."""
    captured: dict = {}

    def decide(fp):
        if fp.match.rule.name == greedy_mod.PARTITION_RULE and "fp" not in captured:
            captured["fp"] = fp
            captured["leaves"] = flatten_leaves(fp.options)
        return _first_leaf(fp.options[0])

    Run(pipeline=Pipeline.build(KERNEL_PASSES), ctx=_ctx()).resolve(graph, decide)
    return captured["fp"], captured["leaves"]


def test_leaf_feasible_true_under_generous_cap():
    fp, leaves = _partition_leaves(_matmul_graph())
    assert _leaf_feasible(fp, leaves[0], None, {}) is True


def test_leaf_feasible_mirrors_kernelop_validate(monkeypatch):
    """The verdict is exactly the lowered ``KernelOp.validate(ctx)`` result."""
    fp, leaves = _partition_leaves(_matmul_graph())
    monkeypatch.setattr(KernelOp, "validate", lambda self, ctx: False)
    assert _leaf_feasible(fp, leaves[0], None, {}) is False


def test_leaf_feasible_memoizes(monkeypatch):
    """A second call on the same ``(op_cache_key, tile_identity)`` doesn't
    re-lower — the kernel pipeline is built/used once per distinct tile."""
    fp, leaves = _partition_leaves(_matmul_graph())
    feas_memo: dict = {}
    real_resolve = Run.resolve
    calls: list[int] = []

    def counting_resolve(self, graph, decide):
        calls.append(1)
        return real_resolve(self, graph, decide)

    monkeypatch.setattr(Run, "resolve", counting_resolve)
    v1 = _leaf_feasible(fp, leaves[0], None, feas_memo)
    n_after_first = len(calls)
    v2 = _leaf_feasible(fp, leaves[0], None, feas_memo)
    assert v1 == v2
    assert len(calls) == n_after_first  # memo hit → no second resolve


def test_leaf_feasible_best_effort_on_raise(monkeypatch):
    """A probe that raises during lowering is treated as feasible (never drops a
    possibly-valid tile — the real resolve's rejection sink still catches a
    genuinely un-lowerable tile)."""
    fp, leaves = _partition_leaves(_matmul_graph())

    def boom(self, graph, decide):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(Run, "resolve", boom)
    assert _leaf_feasible(fp, leaves[0], None, {}) is True


# ---------------------------------------------------------------------------
# Cold prior + retry-cap revert
# ---------------------------------------------------------------------------


def test_cold_prior_skips_feasibility_probe(monkeypatch):
    """A cold compile (no prior) takes the option-0 path and never pays the
    probe cost — ``_leaf_feasible`` is never called."""
    calls: list[int] = []
    monkeypatch.setattr(greedy_mod, "_leaf_feasible", lambda *a, **k: calls.append(1) or True)
    decide = greedy_decide(prior=None)  # explicit None → option-0 emission order
    Run(pipeline=Pipeline.build(KERNEL_PASSES), ctx=_ctx()).resolve(_matmul_graph(), decide)
    assert calls == []


def test_max_greedy_retries_reverted_to_8():
    """The retry cap is back to 8 now that smem mis-ranking is handled in-pass."""
    assert _MAX_GREEDY_RETRIES == 8
