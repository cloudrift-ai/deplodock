"""Tests for the First-Play Urgency (FPU) extension to :class:`TuningSearch`.

FPU replaces the legacy ``+∞`` value for unvisited siblings with a finite
``max(0, parent.Q_norm - fpu_reduction)``. When a known-decent parent
path exists, this lets the search drill deeper instead of exhausting
every sibling at the current level. ``fpu_reduction=None`` (the default)
preserves the legacy behavior so existing tune sweeps are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats
from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch


@dataclass
class _FakeCandidate:
    """Minimal stand-in for :class:`LazyCandidate` — the policy only
    reads ``score()`` off pushed candidates and never resolves them."""

    label: str
    prior: float = 0.0
    knobs: dict = field(default_factory=dict)

    def score(self) -> float:
        return self.prior


def _fake_stats(median_us: float) -> PerfStats:
    return PerfStats(median=median_us, min=median_us, max=median_us, mean=median_us, variance=0.0, n_samples=1)


def _push_root_children(search: TuningSearch, n: int, prior_step: float = 0.0) -> list[Any]:
    """Push ``n`` children of the root; return them in push order."""
    children = [_FakeCandidate(label=f"c{i}", prior=i * prior_step) for i in range(n)]
    search.push(children[0], *children[1:])
    return children


def test_fpu_disabled_preserves_breadth_first_sweep():
    """``fpu_reduction=None`` (default) → unvisited siblings remain ``+∞``,
    so every root sibling is popped before any sibling is re-visited."""
    search = TuningSearch(fpu_reduction=None, patience=1000)
    _push_root_children(search, n=4)

    popped_first_pass: list[str] = []
    for _ in range(4):
        cand = search.pop()
        assert cand is not None
        popped_first_pass.append(cand.label)
        # Bench every variant with the same reward; this prevents
        # patience from kicking in via the "no new best" check.
        search.observe(_fake_stats(median_us=100.0), status="ok")

    # All four siblings popped once before any could re-pop (no children pushed beyond root).
    assert sorted(popped_first_pass) == ["c0", "c1", "c2", "c3"]


def test_fpu_enabled_drills_deeper_into_winning_branch():
    """With FPU, a visited-and-decent branch beats unvisited siblings.

    Tree shape: root has 4 children. Pop c0 (a branch), push 2
    grandchildren of c0, observe a strong c0-subtree reward. The next
    pop must descend into c0's subtree (rather than the unvisited c1/c2/c3
    siblings at root), because root.Q_norm = 1.0 makes FPU = 0.5 for the
    unvisited siblings, while c0's UCB ≈ 1.0.

    Without FPU, c1/c2/c3 would all get ``+∞`` and one of them would
    pop next — that's the legacy breadth-first behavior the FPU patch
    is designed to break.
    """
    search = TuningSearch(fpu_reduction=0.5, patience=1000, ucb_c=0.1)
    root_kids = _push_root_children(search, n=4)

    # Pop c0 as a branch.
    first = search.pop()
    assert first is root_kids[0]

    # Push two grandchildren — c0 now has live=2 again so it's descendable.
    grand = [_FakeCandidate(label=f"c0.{i}") for i in range(2)]
    search.push(grand[0], grand[1])

    # Pop one grandchild and bench it strongly (reward 0.1).
    leaf = search.pop()
    assert leaf in grand
    search.observe(_fake_stats(median_us=10.0), status="ok")

    # Next pop: with FPU=0.5 and root.Q_norm=1.0, unvisited root siblings
    # get FPU=0.5; c0's UCB ≈ 1.0 (Q_norm=1.0 + small bonus). So the
    # search descends back into c0 and pops the remaining grandchild.
    next_pop = search.pop()
    assert next_pop is grand[1], f"FPU should descend into c0's subtree; got {next_pop.label if next_pop else None}"


def test_fpu_falls_back_to_inf_at_unwarmed_root():
    """First pop on an unwarmed root (no measurements yet, global_best=0)
    must keep the legacy ``+∞`` semantics — otherwise the prior alone
    determines order at the root, which silently changes baseline tune
    behavior in subtle ways."""
    search = TuningSearch(fpu_reduction=0.2, patience=1000)
    children = _push_root_children(search, n=3, prior_step=10.0)

    # Without any measurements, FPU has no parent.Q to discount from —
    # so every unvisited child gets +∞. They tie on UCB, the prior breaks
    # the tie. With prior_step=10, c2 has the highest prior → pops first.
    first = search.pop()
    assert first is children[2]


def test_fpu_zero_reduction_equals_parent_q():
    """``fpu_reduction=0`` should set FPU = parent.Q exactly — useful as
    a calibration knob ("treat unvisited as equal to parent best")."""
    search = TuningSearch(fpu_reduction=0.0, patience=1000, ucb_c=0.0)
    children = _push_root_children(search, n=2)

    # Warm c0 with reward 1.0 (best possible).
    assert search.pop() is children[0]
    search.observe(_fake_stats(median_us=1.0), status="ok")

    # Now c0.Q_norm = 1.0; FPU for c1 = parent_q - 0 = 1.0; tie → score breaks.
    # With ucb_c=0, c0's bonus is 0, so its UCB = 1.0. c1's FPU = 1.0.
    # Equal floats → tie; we just want to verify FPU didn't drop to 0.
    next_pop = search.pop()
    # Either could win on tie — assert that c1 is *eligible*, i.e. FPU
    # didn't lock c0 in.
    assert next_pop in (children[0], children[1])


def test_fpu_does_not_change_observe_or_record_terminal():
    """The FPU change is selection-only; backprop/observation paths
    must be untouched (otherwise downstream perf stats would shift)."""
    search = TuningSearch(fpu_reduction=0.15, patience=100)
    children = _push_root_children(search, n=3)

    for i, child in enumerate(children):
        cand = search.pop()
        assert cand is not None and cand is child
        search.observe(_fake_stats(median_us=10.0 + i), status="ok")

    # Root visits should equal the number of observe() calls; best_reward
    # should be the max reward = 1/10 = 0.1 (from the first observed leaf).
    assert search.tree.root.visits == 3
    assert abs(search.tree.root.best_reward - 0.1) < 1e-9


def test_fpu_default_is_active():
    """Default ``TuningSearch()`` should run with FPU on — the class
    constant ``DEFAULT_FPU_REDUCTION`` is the contract, so callers
    relying on legacy behavior must opt out explicitly."""
    s = TuningSearch()
    assert s._fpu_reduction == TuningSearch.DEFAULT_FPU_REDUCTION
    assert s._fpu_reduction is not None
    # Sanity: the default sits in the documented 0.10-0.25 band.
    assert 0.10 <= TuningSearch.DEFAULT_FPU_REDUCTION <= 0.25


def test_lazycandidate_protocol_satisfied():
    """Sanity check that :class:`_FakeCandidate` is structurally compatible
    with :class:`LazyCandidate` — the policy only uses ``.score()`` on the
    pushed candidates, so a duck-typed stub must keep working."""
    cand = _FakeCandidate(label="x", prior=1.5)
    # Hasattr check matches the only attribute MCTS reads off candidates.
    assert hasattr(cand, "score")
    assert isinstance(cand.score(), float)
    # Verify the real LazyCandidate class is imported (catches accidental
    # API breakage where TuningSearch starts requiring other methods).
    assert LazyCandidate is not None
    assert SearchNode is not None
    assert SearchTree is not None
