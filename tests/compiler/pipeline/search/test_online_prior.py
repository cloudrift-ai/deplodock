"""Unit tests for the learned prior (``search/prior/``) and its value-of-position
label extraction + PUCT selection in the MCTS tree (``policy/mcts.py``).

These are GPU-less: they exercise the model fit / Thompson draw and the
tree-snapshot row collection on hand-built trees — no benching, no CUDA.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch, _softmax
from deplodock.compiler.pipeline.search.prior import BayesianRidgePrior
from deplodock.compiler.pipeline.search.prior.base import _spearman


def _node(knobs: dict, parent: SearchNode) -> SearchNode:
    """A SearchNode whose candidate exposes ``fork.knobs`` — the only thing
    ``_node_knobs`` reads."""
    return SearchNode(candidate=SimpleNamespace(fork=SimpleNamespace(knobs=knobs)), parent=parent)


# --- model fit -------------------------------------------------------------


def test_noiseless_linear_is_recovered():
    """With a tiny ridge and an exactly-linear target, the standardized ridge
    fit reproduces the labels — the solve is correct end-to-end."""
    rows = []
    for bm in (16, 32, 64, 128):
        for bn in (16, 32, 64):
            rows.append(({"BM": bm, "BN": bn}, 0.01 * bm - 0.02 * bn))
    p = BayesianRidgePrior(seed=0, ridge=1e-6, min_rows=3)
    p.fit(rows)
    preds = np.array([p.mean_score(k) for k, _ in rows])
    ys = np.array([y for _, y in rows])
    assert np.allclose(preds, ys, atol=1e-3)


def test_ranks_lower_latency_higher():
    """label = log(1/us); the prior must score the faster (lower-us) config
    higher. Bigger BM is faster in this synthetic."""
    rows = [({"BM": bm, "BN": 64}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    scores = [p.mean_score({"BM": bm, "BN": 64}) for bm in (16, 32, 64, 128)]
    assert scores == sorted(scores), f"expected monotone-increasing in BM, got {scores}"


def test_partial_knob_dicts_fit_and_score():
    """Rows with different knob key sets (a branch pins only BR; leaves add BM)
    fit through one feature space — the partial row scores without error."""
    rows = [
        ({"BR": 1}, math.log(0.5)),  # branch: value-of-position
        ({"BR": 1, "BM": 32}, math.log(0.4)),
        ({"BR": 1, "BM": 64}, math.log(0.5)),
        ({"BR": 2, "BM": 64}, math.log(0.3)),
    ]
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    # Scoring a never-seen partial state must not raise and must be finite.
    v = p.mean_score({"BR": 1})
    assert math.isfinite(v)


def test_score_is_zero_before_fit():
    p = BayesianRidgePrior(seed=0)
    assert p.score({"BM": 64}) == 0.0
    assert p.mean_score({"BM": 64}) == 0.0


def test_thompson_draw_is_deterministic_until_resample():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p = BayesianRidgePrior(seed=7, min_rows=3)
    p.fit(rows)  # fit() ends with a resample
    a = p.score({"BM": 64})
    b = p.score({"BM": 64})
    assert a == b, "score must be stable under a fixed Thompson draw"
    p.resample()
    # A fresh draw almost surely shifts the value (sanity that resample fires).
    assert p.score({"BM": 64}) != a


def test_seeded_priors_are_reproducible():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p1 = BayesianRidgePrior(seed=3, min_rows=3)
    p2 = BayesianRidgePrior(seed=3, min_rows=3)
    p1.fit(rows)
    p2.fit(rows)
    assert p1.score({"BM": 32}) == p2.score({"BM": 32})


# --- value-of-position labels from the tree --------------------------------


def test_collect_rows_labels_branch_with_best_descendant():
    """A branch shared by two benched leaves is labeled with the BETTER leaf's
    reward (max-propagation), not its own (it has no direct measurement)."""
    tree = SearchTree()
    branch = _node({"BR": 1}, tree.root)
    leaf_slow = _node({"BR": 1, "BM": 32}, branch)
    leaf_fast = _node({"BR": 1, "BM": 64}, branch)
    tree.root.children = [branch]
    branch.children = [leaf_slow, leaf_fast]
    tree.record_terminal(leaf_slow, 1.0)  # reward = 1/us
    tree.record_terminal(leaf_fast, 2.0)  # faster → higher reward

    assert branch.best_reward == 2.0
    rows = TuningSearch(tree=tree)._collect_rows()
    by_knobs = {tuple(sorted(k.items())): lab for k, lab in rows}
    # Branch row carries only its pinned slice, labeled with the best leaf.
    assert abs(by_knobs[(("BR", 1),)] - math.log(2.0)) < 1e-12
    assert abs(by_knobs[(("BM", 64), ("BR", 1))] - math.log(2.0)) < 1e-12
    assert abs(by_knobs[(("BM", 32), ("BR", 1))] - math.log(1.0)) < 1e-12


def test_collect_rows_skips_unbenched_nodes():
    """Nodes with no benched descendant (visits 0 / best_reward 0) are not
    training rows — they are the prediction frontier, not labels."""
    tree = SearchTree()
    visited = _node({"BR": 1, "BM": 64}, tree.root)
    frontier = _node({"BR": 2, "BM": 64}, tree.root)  # never benched
    tree.root.children = [visited, frontier]
    tree.record_terminal(visited, 1.5)
    rows = TuningSearch(tree=tree)._collect_rows()
    knob_sets = [tuple(sorted(k.items())) for k, _ in rows]
    assert (("BM", 64), ("BR", 1)) in knob_sets
    assert (("BM", 64), ("BR", 2)) not in knob_sets


def test_node_knobs_accumulates_along_path():
    tree = SearchTree()
    b1 = _node({"BR": 2}, tree.root)
    b2 = _node({"BM": 64}, b1)
    leaf = _node({"BN": 32}, b2)
    assert TuningSearch._node_knobs(leaf) == {"BR": 2, "BM": 64, "BN": 32}


def test_pure_ucb_when_no_model():
    tree = SearchTree()
    child = _node({"BM": 64}, tree.root)
    s = TuningSearch(tree=tree)  # prior_model=None
    assert s._prior_score(child) == 0.0


# --- depth-2 PUCT acquisition ----------------------------------------------


def _fitted_prior_bm():
    """A prior that has learned bigger BM = lower latency = higher reward."""
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (2, 4, 8, 16, 32, 64)]
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    return p


def test_puct_deprioritizes_bad_unvisited_vs_forced_breadth():
    """The whole point of depth-2: a confidently-bad *unvisited* sibling must
    NOT be force-selected over a good *visited* one — unlike depth-1, where the
    unvisited child's UCB1 = +∞ forces it."""
    prior = _fitted_prior_bm()
    tree = SearchTree()
    good_visited = _node({"BM": 64}, tree.root)
    bad_unvisited = _node({"BM": 2}, tree.root)
    tree.root.children = [good_visited, bad_unvisited]
    tree.record_terminal(good_visited, 1.0)  # global_best = 1.0, Q(good) = 1
    s = TuningSearch(tree=tree, prior_model=prior)
    prior.resample()

    # Depth-1 (forced breadth): the unvisited child wins on +∞.
    assert s._ucb_key(bad_unvisited, tree.root)[0] == float("inf")
    assert max([good_visited, bad_unvisited], key=lambda c: s._ucb_key(c, tree.root)) is bad_unvisited
    # Depth-2 PUCT: the bad unvisited child is deprioritized; we keep exploiting.
    assert s._select_puct([good_visited, bad_unvisited], tree.root) is good_visited


def test_puct_still_explores_promising_unvisited():
    """A *promising* unvisited sibling (high prior) should still be explored
    over a mediocre visited one — depth-2 deprioritizes only the bad ones."""
    prior = _fitted_prior_bm()
    tree = SearchTree()
    mediocre_visited = _node({"BM": 8}, tree.root)
    good_unvisited = _node({"BM": 64}, tree.root)
    tree.root.children = [mediocre_visited, good_unvisited]
    tree.record_terminal(mediocre_visited, 0.3)  # global_best 0.3, but BM=8 is so-so
    s = TuningSearch(tree=tree, prior_model=prior)
    prior.resample()
    assert s._select_puct([mediocre_visited, good_unvisited], tree.root) is good_unvisited


def test_acquisition_falls_back_to_ucb_during_warmup():
    """Before the prior is fit, ``_select`` is plain UCB1 + forced breadth so
    the model gets seed data."""
    tree = SearchTree()
    a = _node({"BM": 64}, tree.root)
    b = _node({"BM": 2}, tree.root)
    tree.root.children = [a, b]
    prior = BayesianRidgePrior(seed=0, acquisition=True)  # acquisition on, but not fitted
    s = TuningSearch(tree=tree, prior_model=prior)
    # Acquisition is requested but the prior isn't fit → _select falls back to
    # depth-1 UCB1: both unvisited → +∞ → first wins (emission order).
    assert s._select([a, b], tree.root) is a


def test_softmax_sums_to_one_and_orders():
    p = _softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]
    # Shift-invariance (a constant offset cancels).
    p2 = _softmax([101.0, 102.0, 103.0])
    assert all(abs(a - b) < 1e-12 for a, b in zip(p, p2, strict=True))


# --- spearman helper -------------------------------------------------------


def test_spearman_monotone_and_constant():
    assert _spearman(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) > 0.999
    assert _spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) < -0.999
    assert _spearman(np.array([1.0, 1, 1, 1]), np.array([1.0, 2, 3, 4])) == 0.0
