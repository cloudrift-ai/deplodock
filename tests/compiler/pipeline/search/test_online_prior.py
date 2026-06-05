"""Unit tests for the online learned prior (``search/prior.py``) and its
value-of-position label extraction from the MCTS tree (``policy/mcts.py``).

These are GPU-less: they exercise the model fit / Thompson draw and the
tree-snapshot row collection on hand-built trees — no benching, no CUDA.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch
from deplodock.compiler.pipeline.search.prior import OnlinePrior, _spearman


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
    p = OnlinePrior(seed=0, ridge=1e-6, min_rows=3)
    p.fit(rows)
    preds = np.array([p.mean_score(k) for k, _ in rows])
    ys = np.array([y for _, y in rows])
    assert np.allclose(preds, ys, atol=1e-3)


def test_ranks_lower_latency_higher():
    """label = log(1/us); the prior must score the faster (lower-us) config
    higher. Bigger BM is faster in this synthetic."""
    rows = [({"BM": bm, "BN": 64}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p = OnlinePrior(seed=0, min_rows=3)
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
    p = OnlinePrior(seed=0, min_rows=3)
    p.fit(rows)
    # Scoring a never-seen partial state must not raise and must be finite.
    v = p.mean_score({"BR": 1})
    assert math.isfinite(v)


def test_score_is_zero_before_fit():
    p = OnlinePrior(seed=0)
    assert p.score({"BM": 64}) == 0.0
    assert p.mean_score({"BM": 64}) == 0.0


def test_thompson_draw_is_deterministic_until_resample():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p = OnlinePrior(seed=7, min_rows=3)
    p.fit(rows)  # fit() ends with a resample
    a = p.score({"BM": 64})
    b = p.score({"BM": 64})
    assert a == b, "score must be stable under a fixed Thompson draw"
    p.resample()
    # A fresh draw almost surely shifts the value (sanity that resample fires).
    assert p.score({"BM": 64}) != a


def test_seeded_priors_are_reproducible():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p1 = OnlinePrior(seed=3, min_rows=3)
    p2 = OnlinePrior(seed=3, min_rows=3)
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


# --- spearman helper -------------------------------------------------------


def test_spearman_monotone_and_constant():
    assert _spearman(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) > 0.999
    assert _spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) < -0.999
    assert _spearman(np.array([1.0, 1, 1, 1]), np.array([1.0, 2, 3, 4])) == 0.0
