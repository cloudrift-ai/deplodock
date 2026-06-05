"""Unit tests for the learned prior (``search/prior/``) and its value-of-position
label extraction + PUCT selection in the MCTS tree (``policy/mcts.py``).

These are GPU-less: they exercise the model fit / Thompson draw and the
tree-snapshot row collection on hand-built trees — no benching, no CUDA.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch, _softmax
from deplodock.compiler.pipeline.search.prior import BayesianRidgePrior


def _node(knobs: dict, parent: SearchNode) -> SearchNode:
    """A SearchNode whose candidate exposes ``fork.knobs`` — the only thing
    ``_node_knobs`` reads."""
    return SearchNode(candidate=SimpleNamespace(fork=SimpleNamespace(knobs=knobs)), parent=parent)


# --- model fit -------------------------------------------------------------


def test_fits_and_ranks_linear_target():
    """On an exactly-linear target the BayesianRidge fit ranks the rows in the
    true order (Spearman ~1) — exact label recovery isn't expected (it fits its
    own regularization), only correct ranking."""
    rows = []
    for bm in (16, 32, 64, 128):
        for bn in (16, 32, 64):
            rows.append(({"BM": bm, "BN": bn}, 0.01 * bm - 0.02 * bn))
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    preds = [p.mean_score(k) for k, _ in rows]
    ys = [y for _, y in rows]
    order_pred = sorted(range(len(rows)), key=lambda i: preds[i])
    order_true = sorted(range(len(rows)), key=lambda i: ys[i])
    assert order_pred == order_true


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


def test_score_is_thompson_stochastic_mean_is_not():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p = BayesianRidgePrior(seed=7, min_rows=3)
    p.fit(rows)
    # score() draws a fresh Thompson sample each call (per-point predictive std).
    assert p.score({"BM": 64}) != p.score({"BM": 64})
    # mean_score() is the deterministic posterior mean.
    assert p.mean_score({"BM": 64}) == p.mean_score({"BM": 64})


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
    assert TuningSearch(tree=tree)._node_knobs(leaf) == {"BR": 2, "BM": 64, "BN": 32}


def test_node_knobs_includes_base_structural_knobs():
    """base_knobs (the op's S_* identity) is merged under the fork deltas, so
    the global prior's features carry op-structure."""
    tree = SearchTree()
    leaf = _node({"BR": 2}, tree.root)
    s = TuningSearch(tree=tree, base_knobs={"S_n_loop": 3.0, "S_ext_reduce_max": 64.0})
    assert s._node_knobs(leaf) == {"S_n_loop": 3.0, "S_ext_reduce_max": 64.0, "BR": 2}


def test_prior_score_zero_when_no_model():
    tree = SearchTree()
    child = _node({"BM": 64}, tree.root)
    s = TuningSearch(tree=tree)  # prior_model=None
    assert s._prior_score(child) == 0.0


# --- PUCT selection (the only rule) ----------------------------------------


def _fitted_prior_bm():
    """A prior that has learned bigger BM = lower latency = higher reward."""
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (2, 4, 8, 16, 32, 64)]
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    return p


def test_puct_deprioritizes_bad_unvisited():
    """A confidently-bad *unvisited* sibling must NOT be selected over a good
    *visited* one — there is no ``+∞``-unvisited rule to force it."""
    prior = _fitted_prior_bm()
    tree = SearchTree()
    good_visited = _node({"BM": 64}, tree.root)
    bad_unvisited = _node({"BM": 2}, tree.root)
    tree.root.children = [good_visited, bad_unvisited]
    tree.record_terminal(good_visited, 1.0)  # global_best = 1.0, Q(good) = 1
    s = TuningSearch(tree=tree, prior_model=prior)
    prior.resample()
    assert s._select([good_visited, bad_unvisited], tree.root) is good_visited


def test_puct_still_explores_promising_unvisited():
    """A *promising* unvisited sibling (high prior) should still be explored
    over a mediocre visited one — PUCT deprioritizes only the bad ones."""
    prior = _fitted_prior_bm()
    tree = SearchTree()
    mediocre_visited = _node({"BM": 8}, tree.root)
    good_unvisited = _node({"BM": 64}, tree.root)
    tree.root.children = [mediocre_visited, good_unvisited]
    tree.record_terminal(mediocre_visited, 0.3)  # global_best 0.3, but BM=8 is so-so
    s = TuningSearch(tree=tree, prior_model=prior)
    prior.resample()
    assert s._select([mediocre_visited, good_unvisited], tree.root) is good_unvisited


def test_cold_or_absent_prior_descends_emission_order():
    """With no prior (single-shot compile) or a cold prior, every score is 0 →
    uniform softmax → PUCT picks the first child (emission order). No forced
    breadth, no +∞."""
    tree = SearchTree()
    a = _node({"BM": 64}, tree.root)
    b = _node({"BM": 2}, tree.root)
    tree.root.children = [a, b]
    assert TuningSearch(tree=tree)._select([a, b], tree.root) is a  # no prior
    cold = BayesianRidgePrior(seed=0)  # attached but unfit
    assert TuningSearch(tree=tree, prior_model=cold)._select([a, b], tree.root) is a


def test_prior_persistence_round_trips():
    """``to_bytes`` → ``from_bytes`` reconstructs a fitted prior whose
    ``mean_score`` matches the original; an unfit prior serializes to ``None``."""
    from deplodock.compiler.pipeline.search.prior import prior_from_bytes

    assert BayesianRidgePrior(seed=0).to_bytes() is None  # unfit → nothing to save
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (2, 4, 8, 16, 32, 64)]
    p = BayesianRidgePrior(seed=0, min_rows=3)
    p.fit(rows)
    blob = p.to_bytes()
    assert blob is not None
    q = prior_from_bytes(blob)
    assert q.fitted
    for bm in (4, 16, 64):
        assert abs(p.mean_score({"BM": bm}) - q.mean_score({"BM": bm})) < 1e-9


def test_prior_file_store_round_trips(tmp_path):
    from deplodock.compiler.pipeline.search.prior import store

    path = tmp_path / "prior.pkl"
    assert store.load(path, "regimeA") is None  # missing file
    store.save(path, "regimeA", b"blobA")
    store.save(path, "regimeB", b"blobB")  # second regime coexists
    assert store.load(path, "regimeA") == b"blobA"
    assert store.load(path, "regimeB") == b"blobB"
    store.save(path, "regimeA", b"newerA")  # upsert
    assert store.load(path, "regimeA") == b"newerA"
    assert store.load(path, "missing") is None


def test_global_prior_archive_accumulates_and_warm_starts():
    """A global prior trained on one op, archived, then reloaded is warm
    (fitted, no warmup) and its archived rows survive the round-trip."""
    from deplodock.compiler.pipeline.search.prior import prior_from_bytes

    p = BayesianRidgePrior(seed=0, min_rows=3)
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (2, 4, 8, 16, 32, 64)]
    p.fit(rows)
    p.archive(rows)  # freeze this op's rows into the global set
    q = prior_from_bytes(p.to_bytes())
    assert q.fitted
    assert q._first_fit_idx == 0  # loaded → warm, no cold warmup
    assert len(q._archived_rows) == len(rows)


def test_softmax_sums_to_one_and_orders():
    p = _softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]
    # Shift-invariance (a constant offset cancels).
    p2 = _softmax([101.0, 102.0, 103.0])
    assert all(abs(a - b) < 1e-12 for a, b in zip(p, p2, strict=True))


def test_seeded_priors_reproducible_mean_score():
    rows = [({"BM": bm}, math.log(1.0 / (100.0 / bm))) for bm in (16, 32, 64, 128)]
    p1, p2 = BayesianRidgePrior(seed=3, min_rows=3), BayesianRidgePrior(seed=3, min_rows=3)
    p1.fit(rows)
    p2.fit(rows)
    assert p1.mean_score({"BM": 32}) == p2.mean_score({"BM": 32})


# --- spearman helper -------------------------------------------------------
