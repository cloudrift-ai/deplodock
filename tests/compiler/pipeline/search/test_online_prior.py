"""Unit tests for the learned prior (``search/prior/``) — CatBoost model, bounded
reservoir dataset + batched refit — and its value-of-position label extraction +
PUCT selection in the MCTS tree (``policy/mcts.py``).

GPU-less: they exercise the model fit / prediction, the dataset bookkeeping, and
the tree-snapshot row collection on hand-built trees — no benching, no CUDA. The
CatBoost fits use a small ``iterations`` to stay fast.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree, TuningSearch
from deplodock.compiler.pipeline.search.prior import CatBoostPrior, prior_from_json


def _node(knobs: dict, parent: SearchNode) -> SearchNode:
    """A SearchNode whose candidate exposes ``fork.knobs`` — the only thing
    ``_node_knobs`` reads."""
    return SearchNode(candidate=SimpleNamespace(fork=SimpleNamespace(knobs=knobs)), parent=parent)


def _fit(rows, **kw):
    """A CatBoostPrior fit on ``rows`` (fast: few iterations)."""
    p = CatBoostPrior(seed=0, iterations=40, min_rows=3, **kw)
    p.add_rows(rows)
    p.fit()
    return p


def _bm_rows(reps=6):
    """Synthetic: bigger BM = lower latency (label = µs), repeated for signal."""
    return [({"BM": bm}, 100.0 / bm) for bm in (2, 4, 8, 16, 32, 64) for _ in range(reps)]


# --- model fit -------------------------------------------------------------


def test_ranks_lower_latency_lower():
    """label = median latency µs; the prior must predict the faster (lower-us,
    bigger-BM) config as lower latency."""
    p = _fit(_bm_rows())
    scores = [p.mean_score({"BM": bm}) for bm in (2, 16, 64)]
    assert scores[0] > scores[1] > scores[2], f"expected decreasing in BM, got {scores}"


def test_partial_knob_dicts_fit_and_score():
    """Rows with different knob key sets (a branch pins only BR; leaves add BM)
    fit through one feature space — a partial row scores without error."""
    rows = [
        ({"BR": 1}, 0.5),
        ({"BR": 1, "BM": 32}, 0.4),
        ({"BR": 1, "BM": 64}, 0.5),
        ({"BR": 2, "BM": 64}, 0.3),
    ]
    p = _fit(rows)
    assert math.isfinite(p.mean_score({"BR": 1}))


def test_score_is_zero_before_fit():
    p = CatBoostPrior(seed=0)
    assert p.score({"BM": 64}) == 0.0
    assert p.mean_score({"BM": 64}) == 0.0
    assert not p.fitted


def test_score_equals_mean_and_is_deterministic():
    """The CatBoost prior is deterministic — ``score`` (PUCT) and ``mean_score``
    (greedy) coincide and repeat exactly (no Thompson draw)."""
    p = _fit(_bm_rows())
    assert p.score({"BM": 32}) == p.mean_score({"BM": 32}) == p.mean_score({"BM": 32})


# --- bounded reservoir dataset + batched refit -----------------------------


def test_add_rows_reservoir_caps_dataset():
    """``add_rows`` keeps the dataset at ``max_rows`` via reservoir sampling while
    counting every row ever seen."""
    p = CatBoostPrior(seed=0, max_rows=20)
    p.add_rows([({"BM": i}, float(i)) for i in range(36)])
    assert len(p._dataset) == 20  # capped
    assert p._seen == 36  # but all counted
    assert p._since_fit == 36


def test_maybe_refit_fires_on_cadence_and_resets():
    """``maybe_refit`` fits only once ``refit_every`` rows have accumulated and
    the dataset clears ``min_rows``, then resets the counter."""
    p = CatBoostPrior(seed=0, iterations=20, min_rows=5, refit_every=20)
    p.add_rows(_bm_rows(reps=2))  # 12 rows < refit_every
    assert not p.maybe_refit() and not p.fitted
    p.add_rows(_bm_rows(reps=2))  # now 24 >= 20
    assert p.maybe_refit() and p.fitted
    assert p._since_fit == 0
    assert not p.maybe_refit()  # counter reset → no immediate re-fit


def test_maybe_refit_gated_by_min_rows():
    p = CatBoostPrior(seed=0, iterations=20, min_rows=100, refit_every=5)
    p.add_rows(_bm_rows(reps=2))  # 12 rows: clears refit_every but not min_rows
    assert not p.maybe_refit() and not p.fitted


def test_force_refit_fits_small_dataset_below_tier():
    """A small tune whose rows never cross the first tier (100) still gets a model
    via ``force=True`` (end-of-run), as long as it clears ``min_rows``; a forced
    refit with nothing new since the last fit is a no-op."""
    p = CatBoostPrior(seed=0, iterations=20, min_rows=10)
    p.add_rows(_bm_rows(reps=6))  # 36 rows: below the 100 tier, above min_rows
    assert not p.maybe_refit()  # interval not reached
    assert p.maybe_refit(force=True) and p.fitted  # forced fit lands
    assert not p.maybe_refit(force=True)  # nothing new → no redundant re-fit


def test_refit_interval_tiers_by_dataset_size():
    """Default (no ``refit_every`` override) coarsens the refit interval as the
    dataset grows: every 100 up to 1k, every 1k up to 10k, every 10k beyond."""
    p = CatBoostPrior(seed=0)  # tiered schedule
    for n, expect in [(0, 100), (500, 100), (999, 100), (1000, 1000), (5000, 1000), (9999, 1000), (10_000, 10_000), (100_000, 10_000)]:
        p._dataset = [None] * n  # only len() matters for the interval
        assert p._refit_interval() == expect, f"size {n} → {p._refit_interval()}, want {expect}"


# --- persistence -----------------------------------------------------------


def test_prior_persistence_round_trips():
    """``to_json`` → ``from_json`` reconstructs a prior whose ``mean_score``
    matches; an empty prior serializes to ``None``."""
    assert CatBoostPrior(seed=0).to_json() is None
    p = _fit(_bm_rows())
    obj = p.to_json()
    assert obj is not None and obj["model"]
    q = prior_from_json(obj)
    assert q.fitted and len(q._dataset) == len(p._dataset)
    for bm in (4, 16, 64):
        assert abs(p.mean_score({"BM": bm}) - q.mean_score({"BM": bm})) < 1e-6


def test_prior_file_checkpoint_round_trips(tmp_path):
    """A fitted prior bound to a path checkpoints to JSON and reloads warm with
    matching predictions; loading a missing file yields a fresh unfit prior."""
    path = tmp_path / "prior.json"
    assert not CatBoostPrior.load(path=path).fitted  # missing → fresh
    p = _fit(_bm_rows())
    p._path = path
    p.checkpoint()
    assert path.exists()
    q = CatBoostPrior.load(path=path)
    assert q.fitted and q._first_fit_idx == 0 and len(q._dataset) == len(p._dataset)
    for bm in (4, 16, 64):
        assert abs(p.mean_score({"BM": bm}) - q.mean_score({"BM": bm})) < 1e-6


def test_load_tolerates_stale_checkpoint(tmp_path):
    """A pre-CatBoost checkpoint (sklearn estimator-state ``model`` dict +
    legacy ``archived_rows`` key) migrates instead of crashing: the unusable
    model is dropped, the rows are salvaged, and the next refit rebuilds it."""
    from deplodock import storage

    path = tmp_path / "prior.json"
    stale = {
        "cols": ["BM"],
        "model": {"class": "BayesianRidge", "state": {}},  # sklearn estimator state, not a cbm blob
        "archived_rows": [[{"BM": bm}, math.log(bm)] for bm in (2, 4, 8, 16)],
    }
    storage.write_json(path, stale)
    p = CatBoostPrior.load(path=path)
    assert not p.fitted  # stale model discarded
    assert len(p._dataset) == 4  # legacy rows salvaged
    p.add_rows(_bm_rows())
    p.fit()
    assert p.fitted  # rebuilt fine from salvaged + new rows


def test_diagnostics_report_reachability():
    """``diagnostics.report`` groups by op and reports argmax reachability — on a
    perfectly-rankable synthetic op the prior recovers the best (ratio 1.0)."""
    from deplodock.compiler.pipeline.search.data import Dataset
    from deplodock.compiler.pipeline.search.prior import diagnostics

    # one op-structure (S_kind), bigger BM = lower latency (label = µs)
    rows = [({"S_kind": 1.0, "BM": bm}, 100.0 / bm) for bm in (2, 4, 8, 16, 32, 64) for _ in range(6)]
    p = _fit(rows)
    groups = Dataset.from_prior(p).group_by_op()
    assert len(groups) == 1
    rr = diagnostics.reachability(p, groups)
    assert len(rr) == 1
    _, best_us, pick_us, ratio, _ = rr[0]
    assert ratio == min(r[3] for r in rr) and ratio < 1.01  # recovers the best leaf
    text = diagnostics.report(p)
    assert "reachability" in text and "golden coverage" in text


def test_dataset_accumulates_across_load():
    """A reloaded prior keeps its dataset + reservoir counters so a follow-up tune
    keeps accumulating from where it left off."""
    p = _fit(_bm_rows())
    seen0, n0 = p._seen, len(p._dataset)
    q = prior_from_json(p.to_json())
    assert q._seen == seen0 and len(q._dataset) == n0
    q.add_rows([({"BM": 8}, 12.5)])
    assert q._seen == seen0 + 1


# --- value-of-position labels from the tree --------------------------------


def test_collect_rows_labels_branch_with_best_descendant():
    """A branch shared by two benched leaves is labeled with the BETTER leaf's
    latency (``1/best_reward``, the min over its subtree), not its own."""
    tree = SearchTree()
    branch = _node({"BR": 1}, tree.root)
    leaf_slow = _node({"BR": 1, "BM": 32}, branch)
    leaf_fast = _node({"BR": 1, "BM": 64}, branch)
    tree.root.children = [branch]
    branch.children = [leaf_slow, leaf_fast]
    tree.record_terminal(leaf_slow, 1.0)
    tree.record_terminal(leaf_fast, 2.0)
    assert branch.best_reward == 2.0
    rows = TuningSearch(tree=tree)._collect_rows()
    by_knobs = {tuple(sorted(k.items())): lab for k, lab in rows}
    assert abs(by_knobs[(("BR", 1),)] - 0.5) < 1e-12  # 1/best_reward = 1/2.0
    assert abs(by_knobs[(("BM", 64), ("BR", 1))] - 0.5) < 1e-12
    assert abs(by_knobs[(("BM", 32), ("BR", 1))] - 1.0) < 1e-12


def test_collect_rows_skips_unbenched_nodes():
    tree = SearchTree()
    visited = _node({"BR": 1, "BM": 64}, tree.root)
    frontier = _node({"BR": 2, "BM": 64}, tree.root)
    tree.root.children = [visited, frontier]
    tree.record_terminal(visited, 1.5)
    knob_sets = [tuple(sorted(k.items())) for k, _ in TuningSearch(tree=tree)._collect_rows()]
    assert (("BM", 64), ("BR", 1)) in knob_sets
    assert (("BM", 64), ("BR", 2)) not in knob_sets


# --- node store extraction (_collect_node_records) -------------------------


def test_collect_node_records_parent_linkage():
    """A leaf points at its branch via parent_key (the real tree edge), and the
    leaf's full ``realized_knobs`` — incl. a deterministically-stamped knob the
    branch never saw — is what gets stored, proving parentage is taken from the
    ``parent`` pointer, not inferred from knob-subset containment."""
    tree = SearchTree()
    branch = _node({"BR": 1}, tree.root)
    leaf = _node({"BR": 1, "BM": 64}, branch)
    leaf.realized_knobs = {"S_n_mma": 1.0, "H_opt": 1.0, "BR": 1, "BM": 64, "FK": 2}
    tree.root.children = [branch]
    branch.children = [leaf]
    tree.record_terminal(leaf, 2.0)
    recs = TuningSearch(tree=tree)._collect_node_records(context_key="ctx", op_sig="sig")
    by_depth = {d: (nk, pk, feats) for nk, pk, feats, _, d in recs}
    branch_key, branch_parent, branch_feats = by_depth[1]
    leaf_key, leaf_parent, leaf_feats = by_depth[2]
    assert branch_parent is None  # top fork → no recorded parent
    assert leaf_parent == branch_key  # leaf → branch edge from node.parent
    assert "FK" in leaf_feats and "FK" not in branch_feats  # stamped knob only on the leaf


def test_collect_node_records_value_min_invariant():
    """Each node's value-of-position (1/best_reward) is >= its parent's — the
    max-propagation invariant the walk asserts."""
    tree = SearchTree()
    branch = _node({"BR": 1}, tree.root)
    leaf_slow = _node({"BR": 1, "BM": 32}, branch)
    leaf_fast = _node({"BR": 1, "BM": 64}, branch)
    tree.root.children = [branch]
    branch.children = [leaf_slow, leaf_fast]
    tree.record_terminal(leaf_slow, 1.0)
    tree.record_terminal(leaf_fast, 2.0)
    recs = TuningSearch(tree=tree)._collect_node_records(context_key="ctx", op_sig="sig")
    by_key = {nk: (pk, v) for nk, pk, _, v, _ in recs}
    assert any(pk is not None for pk, _ in by_key.values())  # at least one real edge exercised
    for pk, v in by_key.values():
        if pk is not None:
            assert v >= by_key[pk][1] - 1e-12  # child never labeled faster than its parent


def test_collect_node_records_skips_unbenched_and_sentinel():
    """An unbenched frontier yields no row, and the sentinel root is never emitted
    (its benched child is depth 1 with ``parent_key`` None)."""
    tree = SearchTree()
    visited = _node({"BR": 1, "BM": 64}, tree.root)
    frontier = _node({"BR": 2, "BM": 64}, tree.root)
    tree.root.children = [visited, frontier]
    tree.record_terminal(visited, 1.5)
    recs = TuningSearch(tree=tree)._collect_node_records(context_key="ctx", op_sig="sig")
    assert len(recs) == 1
    nk, pk, _, v, d = recs[0]
    assert pk is None and d == 1
    assert abs(v - 1 / 1.5) < 1e-12


def test_node_key_excludes_s_and_h():
    """Within one op (op_sig) + regime (context_key), node identity is the tunable
    knob set only — two nodes differing solely in ``S_*``/``H_*`` collide, and a
    differing tunable knob splits them."""
    s = TuningSearch(tree=SearchTree())
    base = s._node_key({"BM": 64, "S_n_mma": 1.0, "H_opt": 1.0}, "sig", "ctx")
    s_h_perturbed = s._node_key({"BM": 64, "S_n_mma": 9.0, "H_opt": 3.0}, "sig", "ctx")
    tunable_perturbed = s._node_key({"BM": 32, "S_n_mma": 1.0, "H_opt": 1.0}, "sig", "ctx")
    assert base == s_h_perturbed
    assert base != tunable_perturbed


def _stats(median: float):
    from deplodock.compiler.pipeline.search.db import PerfStats  # noqa: PLC0415

    return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)


def test_o3_worthy_within_tolerance_band_and_dedup(monkeypatch):
    """``observe`` flags ``last_o3_worthy`` for the new best AND any later config
    within ``DEPLODOCK_O3_TOL`` of the best -O1 — not just strict improvements —
    and dedups so each config is -O3'd at most once."""
    monkeypatch.setenv("DEPLODOCK_O3_TOL", "0.10")
    tree = SearchTree()
    best = _node({"WM": 1, "WN": 4}, tree.root)  # the fast one
    near = _node({"WM": 2, "WN": 2}, tree.root)  # within 10%
    far = _node({"WM": 8, "WN": 1}, tree.root)  # outside 10%
    tree.root.children = [best, near, far]
    s = TuningSearch(tree=tree)

    s.observe(best, _stats(100.0), "ok")
    assert s.last_o3_worthy is True and s.last_improved_best is True  # new best
    s.observe(near, _stats(108.0), "ok")
    assert s.last_o3_worthy is True and s.last_improved_best is False  # within 10%, not a new best
    s.observe(far, _stats(130.0), "ok")
    assert s.last_o3_worthy is False  # 30% > tol
    # Re-benching the same config again does not re-flag (dedup).
    s.observe(near, _stats(108.0), "ok")
    assert s.last_o3_worthy is False


def test_node_knobs_accumulates_along_path():
    tree = SearchTree()
    b1 = _node({"BR": 2}, tree.root)
    b2 = _node({"BM": 64}, b1)
    leaf = _node({"BN": 32}, b2)
    assert TuningSearch(tree=tree)._node_knobs(leaf) == {"BR": 2, "BM": 64, "BN": 32}


def test_node_knobs_includes_base_structural_knobs():
    """base_knobs (the op's S_* identity) is merged under the fork deltas, so the
    global prior's features carry op-structure."""
    tree = SearchTree()
    leaf = _node({"BR": 2}, tree.root)
    s = TuningSearch(tree=tree, base_knobs={"S_n_loop": 3.0, "S_ext_reduce_max": 64.0})
    assert s._node_knobs(leaf) == {"S_n_loop": 3.0, "S_ext_reduce_max": 64.0, "BR": 2}


def test_prior_score_zero_when_no_model():
    tree = SearchTree()
    child = _node({"BM": 64}, tree.root)
    assert TuningSearch(tree=tree)._prior_score(child) == 0.0


# --- PUCT selection (the only rule) ----------------------------------------


def test_puct_deprioritizes_bad_unvisited():
    """A confidently-bad *unvisited* sibling must NOT be selected over a good
    *visited* one — there is no ``+∞``-unvisited rule to force it."""
    prior = _fit(_bm_rows())
    tree = SearchTree()
    good_visited = _node({"BM": 64}, tree.root)
    bad_unvisited = _node({"BM": 2}, tree.root)
    tree.root.children = [good_visited, bad_unvisited]
    tree.record_terminal(good_visited, 1.0)  # Q(good) = 1
    s = TuningSearch(tree=tree, prior_model=prior)
    assert s._select([good_visited, bad_unvisited], tree.root) is good_visited


def test_puct_still_explores_promising_unvisited():
    """A *promising* unvisited sibling (high prior) should still be explored over
    a mediocre visited one."""
    prior = _fit(_bm_rows())
    tree = SearchTree()
    mediocre_visited = _node({"BM": 8}, tree.root)
    good_unvisited = _node({"BM": 64}, tree.root)
    tree.root.children = [mediocre_visited, good_unvisited]
    tree.record_terminal(mediocre_visited, 0.3)
    s = TuningSearch(tree=tree, prior_model=prior)
    assert s._select([mediocre_visited, good_unvisited], tree.root) is good_unvisited


def test_cold_or_absent_prior_descends_emission_order():
    """With no prior or a cold (unfit) prior, ``P`` is uniform (= 1) → every
    child's PUCT value ties → it picks the first child (emission order)."""
    tree = SearchTree()
    a = _node({"BM": 64}, tree.root)
    b = _node({"BM": 2}, tree.root)
    tree.root.children = [a, b]
    assert TuningSearch(tree=tree)._select([a, b], tree.root) is a
    cold = CatBoostPrior(seed=0)
    assert TuningSearch(tree=tree, prior_model=cold)._select([a, b], tree.root) is a


# --- deployable-evidence pick (Prior.pick / evidence_pick) ------------------


def _o3_row(knobs: dict, us: float) -> tuple[dict, float]:
    """A reservoir row shaped like tune's -O3 re-bench: S_* identity + H_opt=3
    stamp + tunable knobs."""
    return ({"S_sig": 7.0, "H_opt": 3.0, **knobs}, us)


def _cand(knobs: dict) -> dict:
    """A greedy candidate row: same S_* identity, deploy regime H_opt=3."""
    return {"S_sig": 7.0, "H_opt": 3.0, **knobs}


def test_evidence_pick_prefers_measured_best():
    """A candidate matching the fastest -O3 reservoir row wins the pick even when
    the model would rank an unmeasured candidate lower (the golden-sweep
    gate_up_proj.s128 class: measured rank-1 config lost the deploy to an
    extrapolation)."""
    p = CatBoostPrior(seed=0, iterations=40, min_rows=3)
    p.add_rows([_o3_row({"FM": 6, "FN": 4}, 24.1), _o3_row({"FM": 4, "FN": 4}, 30.0)])
    p.add_rows(_bm_rows())  # unrelated S_-less rows give the model signal
    p.fit()
    rows = [_cand({"FM": 8, "FN": 4}), _cand({"FM": 6, "FN": 4}), _cand({"FM": 4, "FN": 4})]
    best_i, us = p.pick(rows)
    assert best_i == 1, "pick must land on the measured-fastest config, not an unmeasured one"
    assert us == 24.1


def test_evidence_prefix_consistency():
    """A partial candidate (knob undecided) inherits the best measured outcome
    consistent with its prefix — value-of-position semantics; a candidate that
    contradicts every measured row gets no evidence."""
    p = CatBoostPrior(seed=0)
    p.add_rows([_o3_row({"FM": 6, "RING": 2}, 24.0), _o3_row({"FM": 6, "RING": 3}, 28.0)])
    ev = p.evidence_pick([_cand({"FM": 6}), _cand({"FM": 8})])
    assert ev == (0, 24.0)  # FM=6 prefix → min over its two measured rows
    assert p.evidence_pick([_cand({"FM": 8})]) is None


def test_evidence_ignores_o1_rows_and_other_ops():
    """-O1 ranking rows (H_opt=1) and rows from a different S_* signature are not
    evidence; without -O3 rows the pick falls back to the model argmin."""
    p = CatBoostPrior(seed=0, iterations=40, min_rows=3)
    p.add_rows([({"S_sig": 7.0, "H_opt": 1.0, "FM": 6}, 5.0), ({"S_sig": 9.0, "H_opt": 3.0, "FM": 6}, 5.0)])
    assert p.evidence_pick([_cand({"FM": 6})]) is None
    p.add_rows(_bm_rows())
    p.fit()
    best_i, us = p.pick([{"H_opt": 3.0, "BM": 2}, {"H_opt": 3.0, "BM": 64}])
    assert best_i == 1  # model argmin (bigger BM = faster in _bm_rows)


def test_evidence_skipped_off_o3_regime():
    """Deploying a non--O3 regime (e.g. ``--nvcc-flags -Xcicc -O1``) must not
    consult -O3 evidence."""
    p = CatBoostPrior(seed=0)
    p.add_rows([_o3_row({"FM": 6}, 24.0)])
    assert p.evidence_pick([{"S_sig": 7.0, "H_opt": 1.0, "FM": 6}]) is None


def test_evidence_matches_masked_structural_feature():
    """``S_masked_*`` (the per-role structural feature that replaced the OVERHANG
    knob) joins the ``S_*`` evidence signature: a masked candidate matches a
    masked -O3 row, and the scalar float hashes cleanly into the frozenset
    signature (the whole point of moving it off the sequence-valued knob)."""
    p = CatBoostPrior(seed=0)
    p.add_rows([_o3_row({"S_masked_m": 1.0, "FM": 6}, 24.0)])
    ev = p.evidence_pick([_cand({"S_masked_m": 1.0, "FM": 6})])
    assert ev == (0, 24.0)


def test_fallback_pick_uses_learned_evidence_when_cold():
    """FallbackPrior.pick consults the learned half's reservoir even before the
    model is fitted (a freshly-seeded reservoir holds real measurements), and
    falls through to the analytic ranking when no evidence matches."""
    from deplodock.compiler.pipeline.search.prior import FallbackPrior

    learned = CatBoostPrior(seed=0)
    learned.add_rows([_o3_row({"FM": 6}, 24.0)])
    fb = FallbackPrior(learned)
    assert not fb.fitted
    best_i, us = fb.pick([_cand({"FM": 8}), _cand({"FM": 6})])
    assert (best_i, us) == (1, 24.0)
    # No evidence → analytic mean_scores argmin path (finite, in-range index).
    best_i, score = fb.pick([_cand({"FM": 8}), _cand({"FM": 12})])
    assert best_i in (0, 1) and math.isfinite(score)
