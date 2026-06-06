"""Single-shot compile driver — picks each fork via the global learned prior
when one is trained, else option-0.

This is the deterministic ``Pipeline.run`` driver for ``compile`` / ``run`` and
the assembled-graph lowering, with **O(1) work per step** (a single pending
slot, no MCTS tree — routing a whole-model compile through ``TuningSearch``
would be O(N²) and hang). It is NOT an exploration policy: it benches nothing,
so it can only *use* a prior trained earlier by ``tune``, never train one.

When a trained global prior exists (``CatBoostPrior.load`` finds a
checkpoint), it picks the sibling with the lowest :meth:`Prior.mean_score` (the
prior predicts latency µs — lower is better) over the same feature vector the
prior was trained on: the ``H_*`` host/hardware regime + the op's ``S_*``
structural knobs (read off the parent op) + the deltas chosen so far down this
fork tree + the candidate's own delta. With no trained prior the model is unfit →
it falls back to the first emitted sibling (option-0).
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.policy.base import Search


class GreedySearch(Search):
    """Keep one pending candidate; pick it by the learned prior (argmax) or, with
    no trained prior, by emission order (option-0)."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None
        self._prior = None  # lazily loaded on first push (the regime's checkpoint)
        self._prior_loaded = False
        # Fork-tree state: ``_cur_inner`` is the shared ``inner`` Candidate of the
        # current fork tree (changes only when a leaf resolves into a new fork
        # point); ``_path_knobs`` accumulates the deltas chosen down that tree (the
        # branch slices aren't applied to the graph until a leaf resolves).
        self._cur_inner = None
        self._path_knobs: dict = {}

    def push(self, *cands: LazyCandidate, parent: object | None = None) -> None:
        del parent  # greedy keeps no lineage — one pending slot, no tree
        if not cands:
            self._pending = None
            return
        prior = self._ensure_prior(cands[0])
        if prior is None or not prior.fitted:
            self._pending = cands[0]  # no trained prior → option-0
            return
        c0 = cands[0]
        if c0.inner is not self._cur_inner:  # new fork point → reset the path
            self._cur_inner = c0.inner
            self._path_knobs = {}
        base = self._base_knobs(c0)
        best, best_v = cands[0], float("inf")
        for c in cands:
            full = {**base, **self._path_knobs, **(c.fork.knobs if c.fork is not None else {})}
            v = prior.mean_score(full)  # predicted latency µs — lower is better
            if v < best_v:
                best_v, best = v, c
        self._pending = best
        if best.fork is not None:
            self._path_knobs.update(best.fork.knobs)

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None

    def _ensure_prior(self, c: LazyCandidate):
        """Load the one global prior once (warm if a checkpoint exists, else an
        unfit prior → option-0). Best-effort: any load failure falls back to no
        prior."""
        if self._prior_loaded:
            return self._prior
        self._prior_loaded = True
        try:
            from deplodock.compiler.pipeline.search.prior import CatBoostPrior  # noqa: PLC0415

            self._prior = CatBoostPrior.load()
        except Exception:  # noqa: BLE001 — a bad/missing prior must never break compile
            self._prior = None
        return self._prior

    @staticmethod
    def _base_knobs(c: LazyCandidate) -> dict:
        """The constant base under this fork tree's deltas — the parent op's
        knobs (its ``S_*`` structural identity) plus the ``H_*`` host/hardware
        regime, matching the feature base tune trained on
        (:meth:`two_level.inner_reward`)."""
        ctx_feats = c.inner.ctx.features()
        if c.pending is None:
            return ctx_feats
        match, _ = c.pending
        node = c.inner.graph.nodes.get(match.root_node_id)
        return {**ctx_feats, **dict(node.op.knobs)} if node is not None else ctx_feats
