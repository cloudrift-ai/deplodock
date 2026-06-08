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

# Tile-identity knobs a blocklist entry keys on — the planner/enumeration choices
# that fully determine a tile (so two leaves are "the same tile" iff these match).
# Excludes the post-lowering staging knobs (RING / STAGE), which are stamped after
# the greedy fork pick and differ between the leaf and the rejected node.
_TILE_IDENTITY = ("BN", "BM", "FM", "FN", "BK", "FK", "SPLITK", "BR", "WM", "WN", "MMA")


def tile_identity(knobs: dict) -> frozenset:
    """The blocklist key for a tile — its planner-chosen knobs as a hashable set.
    Computed identically for a greedy leaf's fork knobs and for a rejected node's
    realized knobs, so :class:`GreedySearch` can skip a leaf that already failed
    ``validate(ctx)`` downstream (the smem / thread-budget gate)."""
    return frozenset((k, str(knobs[k])) for k in _TILE_IDENTITY if k in knobs)


def _tile_blocked(fork_knobs: dict, blocked: set[frozenset]) -> bool:
    """True if a leaf's complete knob row matches a blocklisted tile. Only a leaf
    fork carries every identity knob, so a partial (branch) fork — whose identity
    is a strict subset — never equals a full-row entry and is never skipped."""
    return tile_identity(fork_knobs) in blocked


class GreedySearch(Search):
    """Keep one pending candidate; pick it by the prior's ``mean_score`` argmin —
    the learned ``CatBoostPrior`` once trained, the ``AnalyticPrior`` cold-start
    heuristic otherwise (both behind ``load_prior``'s ``FallbackPrior``). Falls
    to emission order (option-0) only if the prior fails to load entirely.

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists tiles that failed
    ``validate(ctx)`` on a previous compile attempt — ``Pipeline.run`` retries the
    deterministic compile with the failed leaf blocklisted so greedy falls back to
    the next prior-ranked sibling (the analogue of how ``tune`` benches-and-skips
    an unviable tile; greedy benches nothing, so the validity signal must come from
    the retry)."""

    def __init__(self, blocked: dict[str, set[frozenset]] | None = None) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None
        self._prior = None  # lazily loaded on first push (the regime's checkpoint)
        self._prior_loaded = False
        self._blocked = blocked or {}
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
        if prior is None:
            self._pending = cands[0]  # prior failed to load → emission order
            return
        c0 = cands[0]
        if c0.inner is not self._cur_inner:  # new fork point → reset the path
            self._cur_inner = c0.inner
            self._path_knobs = {}
        base = self._base_knobs(c0)
        # Tiles this node already failed to lower on an earlier attempt — skip the
        # matching leaf so greedy falls back to the next prior-ranked sibling. A
        # non-leaf fork's partial knobs never match a full-row entry, so only the
        # offending leaf is skipped, never a whole branch.
        blocked = self._blocked.get(self._node_id(c0)) if self._blocked else None
        best, best_v = None, float("inf")
        for c in cands:
            fork_knobs = c.fork.knobs if c.fork is not None else {}
            if blocked is not None and _tile_blocked(fork_knobs, blocked):
                continue
            full = {**base, **self._path_knobs, **fork_knobs}
            v = prior.mean_score(full)  # predicted latency µs — lower is better
            if v < best_v:
                best_v, best = v, c
        if best is None:  # every sibling blocklisted → no valid alternative left
            best = cands[0]
        self._pending = best
        if best.fork is not None:
            self._path_knobs.update(best.fork.knobs)

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None

    def _ensure_prior(self, c: LazyCandidate):
        """Load the one global prior once: the learned ``CatBoostPrior`` (warm if a
        checkpoint exists) behind an ``AnalyticPrior`` cold-start fallback
        (``load_prior``), so a fresh compile still ranks by the analytic heuristic
        rather than raw emission order. Best-effort: any load failure → no prior →
        emission order."""
        if self._prior_loaded:
            return self._prior
        self._prior_loaded = True
        try:
            from deplodock.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

            self._prior = load_prior()
        except Exception:  # noqa: BLE001 — a bad/missing prior must never break compile
            self._prior = None
        return self._prior

    @staticmethod
    def _node_id(c: LazyCandidate) -> str | None:
        """The graph node this fork tree is lowering — the blocklist key. ``None``
        for the root candidate (no pending match)."""
        return c.pending[0].root_node_id if c.pending else None

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
