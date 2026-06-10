"""Single-shot compile driver — picks each fork point's globally-best **complete**
leaf via the global learned prior when one is trained, else option-0.

This is the deterministic ``Pipeline.run`` driver for ``compile`` / ``run`` and
the assembled-graph lowering, with a single pending slot and no MCTS tree
(routing a whole-model compile through ``TuningSearch`` would be O(N²) and hang).
It is NOT an exploration policy: it benches nothing, so it can only *use* a prior
trained earlier by ``tune``, never train one.

**Flatten, don't descend.** The lazy fork tree (``lowering/tile`` planner) is an
MCTS data structure — it stages knob choices across levels (``BR`` → ``BM/BN`` →
``FM/FN``) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level:
a branch carries only a *partial* tile, and ``knob.knob_features`` can't compute
the tile's area / occupancy until ``FM/FN`` are pinned — so the prior is blind at
the ``BM/BN`` choice and defaults to ``BN=16`` for every shape. Instead greedy
**flattens** each fork point to its complete leaves (cheap — ``expand`` builds
only knob dicts; materialization stays deferred to the single chosen leaf's
``resolve``) and picks the one with the lowest :meth:`Prior.mean_scores` over the
full feature vector the prior trained on: the ``H_*`` host/hardware regime + the
op's ``S_*`` structural knobs (read off the parent op) + the leaf's complete knob
row. The pick equals scoring the flat candidate set, invariant to the tree's
level order. With no trained prior the model is unfit → it falls back to the first
emitted sibling (option-0).
"""

from __future__ import annotations

from collections.abc import Sequence

from deplodock.compiler.graph import Graph
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


def _is_structural(c: LazyCandidate) -> bool:
    """True for a leaf wrapping a ``Graph`` rewrite option — a kernel-set-changing
    (structural) choice, per the Op-rebind / Graph-splice classification of
    ``plans/structural-forks-in-two-level.md`` step 1."""
    fork = c.fork
    return fork is not None and getattr(fork, "is_leaf", False) and isinstance(getattr(fork, "option", None), Graph)


def _leaves(cands: Sequence[LazyCandidate]) -> list[LazyCandidate]:
    """Expand every offered candidate down to its leaf candidates, **depth-first
    in emission order** — each candidate's leaves precede the next's, so a tie in
    the prior's scores still falls to enumeration order (option-0 first), the
    no-information fallback the old per-level descent kept. Branch forks expand via
    ``LazyCandidate.expand`` — cheap, building only the next level's knob-dict
    forks; leaf candidates (``is_expandable`` False) terminate. Nothing is
    materialized: ``resolve`` (the one expensive build) runs only on the single
    leaf greedy ultimately pends. So the whole lazy tree collapses to its flat
    leaf set for one scoring pass."""
    out: list[LazyCandidate] = []
    for c in cands:
        if c.is_expandable():
            out.extend(_leaves(c.expand()))
        else:
            out.append(c)
    return out


class GreedySearch(Search):
    """Keep one pending candidate; pick it by the prior's ``mean_scores`` argmin
    over the fork point's flattened complete leaves — the learned ``CatBoostPrior``
    once trained, the ``AnalyticPrior`` cold-start heuristic otherwise (both behind
    ``load_prior``'s ``FallbackPrior``). Falls to emission order (option-0) only if
    the prior fails to load entirely.

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists tiles that failed
    ``validate(ctx)`` on a previous compile attempt — ``Pipeline.run`` retries the
    deterministic compile with the failed leaf blocklisted so greedy picks the next
    best non-blocked leaf (the analogue of how ``tune`` benches-and-skips an
    unviable tile; greedy benches nothing, so the validity signal must come from
    the retry)."""

    def __init__(self, blocked: dict[str, set[frozenset]] | None = None) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None
        self._prior = None  # lazily loaded on first push (the regime's checkpoint)
        self._prior_loaded = False
        self._blocked = blocked or {}

    def push(self, *cands: LazyCandidate, parent: object | None = None, structural: bool = False) -> None:
        del parent, structural  # greedy keeps no lineage — one pending slot, no tree; structural leaves are filtered below
        if not cands:
            self._pending = None
            return
        prior = self._ensure_prior(cands[0])
        if prior is None:
            self._pending = cands[0]  # prior failed to load → emission order
            return
        # Flatten: greedy benches nothing, so it must pick the globally best
        # COMPLETE tile, not a partial branch. The lazy fork tree exists for MCTS
        # (one node per pop); descending it level-by-level here would score the
        # ``BM/BN`` branch before ``FM/FN`` exist, so ``knob_features`` can't yet
        # compute the tile's area / occupancy and the prior is blind at the BN
        # choice (it defaults to ``BN=16`` for every shape — see the fork-tree
        # ``Level`` order in ``lowering/tile/010_partition_loops``). So instead
        # expand the offered fork(s) fully to their leaves — cheap, since
        # ``expand`` only builds knob dicts (materialization stays deferred to
        # ``resolve``) — and score every complete row in one batched ``predict``,
        # resolving straight to the argmin. The pick then equals scoring the flat
        # candidate set, invariant to how the tree's levels are arranged.
        leaves = _leaves(cands)
        # Structural options (Graph splices that change the kernel set — the
        # demoted-matmul split in ``lowering/tile/005_split_demoted``, 017's
        # atomic-free combine) are never greedy-picked while an in-place Op
        # variant exists: the per-op prior prices ONE kernel's knob row, so its
        # score for a multi-kernel Graph option is meaningless and the "pick"
        # would be prior noise. ``tune`` explores them (MCTS walks every
        # sibling); an env pin makes the Graph the rule's only option, which
        # passes through here untouched.
        op_leaves = [c for c in leaves if not _is_structural(c)]
        if op_leaves:
            leaves = op_leaves
        if len(leaves) <= 1:
            self._pending = leaves[0] if leaves else cands[0]
            return
        base = self._base_knobs(leaves[0])
        # Tiles this node already failed to lower on an earlier attempt — skip the
        # matching leaf so greedy falls back to the next prior-ranked candidate.
        blocked = self._blocked.get(self._node_id(leaves[0])) if self._blocked else None
        live = [(c, c.fork.knobs if c.fork is not None else {}) for c in leaves]
        if blocked is not None:
            live = [(c, k) for c, k in live if not _tile_blocked(k, blocked)]
        if not live:  # every leaf blocklisted → no valid alternative left
            self._pending = leaves[0]
            return
        scores = prior.mean_scores([{**base, **k} for _, k in live])  # predicted µs — lower is better
        best_i = min(range(len(live)), key=scores.__getitem__)
        self._pending = live[best_i][0]

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
