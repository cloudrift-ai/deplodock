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

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph
from deplodock.compiler.pipeline.fork import Fork, flatten_leaves
from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.policy.base import Search

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.pipeline import ForkPoint


@lru_cache(maxsize=1)
def _tile_pipeline():
    """The ``lowering/tile``-only pipeline the structural price probes drive —
    frozen and shareable, so one load serves every nested descent."""
    from deplodock.compiler.pipeline import Pipeline  # noqa: PLC0415

    return Pipeline.build(["lowering/tile"])


# Tile-identity knobs a blocklist entry keys on — the planner/enumeration choices
# that fully determine a tile (so two leaves are "the same tile" iff these match).
# Excludes the post-lowering staging knobs (RING / STAGE), which are stamped after
# the greedy fork pick and differ between the leaf and the rejected node.
_TILE_IDENTITY = ("BN", "BM", "FM", "FN", "BK", "FK", "SPLITK", "BR", "WM", "WN", "MMA")

# The rule whose fork prices a kernel: the prior's predicted µs for the chosen
# complete tile row at the partition fork is the per-kernel cost the structural
# pricing sums (defined here, not in ``two_level``, because that module imports
# this package at module scope — the reverse would cycle).
PARTITION_RULE = "010_partition_loops"


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
    the retry).

    Structural (``Graph``-splicing) options are priced with the trained prior —
    see :meth:`_pick_structural` — so an unpinned ``compile`` / ``run`` can
    deploy the kernel sets ``tune`` measured best (the demoted-matmul split);
    cold, the structural leaf is filtered and kernel sets stay unchanged."""

    def __init__(self, blocked: dict[str, set[frozenset]] | None = None, *, prior=None, price_structural: bool = True) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None
        self._prior = prior  # injected, or lazily loaded on first push (the regime's checkpoint)
        self._prior_loaded = prior is not None
        self._blocked = blocked or {}
        # Structural-option pricing (plans/structural-forks-in-two-level.md step
        # 3): with the *trained* prior loaded, a ``Graph``-splicing option is
        # priced as the Σ of its kernels' predicted-best µs (nested greedy
        # descent per kernel) against the keep-fused side's predicted-best, and
        # the cheaper kernel set wins. ``price_structural=False`` keeps the old
        # filter behavior — used by ``Pipeline.run``'s retry after a structural
        # pick failed to lower, and by the nested pricing descents themselves
        # (no recursive splitting inside a price probe).
        self._price_structural = price_structural
        self._price_memo: dict[str, float | None] = {}  # op_cache_key → predicted µs (None = unpriceable)
        # True once this run pended a structural (kernel-set-changing) leaf —
        # ``Pipeline.run`` reads it to retire structural picks when the drive
        # leaves a node un-lowered.
        self.picked_structural = False
        # The prior's predicted µs for the chosen leaf at each partition fork,
        # by node id — the per-kernel price a nested pricing descent reads off.
        self.partition_scores: dict[str, float] = {}

    def push(self, *cands: LazyCandidate, parent: object | None = None, structural: bool = False) -> None:
        del parent, structural  # greedy keeps no lineage — one pending slot, no tree; structural leaves are priced (or filtered) below
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
        # atomic-free combine): the per-op prior prices ONE kernel's knob row,
        # so its score for a multi-kernel Graph option is meaningless. With the
        # *trained* prior loaded, :meth:`_pick_structural` prices the option
        # properly — Σ of nested per-kernel predicted-bests vs the keep-fused
        # side — and pends the split when it predicts faster. Cold (analytic /
        # no prior), or when an option can't be priced, the structural leaf is
        # filtered as before so a cold compile never changes kernel sets.
        # ``tune`` explores them regardless (MCTS walks every sibling); an env
        # pin makes the Graph the rule's only option, which passes through
        # here untouched.
        if any(_is_structural(c) for c in leaves):
            pick = self._pick_structural(leaves, prior)
            if pick is not None:
                self.picked_structural = True
                self._pending = pick
                return
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
        # Record the chosen tile's predicted µs at the kernel's partition fork —
        # the per-kernel price :meth:`_price_kernel`'s nested descent reads off.
        rule = leaves[0].pending[0].rule if leaves[0].pending is not None else None
        if rule is not None and rule.name == PARTITION_RULE:
            nid = self._node_id(leaves[0])
            if nid is not None:
                self.partition_scores[nid] = scores[best_i]

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None

    def _pick_structural(self, leaves: list[LazyCandidate], prior) -> LazyCandidate | None:
        """Price the structural (``Graph``-splicing) leaves of one fork against
        the keep-fused ``Op`` side and return the winning structural leaf, or
        ``None`` to keep the op-variant path (cold prior, unpriceable option,
        or fused predicted faster).

        Both sides are priced the same way: the prior's predicted-best µs at
        each kernel's partition fork, obtained by a nested greedy descent over
        the kernel's single-node slice (``lowering/tile`` only, no backend,
        CPU-only — see :meth:`_price_kernel`); a structural option's price is
        the Σ over its fragment's kernels. Gated on the *trained*
        ``CatBoostPrior`` (``prior.fitted``): Σ-comparisons through the
        analytic cold-start model are unvalidated, and a cold compile must
        never change kernel sets. Greedy is prior-only by design — the price
        never reads the DB (the learned-prior work removed ``_best_fork``
        replay deliberately)."""
        if not self._price_structural or prior is None or not getattr(prior, "fitted", False):
            return None
        op_leaves = [c for c in leaves if not _is_structural(c)]
        if not op_leaves:
            return None  # nothing to compare against — the no-op-variant edge keeps today's scoring path
        fused_prices = [self._price_op_leaf(c) for c in op_leaves]
        if any(p is None for p in fused_prices):
            return None
        split_prices = [(c, self._price_graph(c.fork.option, c.inner.ctx)) for c in leaves if _is_structural(c)]
        split_prices = [(c, p) for c, p in split_prices if p is not None]
        if not split_prices:
            return None
        best_split, best_split_us = min(split_prices, key=lambda cp: cp[1])
        return best_split if best_split_us < min(fused_prices) else None

    def _price_op_leaf(self, c: LazyCandidate) -> float | None:
        """The keep-fused side's price: the leaf's ``Op`` rebound into a
        single-node slice of the current graph, priced like any kernel."""
        from deplodock.compiler.ir.base import Op  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

        option = getattr(c.fork, "option", None)
        node_id = self._node_id(c)
        if not isinstance(option, Op) or node_id is None:
            return None
        sub = single_node_graph(c.inner.graph, node_id)
        sub.nodes[node_id].op = option
        return self._price_graph(sub, c.inner.ctx)

    def _price_graph(self, graph: Graph, ctx) -> float | None:
        """Σ of per-kernel predicted-best µs over ``graph``'s kernel-bearing
        nodes, or ``None`` when any kernel is unpriceable (no partition fork —
        e.g. a pre-tiled combine ``TileOp`` — or a failed nested descent)."""
        from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

        prices = [self._price_kernel(graph, nid, ctx) for nid, n in graph.nodes.items() if op_cache_key(n.op) is not None]
        if not prices or any(p is None for p in prices):
            return None
        return sum(prices)

    def _price_kernel(self, graph: Graph, nid: str, ctx) -> float | None:
        """One kernel's price: a nested greedy descent over its single-node
        slice through ``lowering/tile`` only (the partition fork is where the
        prior prices a complete tile row; the kernel/cuda passes add nothing
        and cost real CPU), reading the chosen leaf's predicted µs off
        ``partition_scores``. Memoized per ``op_cache_key`` so 28 identical
        per-layer kernels price once. Best-effort: any descent failure prices
        as ``None`` (→ the caller keeps the op-variant path)."""
        from deplodock.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

        key = op_cache_key(graph.nodes[nid].op)
        if key in self._price_memo:
            return self._price_memo[key]
        us: float | None = None
        try:
            nested = GreedySearch(prior=self._prior, price_structural=False)
            next(_tile_pipeline().tune(single_node_graph(graph, nid), search=nested, ctx=ctx, db=SearchDB()), None)
            us = nested.partition_scores.get(nid)
        except Exception:  # noqa: BLE001 — a price-probe failure must never break compile
            us = None
        self._price_memo[key] = us
        return us

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


# ---------------------------------------------------------------------------
# ``greedy_decide`` — the same greedy pick as a ``Run.resolve`` decide
# callback. ``Pipeline.run`` and the structural pricing probes route through
# this; the ``GreedySearch`` class above is the legacy ``drive``-protocol form
# (deleted in ``plans/resolve-trace-driver.md`` M3).
# ---------------------------------------------------------------------------

# Sentinel distinguishing "load the global prior lazily on the first fork"
# (the ``Pipeline.run`` default) from an explicitly injected prior — which may
# legitimately be ``None`` (= no prior, option-0 emission order).
_LOAD_PRIOR = object()


def _load_prior_safe():
    """Load the one global prior (learned ``CatBoostPrior`` behind the
    ``AnalyticPrior`` cold-start fallback). Best-effort: any load failure →
    ``None`` → emission order — a bad/missing prior must never break compile."""
    try:
        from deplodock.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

        return load_prior()
    except Exception:  # noqa: BLE001
        return None


def _first_leaf(option: object) -> object:
    """Descend an option to its first leaf (branch Forks take child 0) — the
    no-information emission-order pick the no-prior fallback keeps."""
    while isinstance(option, Fork) and not option.is_leaf:
        option = option.expand()[0]
    return option


def _leaf_knobs(leaf: object) -> dict:
    """A flattened leaf's complete knob row: a leaf ``Fork`` carries it as
    ``knobs``; a concrete ``Op`` carries its own; a ``Graph`` splice has no
    single row (scored structurally, never by knobs) — empty, matching the
    ``LazyCandidate.from_option`` lift the drive path used."""
    if isinstance(leaf, Fork):
        return dict(leaf.knobs)
    return dict(getattr(leaf, "knobs", None) or {}) if not isinstance(leaf, Graph) else {}


def _leaf_op(leaf: object):
    """The concrete ``Op`` behind a flattened leaf, or ``None``. Reads
    ``OptionFork.option`` rather than firing ``expand()`` — a planner tree
    ``_Leaf``'s thunk would materialize a TileOp just to inspect it."""
    from deplodock.compiler.ir.base import Op  # noqa: PLC0415

    if isinstance(leaf, Op):
        return leaf
    option = getattr(leaf, "option", None)
    return option if isinstance(option, Op) else None


def _leaf_graph(leaf: object) -> Graph:
    """The ``Graph`` behind a structural leaf (raw or ``OptionFork``-wrapped)."""
    return leaf if isinstance(leaf, Graph) else leaf.option


def _price_kernel(graph: Graph, nid: str, ctx: Context, prior, memo: dict[str, float | None]) -> float | None:
    """One kernel's price: a nested deterministic resolution of its
    single-node slice through ``lowering/tile`` only (the partition fork is
    where the prior prices a complete tile row; the kernel/cuda passes add
    nothing and cost real CPU), reading the chosen leaf's predicted µs off the
    slice-resolve's trace entry at the partition fork. Memoized per
    ``op_cache_key`` so 28 identical per-layer kernels price once.
    Best-effort: any resolve failure prices as ``None`` (→ the caller keeps
    the op-variant path)."""
    from deplodock.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    key = op_cache_key(graph.nodes[nid].op)
    if key in memo:
        return memo[key]
    us: float | None = None
    try:
        nested = greedy_decide(prior=prior, price_structural=False)
        _, trace = Run(pipeline=_tile_pipeline(), ctx=ctx).resolve(single_node_graph(graph, nid), nested)
        us = next((d.score for d in trace if d.rule_name == PARTITION_RULE and d.node_id == nid), None)
    except Exception:  # noqa: BLE001 — a price-probe failure must never break compile
        us = None
    memo[key] = us
    return us


def _price_graph(graph: Graph, ctx: Context, prior, memo: dict[str, float | None]) -> float | None:
    """Σ of per-kernel predicted-best µs over ``graph``'s kernel-bearing
    nodes, or ``None`` when any kernel is unpriceable (no partition fork —
    e.g. a pre-tiled combine ``TileOp`` — or a failed nested resolve)."""
    from deplodock.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

    prices = [_price_kernel(graph, nid, ctx, prior, memo) for nid, n in graph.nodes.items() if op_cache_key(n.op) is not None]
    if not prices or any(p is None for p in prices):
        return None
    return sum(prices)


def _price_op_leaf(fp: ForkPoint, leaf: object, prior, memo: dict[str, float | None]) -> float | None:
    """The keep-fused side's price: the leaf's ``Op`` rebound into a
    single-node slice of the current graph, priced like any kernel."""
    from deplodock.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

    option = _leaf_op(leaf)
    if option is None:
        return None
    sub = single_node_graph(fp.match.graph, fp.node_id)
    sub.nodes[fp.node_id].op = option
    return _price_graph(sub, fp.ctx, prior, memo)


def _pick_structural(fp: ForkPoint, leaves: list, prior, memo: dict[str, float | None], price_structural: bool) -> object | None:
    """Price the structural (``Graph``-splicing) leaves of one fork against
    the keep-fused ``Op`` side and return the winning structural leaf, or
    ``None`` to keep the op-variant path (cold prior, unpriceable option, or
    fused predicted faster).

    Both sides are priced the same way: the prior's predicted-best µs at each
    kernel's partition fork, obtained by a nested deterministic resolution of
    the kernel's single-node slice (``lowering/tile`` only, no backend,
    CPU-only — :func:`_price_kernel`); a structural option's price is the Σ
    over its fragment's kernels. Gated on the *trained* ``CatBoostPrior``
    (``prior.fitted``): Σ-comparisons through the analytic cold-start model
    are unvalidated, and a cold compile must never change kernel sets. Greedy
    is prior-only by design — the price never reads the DB (the learned-prior
    work removed ``_best_fork`` replay deliberately)."""
    from deplodock.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    if not price_structural or prior is None or not getattr(prior, "fitted", False):
        return None
    op_leaves = [o for o in leaves if not _is_structural_option(o)]
    if not op_leaves:
        return None  # nothing to compare against — the no-op-variant edge keeps today's scoring path
    fused_prices = [_price_op_leaf(fp, o, prior, memo) for o in op_leaves]
    if any(p is None for p in fused_prices):
        return None
    split_prices = [(o, _price_graph(_leaf_graph(o), fp.ctx, prior, memo)) for o in leaves if _is_structural_option(o)]
    split_prices = [(o, p) for o, p in split_prices if p is not None]
    if not split_prices:
        return None
    best_split, best_split_us = min(split_prices, key=lambda op_us: op_us[1])
    return best_split if best_split_us < min(fused_prices) else None


def greedy_decide(
    blocked: dict[str, set[frozenset]] | None = None,
    *,
    prior: object = _LOAD_PRIOR,
    price_structural: bool = True,
) -> Callable[[ForkPoint], object]:
    """The greedy compile pick as a :meth:`Run.resolve` ``decide`` callback:
    flatten the fork point to its complete leaves (:func:`flatten_leaves`),
    skip ``blocked`` tile identities, and take the prior's ``mean_scores``
    argmin — the learned ``CatBoostPrior`` once trained, the ``AnalyticPrior``
    cold-start heuristic otherwise (both behind ``load_prior``'s
    ``FallbackPrior``). Falls to emission order (option-0, first leaf) only if
    the prior fails to load entirely. Stamps the pick's predicted µs on
    ``fp.score``, so the resolve trace carries the per-fork price (the
    structural pricing probe reads a kernel's cost off the partition fork's
    trace entry).

    ``blocked`` (``{node_id: {tile_identity, ...}}``) lists tiles that failed
    ``validate(ctx)`` on a previous compile attempt — ``Pipeline.run`` retries
    the deterministic resolution with the failed leaf blocklisted so the next
    best non-blocked leaf is picked (the analogue of how ``tune``
    benches-and-skips an unviable tile; greedy benches nothing, so the
    validity signal must come from the retry).

    Structural (``Graph``-splicing) options are priced with the trained prior
    — :func:`_pick_structural` — so an unpinned ``compile`` / ``run`` can
    deploy the kernel sets ``tune`` measured best (the demoted-matmul split);
    cold, the structural leaf is filtered and kernel sets stay unchanged.
    ``price_structural=False`` keeps the filter behavior — used by
    ``Pipeline.run``'s retry after a structural pick failed to lower, and by
    the nested pricing probes themselves (no recursive splitting inside a
    price probe). The price memo is per-factory-call (one compile attempt),
    keyed by ``op_cache_key``."""
    from deplodock.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

    memo: dict[str, float | None] = {}  # op_cache_key → predicted µs (None = unpriceable)
    loaded = prior is not _LOAD_PRIOR
    the_prior = prior if loaded else None

    def decide(fp: ForkPoint) -> object:
        nonlocal loaded, the_prior
        if not loaded:
            loaded = True
            the_prior = _load_prior_safe()
        if the_prior is None:
            return _first_leaf(fp.options[0])  # prior failed to load → emission order
        # Flatten: greedy benches nothing, so it must pick the globally best
        # COMPLETE tile, not a partial branch — see ``flatten_leaves`` (the
        # prior is blind at a partial ``BM/BN`` branch: ``knob_features``
        # can't compute the tile's area / occupancy until ``FM/FN`` exist).
        # The pick equals scoring the flat candidate set, invariant to how
        # the lazy tree's levels are arranged.
        leaves = flatten_leaves(fp.options)
        # Structural options (Graph splices that change the kernel set): the
        # per-op prior prices ONE kernel's knob row, so its score for a
        # multi-kernel Graph option is meaningless. With the *trained* prior
        # loaded, :func:`_pick_structural` prices the option properly — Σ of
        # nested per-kernel predicted-bests vs the keep-fused side — and
        # returns the split when it predicts faster. Cold (analytic / no
        # prior), or when an option can't be priced, the structural leaf is
        # filtered so a cold compile never changes kernel sets. ``tune``
        # explores them regardless (MCTS walks every sibling); an env pin
        # makes the Graph the rule's only option, which applies inline and
        # never reaches a decide.
        if any(_is_structural_option(o) for o in leaves):
            pick = _pick_structural(fp, leaves, the_prior, memo, price_structural)
            if pick is not None:
                return pick
            op_leaves = [o for o in leaves if not _is_structural_option(o)]
            if op_leaves:
                leaves = op_leaves
        if len(leaves) <= 1:
            return leaves[0] if leaves else _first_leaf(fp.options[0])
        # The constant base under this fork's deltas: the offer op's knobs
        # (its ``S_*`` structural identity) plus the ``H_*`` host/hardware
        # regime — the feature base tune trained on (``two_level.inner_reward``).
        base = {**fp.ctx.features(), **dict(fp.root_op.knobs)}
        # Tiles this node already failed to lower on an earlier attempt — skip
        # the matching leaf so greedy falls back to the next prior-ranked one.
        node_blocked = blocked.get(fp.node_id) if blocked else None
        live = [(o, _leaf_knobs(o)) for o in leaves]
        if node_blocked is not None:
            live = [(o, k) for o, k in live if not _tile_blocked(k, node_blocked)]
        if not live:  # every leaf blocklisted → no valid alternative left
            return leaves[0]
        scores = the_prior.mean_scores([{**base, **k} for _, k in live])  # predicted µs — lower is better
        best_i = min(range(len(live)), key=scores.__getitem__)
        fp.score = scores[best_i]
        return live[best_i][0]

    return decide
