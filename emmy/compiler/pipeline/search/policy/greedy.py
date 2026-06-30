"""The greedy compile pick — :func:`greedy_decide`, a ``Run.resolve`` decide
factory picking each fork point's globally-best **complete** leaf via the
global learned prior when one is trained, else option-0.

This is the deterministic pick for ``compile`` / ``run``, the structural
pricing probes, and the assembled-graph lowering. It is NOT a search and not
a ``Search`` policy: there is no frontier to rank, no tree, no benching — a
deterministic resolution is a fold over the pipeline (at each fork, a pure
function of ``(options, op, prior)``, argmin, continue), so its process state
is :meth:`Run.resolve`'s returned trace, never accumulated policy attributes.
It can only *use* a prior trained earlier by ``tune``, never train one.
Exploration stays in :class:`~.mcts.TuningSearch` (``Pipeline.tune``).

**Flatten, don't descend.** The lazy fork tree (``lowering/tile`` planner) is an
MCTS data structure — it stages knob choices across levels (``BR`` → ``BM/BN`` →
``FM/FN``) so MCTS pays one node per pop. Greedy must NOT walk it level-by-level:
a branch carries only a *partial* tile, and ``knob.knob_features`` can't compute
the tile's area / occupancy until ``FM/FN`` are pinned — so the prior is blind at
the ``BM/BN`` choice and defaults to ``BN=16`` for every shape. Instead greedy
**flattens** each fork point to its complete leaves
(:func:`~emmy.compiler.pipeline.fork.flatten_leaves` — cheap, ``expand``
builds only knob dicts; materialization stays deferred to the one chosen leaf)
and picks the one with the lowest :meth:`Prior.mean_scores` over the
full feature vector the prior trained on: the ``H_*`` host/hardware regime + the
op's ``S_*`` structural knobs (read off the offer op) + the leaf's complete knob
row. The pick equals scoring the flat candidate set, invariant to the tree's
level order. With no trained prior the model is unfit → it falls back to the first
emitted sibling (option-0).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.fork import Fork, flatten_leaves

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.pipeline import ForkPoint


@lru_cache(maxsize=1)
def _tile_pipeline():
    """The ``lowering/tile``-only pipeline the structural price probes drive —
    frozen and shareable, so one load serves every nested descent."""
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    return Pipeline.build(["lowering/tile/enumeration"])


# Tile-identity knobs a blocklist entry keys on — the planner/enumeration choices
# that fully determine a tile (so two leaves are "the same tile" iff these match).
# Excludes the post-lowering staging knobs (RING / STAGE), which are stamped after
# the greedy fork pick and differ between the leaf and the rejected node. Every
# tile-geometry family is now native (``SPLIT@``/``REDUCE@``/``ATOM@`` per-element keys),
# matched by key prefix since the element names are per-kernel; no legacy exact-name
# knob remains.
_TILE_IDENTITY: tuple[str, ...] = ()
_TILE_IDENTITY_PREFIXES = ("REDUCE@", "SPLIT@", "ATOM@")

# The rule whose fork prices a kernel: the prior's predicted µs for the chosen
# complete tile row at the partition fork is the per-kernel cost the structural
# pricing sums (defined here, not in ``two_level``, because that module imports
# this package at module scope — the reverse would cycle).
PARTITION_RULE = "100_register_tile"


def tile_identity(knobs: dict) -> frozenset:
    """The blocklist key for a tile — its planner-chosen knobs as a hashable set.
    Computed identically for a greedy leaf's fork knobs and for a rejected node's
    realized knobs, so :func:`greedy_decide` can skip a leaf that already failed
    ``validate(ctx)`` downstream (the smem / thread-budget gate)."""
    exact = [(k, str(knobs[k])) for k in _TILE_IDENTITY if k in knobs]
    native = [(k, str(v)) for k, v in knobs.items() if k.startswith(_TILE_IDENTITY_PREFIXES)]
    return frozenset(exact + native)


def _tile_blocked(fork_knobs: dict, blocked: set[frozenset]) -> bool:
    """True if a leaf's complete knob row matches a blocklisted tile. Only a leaf
    fork carries every identity knob, so a partial (branch) fork — whose identity
    is a strict subset — never equals a full-row entry and is never skipped."""
    return tile_identity(fork_knobs) in blocked


# ---------------------------------------------------------------------------
# ``greedy_decide`` — the greedy pick as a ``Run.resolve`` decide callback.
# ``Pipeline.run`` and the structural pricing probes route through this.
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
        from emmy.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

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
    from emmy.compiler.ir.base import Op  # noqa: PLC0415

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
    from emmy.compiler.pipeline.pipeline import Run  # noqa: PLC0415
    from emmy.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

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
    from emmy.compiler.pipeline.search.keys import op_cache_key  # noqa: PLC0415

    prices = [_price_kernel(graph, nid, ctx, prior, memo) for nid, n in graph.nodes.items() if op_cache_key(n.op) is not None]
    if not prices or any(p is None for p in prices):
        return None
    return sum(prices)


def _price_op_leaf(fp: ForkPoint, leaf: object, prior, memo: dict[str, float | None]) -> float | None:
    """The keep-fused side's price: the leaf's ``Op`` rebound into a
    single-node slice of the current graph, priced like any kernel."""
    from emmy.compiler.pipeline.search.slice import single_node_graph  # noqa: PLC0415

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
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

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
    from emmy.compiler.pipeline.pipeline import _is_structural_option  # noqa: PLC0415

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
        rows = [{**base, **k} for _, k in live]
        picker = getattr(the_prior, "pick", None)
        if picker is not None:
            # ``Prior.pick``: measured -O3 reservoir evidence first, model argmin
            # otherwise — a config the tune proved fastest at -O3 must not lose
            # the deploy to an unmeasured extrapolation (still prior-only: the
            # evidence ships inside the prior's checkpoint, not the DB).
            best_i, price = picker(rows)
        else:  # bare-mean_scores prior object (tests / custom callers)
            scores = the_prior.mean_scores(rows)
            best_i = min(range(len(live)), key=scores.__getitem__)
            price = scores[best_i]
        fp.score = price  # measured µs when evidence decided, predicted µs otherwise
        return live[best_i][0]

    return decide
