"""Stage frequently-reused external inputs into shared memory.

Runs early in the tile-lowering chain — *before* ``008_register_tile``
— so the classifier sees the clean PAT × PAT thread-axis layout from
``005_blockify_launch``. ``register_tile`` σ-substitutes cache-axis
Vars in the consumer Loads (Stages stay singleton across F²; only
consumer Loads multiply).

**Pipeline:**

1. **Walk the scope**, collecting every Load from every reduce
   ``Loop`` / ``StridedLoop``, tagged with its reduce axis.
2. **Group by source buffer.** Sibling reduces (softmax max/sum/output)
   contribute their Loads to the same buffer's bucket; per-axis-name
   differences in the reduce axis are normalized away by the slab
   signature (reduce axes contribute extent only to the pattern).
3. **Per buffer, fit one slab.** Classify every Load. If they all agree
   on slab geometry, emit one Stage; if they disagree, bail — that
   buffer's Loads stay on DRAM. Bailing is preferred to per-pattern
   partitioning: the cost is missing optimization on rare buffers with
   multiple distinct access shapes within one scope.
4. **Admit & emit.** Greedily admit Stages until the per-scope smem
   budget is hit; emit at scope head; rewrite admitted Loads to read
   from staged smem.

**Slab geometry from a Load's index** (computed in ``_classify``):

- ``origin`` — the index with every cache var (thread + reduce axis)
  substituted to 0. The per-CTA anchor.
- For each cache var, find which source dim its Var appears in.
  Zero dims = fan-in axis (this var's threads/iterations all read the
  same staged value). One dim = cache axis at that dim. Multiple dims
  = bail (collapsed-reshape that can't be additively split).
- Coefficient-1 check: substitute the var → 1 (others → 0) and compare
  to ``origin[d] + 1``. If any axis fails (collapsed-reshape with a
  surviving stride), emit ``TemplateAddressing`` carrying the original
  Load index instead of ``AffineAddressing``.

**Reuse.** A Load qualifies for staging iff at least one bound axis
(thread or reduce) doesn't appear in its index — that axis's threads
or iterations all read the same staged value. If every bound axis
appears, no fan-in, skip.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, AffineAddressing, Stage, TemplateAddressing, TileOp, trivial_stage_body
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]


class _Slab(NamedTuple):
    """Slab geometry derived from one Load's index."""

    origin: tuple[Expr, ...]
    cache_axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]
    template: tuple[Expr, ...] | None
    n_bytes: int


def rewrite(root: Node, ctx) -> list[TileOp] | None:
    """Emit one TileOp option per subset of stage-able input buffers,
    ordered most-staged first. Option-0 stages every qualifying buffer
    (the original behavior — best perf when smem fits). Subsequent
    options progressively drop buffers (largest slab first, freeing the
    most smem). The final option stages nothing (a no-op body rewrite).
    The engine's :meth:`Op.validate` filter then picks the first
    surviving variant — greedy gets the highest-perf one that fits;
    the autotuner explores them all.

    Idempotence is gated on the ``stage_inputs`` knob (stamped on every
    emitted variant) rather than on body structure, so the no-staging
    variant — whose body matches the parent — still has a distinct
    ``op_cache_key`` from the unstaged input and the rule doesn't
    re-fire on it."""
    if root.op.knobs.get("stage_inputs"):
        raise RuleSkipped("stage_inputs already applied (idempotence via knob)")
    budget = ctx.max_dynamic_smem
    variants = _enumerate_variants(root.op.body, slab_cap=budget, scope_budget=budget, parent_op=root.op)
    if not variants:
        raise RuleSkipped("no Load qualifies for staging")
    return variants


_WARP_SIZE = 32


def _forced_stage_mask(n: int) -> int | None:
    """Parse ``DEPLODOCK_STAGE_INPUTS`` as an int (decimal or ``0x``-hex)
    and clamp to ``n`` bits. ``None`` when unset, so the rule falls back
    to enumerating every subset."""
    raw = os.environ.get("DEPLODOCK_STAGE_INPUTS")
    if raw is None or raw == "":
        return None
    return int(raw, 0) & ((1 << n) - 1)


def _enumerate_variants(body: Body, *, slab_cap: int, scope_budget: int, parent_op: TileOp) -> list[TileOp]:
    """Enumerate one TileOp variant per subset of stage-able buffers —
    ``2^N`` options for ``N`` candidates. Ordered most-staged first
    (greedy / option-0 = stage everything qualifying), then by descending
    population count; the empty-subset variant comes last and is the
    "stage nothing" fallback whose body matches the parent.

    Each variant carries a ``stage_inputs=True`` knob plus a per-buffer
    ``stage:<buf>=True|False`` knob, giving each variant a distinct
    ``op_cache_key`` even when bodies collide. The knob is the rule's
    idempotence anchor so re-firing on the no-staging variant doesn't
    loop.

    ``DEPLODOCK_STAGE_INPUTS`` overrides the enumeration with a single
    fixed mask over the ranked buffer list (bit ``i`` selects the i-th
    buffer): ``0`` stages nothing, ``0xFFFF`` (any all-bits-set value)
    stages everything that qualifies, and a specific bitmask stages
    exactly that subset. Used by exhaustive autotune tests to mimic the
    legacy single-variant behavior and avoid the ``2^N`` blow-up
    multiplying with every other fork point.

    Returns ``[]`` when no buffer qualifies for staging (no fan-in)."""
    candidates = _candidate_buffers(body)
    if not candidates:
        return []
    # Rank by descending slab size so larger buffers get priority within
    # a given population count — the highest-perf variant at each level
    # of staging coverage sits earliest in the option list.
    ranked = sorted(candidates, key=lambda kv: -kv[1])
    bufs_ranked = [b for b, _ in ranked]
    n = len(bufs_ranked)
    # All 2^N subsets, ordered by descending popcount (most-staged first),
    # then by ascending mask within each popcount bucket so the same
    # population stays grouped. ``DEPLODOCK_STAGE_INPUTS=<mask>`` pins
    # the enumeration to a single mask (clamped to ``n`` bits), matching
    # legacy single-variant behavior when set to ``0xFFFF`` (all-on).
    forced = _forced_stage_mask(n)
    if forced is not None:
        masks = [forced]
    else:
        masks = sorted(range(1 << n), key=lambda m: (-bin(m).count("1"), m))
    variants: list[TileOp] = []
    for mask in masks:
        allow = frozenset(b for i, b in enumerate(bufs_ranked) if mask & (1 << i))
        new_body = _maybe_rewrite(body, slab_cap=slab_cap, scope_budget=scope_budget, allowed_bufs=allow)
        if new_body is None:
            continue
        knobs = {**parent_op.knobs, "stage_inputs": True}
        for b in bufs_ranked:
            knobs[f"stage:{b}"] = b in allow
        variants.append(TileOp(body=new_body, name=parent_op.name, knobs=knobs))
    return variants


def _candidate_buffers(body: Body) -> list[tuple[str, int]]:
    """List of ``(buf_name, slab_bytes)`` for every buffer that would
    qualify for staging under the default classifier. Used to drive the
    per-variant allow-set enumeration."""
    idx, tile = single_tile(body)
    if any(isinstance(s, Stage) for s in tile.body.iter()):
        return []
    if not tile.thread_axes:
        return []
    n_thread = 1
    for ba in tile.thread_axes:
        n_thread *= int(ba.extent)
    if n_thread <= _WARP_SIZE:
        return []
    block_axis_names = frozenset(ax.name for ax in tile.block_axes)
    found: dict[str, int] = {}
    _collect_candidates(tile, tile.thread_axes, tile.all_axes, block_axis_names, found, slab_cap=10**12)
    return list(found.items())


def _collect_candidates(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    found: dict[str, int],
    *,
    slab_cap: int,
) -> None:
    """Mirror of ``_process_scope``'s buffer-walk side: record each
    candidate buffer's classified slab size without building Stages."""
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...]]]] = {}
    for s in scope.body:
        if isinstance(s, Loop) and not s.is_reduce:
            _collect_candidates(s, thread_axes, (*in_scope_axes, s.axis), block_axis_names, found, slab_cap=slab_cap)
            continue
        if isinstance(s, (Loop, StridedLoop)):
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes))
    for buf, items in loads_by_buf.items():
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _ in items):
            continue
        for load, reduce_axis, scope_axes in items:
            slab = _classify(load, thread_axes, reduce_axis, scope_axes, slab_cap=slab_cap)
            if slab is None:
                continue
            # Sum slab sizes across all distinct slabs in this buffer.
            found[buf] = found.get(buf, 0) + slab.n_bytes
            break  # one representative per partition; budget accounting is best-effort


def _maybe_rewrite(body: Body, *, slab_cap: int, scope_budget: int, allowed_bufs: frozenset[str] | None = None) -> Body | None:
    """Stage every qualifying Load in ``body`` (or only those whose
    buffer is in ``allowed_bufs`` when supplied) into a smem ``Stage``.

    Returns the rewritten ``body`` when any Stage was admitted; ``None``
    when no Stage fits or the tile is structurally unstage-able (idempotent
    re-fire, no thread axes, warp-only cooperative). The variant
    enumerator in :func:`_enumerate_variants` treats ``None`` as "skip
    this allow-set" and tries the next subset.
    """
    idx, tile = single_tile(body)
    # Idempotence is now gated on the ``stage_inputs`` knob in ``rewrite``;
    # the structural ``any(Stage)`` check would block legitimate re-firing
    # of the partial-staging variants where some Stages are present.
    if not tile.thread_axes:
        if allowed_bufs is None:
            raise RuleSkipped("Tile has no thread_axes — no reuse to stage")
        return None

    # Warp-only cooperative tiles (one thread axis, extent ≤ WARP_SIZE)
    # don't benefit from a smem stage: ``materialize_tile`` will emit a
    # register-only ``WarpShuffle`` combine, the row fits in registers
    # across the warp, and L1 absorbs second/third loads of the same
    # row. Skip staging entirely so the kernel stays smem-free.
    n_thread = 1
    for ba in tile.thread_axes:
        n_thread *= int(ba.extent)
    if n_thread <= _WARP_SIZE:
        if allowed_bufs is None:
            raise RuleSkipped(f"warp-only cooperative tile (n_threads={n_thread} ≤ {_WARP_SIZE}); register-resident, no smem stage")
        return None

    used_names: set[str] = set()
    block_axis_names = frozenset(ax.name for ax in tile.block_axes)
    new_tile_body = _process_scope(
        tile,
        tile.thread_axes,
        tile.all_axes,
        block_axis_names,
        used_names,
        slab_cap=slab_cap,
        scope_budget=scope_budget,
        allowed_bufs=allowed_bufs,
    )
    if new_tile_body == tile.body:
        # No stage admitted. For the variant enumerator this is the
        # "unstaged" option; return body unchanged. The legacy single-rewrite
        # entry point still raises RuleSkipped (allowed_bufs=None).
        if allowed_bufs is None:
            raise RuleSkipped("no Load qualifies for staging")
        return body
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
    allowed_bufs: frozenset[str] | None = None,
) -> Body:
    """Free Loops recurse; reduce loops contribute Loads to this scope's
    per-buffer bucket. Per buffer, derive one slab if all Loads agree;
    admit under budget; emit Stages at scope head; rewrite Loads."""
    rewritten_inner: list[Stmt] = []
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...]]]] = {}

    for s in scope.body:
        if isinstance(s, Loop) and not s.is_reduce:
            rewritten_inner.append(
                Loop(
                    axis=s.axis,
                    body=_process_scope(
                        s,
                        thread_axes,
                        (*in_scope_axes, s.axis),
                        block_axis_names,
                        used_names,
                        slab_cap=slab_cap,
                        scope_budget=scope_budget,
                        allowed_bufs=allowed_bufs,
                    ),
                )
            )
            continue
        if isinstance(s, (Loop, StridedLoop)):
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes))
        rewritten_inner.append(s)

    if allowed_bufs is not None:
        loads_by_buf = {b: items for b, items in loads_by_buf.items() if b in allowed_bufs}
    stages, name_rewrites = _build_stages(
        loads_by_buf, thread_axes, block_axis_names, used_names, slab_cap=slab_cap, scope_budget=scope_budget
    )
    if not stages:
        return tuple(rewritten_inner)

    rewritten = tuple(_rewrite_loads(s, name_rewrites) for s in rewritten_inner)
    return tuple([*stages, *rewritten])


def _build_stages(
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...]]]],
    thread_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
) -> tuple[list[Stage], dict[str, tuple[str, tuple[Expr, ...]]]]:
    """Per buffer, partition Loads by structural index equality (within a
    loop they collapse only when textually identical — Python-scope
    shadowing provides this for sibling reduce loops). Each partition
    becomes a candidate Stage; classify the representative, admit if
    it fits the per-scope smem budget. Multiple distinct slabs per
    buffer are allowed (e.g. rotate-half kernels load the same buffer
    at multiple conditional positions, each requiring its own slab)."""
    stages: list[Stage] = []
    name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    used_bytes = 0

    for buf, items in loads_by_buf.items():
        # Skip buffers whose Loads don't reference any BLOCK axis. The
        # data is identical for every CTA, so smem staging only buys a
        # per-CTA copy of the same bytes — wasted smem (kills occupancy)
        # plus an extra cooperative-load cycle plus a syncthreads. The
        # global Load on the original DRAM pointer hits L2 (after the
        # first CTA brings it in) and L1 read-only cache thereafter,
        # which is faster than re-staging in every block. RMSNorm's
        # ``w[k]`` and any matmul's frozen-weight Loads are the
        # canonical examples.
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _ in items):
            continue

        # Partition this buffer's Loads by structural index equality.
        # Each partition gets its own slab; admission decides which fit.
        partitions: list[tuple[Load, Axis, tuple[Axis, ...], list[Load]]] = []
        for load, reduce_axis, scope_axes in items:
            for rep_load, _, _, members in partitions:
                if load.index == rep_load.index:
                    members.append(load)
                    break
            else:
                partitions.append((load, reduce_axis, scope_axes, [load]))

        for rep_load, rep_reduce, rep_scope, members in partitions:
            slab = _classify(rep_load, thread_axes, rep_reduce, rep_scope, slab_cap=slab_cap)
            if slab is None:
                continue
            if used_bytes + slab.n_bytes > scope_budget:
                continue
            smem_name = _gen_name(buf, used_names)
            addressing: AffineAddressing | TemplateAddressing = (
                TemplateAddressing(exprs=slab.template) if slab.template is not None else AffineAddressing(dims=slab.slab_dims)
            )
            stages.append(
                Stage(
                    name=smem_name,
                    axes=slab.cache_axes,
                    body=trivial_stage_body(smem_name, buf, slab.origin, slab.cache_axes, addressing),
                )
            )
            used_bytes += slab.n_bytes
            smem_index = tuple(Var(ax.name) for ax in slab.cache_axes)
            for load in members:
                name_rewrites[load.name] = (smem_name, smem_index)
    return stages, name_rewrites


def _rewrite_loads(stmt: Stmt, name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]]) -> Stmt:
    if not name_rewrites:
        return stmt

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.name in name_rewrites:
            smem_name, new_index = name_rewrites[s.name]
            return Load(name=s.name, input=smem_name, index=new_index)
        return s

    return Body((stmt,)).map(fn)[0]


def _classify(
    load: Load,
    thread_axes: tuple[Axis, ...],
    reduce_axis: Axis,
    scope_axes: tuple[Axis, ...],
    *,
    slab_cap: int,
) -> _Slab | None:
    """Derive the slab for ``load`` by evaluating ``load.index`` over
    the cache vars (thread axes + this loop's reduce axis):

    1. ``origin`` = ``index`` with every cache var → 0.
    2. For each cache var, find which source dim its Var appears in.
       Zero dims = fan-in axis (won't constrain the slab). One dim =
       cache axis at that dim. Multiple dims = bail.
    3. Coefficient-1 check: substitute one cache var → 1 (others → 0)
       and compare to ``origin[d] + 1``. If any axis disagrees, emit
       ``TemplateAddressing`` instead of ``AffineAddressing``.

    Returns ``None`` when no fan-in axis exists (no reuse), when a
    cache var spans multiple dims, or when the slab exceeds the cap.
    """
    candidates = (*thread_axes, reduce_axis)
    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})
    candidate_names = tuple(ax.name for ax in candidates)

    zero_sigma = Sigma({n: Literal(0, "int") for n in candidate_names})
    origin = tuple(zero_sigma.reduce(e, ctx) for e in load.index)

    var_to_dim: dict[str, int] = {}
    for ax in candidates:
        dims = [d for d, e in enumerate(load.index) if ax.name in e.free_vars()]
        if not dims:
            continue
        if len(dims) > 1:
            return None
        var_to_dim[ax.name] = dims[0]

    # No cache axis appears, or every bound axis appears (no fan-in) → no slab.
    if not var_to_dim or len(var_to_dim) == len(candidates):
        return None

    cache_axes_unsorted = tuple(ax for ax in candidates if ax.name in var_to_dim)
    # Sort cache axes by their source-dim index so the source's innermost
    # dim ends up innermost in smem. This preserves source layout, which
    # matters for SIMT FP32 LDS.128 vectorization: per-thread reads of
    # consecutive cells along the source's innermost dim become a single
    # 16-byte transaction. For matmul B in [K, N] layout this puts N
    # innermost in smem (so per-thread N-tile reads vectorize); for B in
    # [N, K] layout it puts K innermost (the existing layout). The
    # default unsorted order placed the reduce axis last regardless of
    # source — fine for [N, K] sources but blocked vectorization on
    # [K, N] B which is exactly the cuBLAS-beating SGEMM convention.
    cache_axes = tuple(sorted(cache_axes_unsorted, key=lambda ax: var_to_dim[ax.name]))
    slab_dims = tuple(var_to_dim[ax.name] for ax in cache_axes)
    n_bytes = BYTES_PER_ELEM
    for ax in cache_axes:
        n_bytes *= int(ax.extent)
    if n_bytes > slab_cap:
        return None

    needs_template = False
    for ax in cache_axes:
        d = var_to_dim[ax.name]
        unit_sigma = Sigma({n: Literal(1 if n == ax.name else 0, "int") for n in candidate_names})
        actual = sorted(t.pretty() for t in _flatten_add(unit_sigma.reduce(load.index[d], ctx)))
        expected = sorted(t.pretty() for t in _flatten_add((origin[d] + Literal(1, "int")).simplify(ctx)))
        if actual != expected:
            needs_template = True
            break
    template = tuple(load.index) if needs_template else None

    return _Slab(
        origin=origin,
        cache_axes=cache_axes,
        slab_dims=slab_dims,
        template=template,
        n_bytes=n_bytes,
    )


def _load_free_vars(load: Load) -> frozenset[str]:
    """Union of free variable names across every index dim of ``load``."""
    out: set[str] = set()
    for e in load.index:
        out |= e.free_vars()
    return frozenset(out)


def _flatten_add(e: Expr) -> list[Expr]:
    if isinstance(e, BinaryExpr) and e.op == "+":
        return _flatten_add(e.left) + _flatten_add(e.right)
    return [e]


def _gen_name(buf: str, used: set[str]) -> str:
    base = f"{buf}_smem"
    if base not in used:
        used.add(base)
        return base
    n = 1
    while f"{base}_{n}" in used:
        n += 1
    name = f"{base}_{n}"
    used.add(name)
    return name
