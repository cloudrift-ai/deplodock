"""Stage frequently-reused external inputs into shared memory (wrap-body).

Emits ``StageBundle(sources=[...], body=<consumer>)`` — the bundle *wraps*
the consumer subtree containing the rewritten Loads. Producer cooperative
``Load+Write`` is synthesized at materialize time from ``Source``
entries (cache axes, origin, source-dim mapping); no producer body is
stored.

Runs *before* ``010_split_register_axes`` (pre-register-tile: there
is exactly one Load per ``(buffer, access-pattern)`` rather than F×F
duplicates). REGISTER axes in scope join cache axes via the
``register_axes`` channel — the slab spans ``BM·FM × BK`` (and similar)
with stride-1 Affine addressing, instead of ``BM × BK`` with
TemplateAddressing as it would post-replicate. Stages emit wrapping
the K_o body; their cache-axis iteration vars shadow the outer
REGISTER Loops, and ``010_split_register_axes`` treats a ``StageBundle``
as opaque for replication (its ``sources`` carry no body — only the
consumer ``body`` / compute phase descend).

**Pipeline:**

1. **Walk the scope**, collecting every Load from every reduce
   ``Loop`` / ``StridedLoop``, tagged with its reduce axis.
2. **Group by source buffer.** Sibling reduces contribute their Loads
   to the same buffer's bucket; per-axis-name differences in the
   reduce axis are normalized away by the slab signature.
3. **Per buffer, fit one slab.** Classify every Load. If they all agree
   on slab geometry, build one Source; if they disagree, bail.
4. **Admit & emit.** Greedily admit Sources until the per-scope smem
   budget is hit; emit a single ``StageBundle`` wrapping the consumer
   stmts; rewrite admitted Loads to read from staged smem.

**Slab geometry from a Load's index** (computed in ``_classify``):

- ``origin`` — the index with every cache var (thread + reduce axis)
  substituted to 0. The per-CTA anchor.
- For each cache var, find which source dim its Var appears in. Zero
  dims = fan-in axis; one dim = cache axis at that dim; multiple dims =
  bail (collapsed-reshape that can't be additively split).
- Two-tier affine recognition (``plans/mma-smem-staging.md``):

  1. **Coefficient-1 composite**: substitute the var → 1 (others → 0)
     and compare to ``origin[d] + 1``. Admits scalar paths (matmul,
     RMSNorm, softmax) byte-clean; stamps
     ``AffineAddressing(dims=..., block=())``.
  2. **Block-stamped composite** (only when parent has ``MMA``):
     extract the literal coefficient on each cache var, divide out the
     running ``extent · block`` suffix product, and stamp
     ``AffineAddressing(dims=..., block=(c_0, c_1, ...))``. The slab
     grows by ``block[i]`` per cache dim so the mma.sync fragment-load can
     read a full ``atom_M × atom_K`` cell.

  Scalar paths that hit a non-1 σ coef fall through to template — the
  block path is gated on MMA so a register-axis sitting between
  origin and cache doesn't get misread as a block multiplier.

**Reuse.** A Load qualifies for staging iff at least one bound axis
doesn't appear in its index (fan-in). When every bound axis appears,
no fan-in, skip — unless ``allow_no_fan_in`` is set (≥ 2 sibling Loads
sharing the same index ⇒ temporal reuse).

**WarpTile / AtomTile passthrough.** When the parallel-tile is a
WarpTile (MMA path), ``n_thread`` is multiplied by ``warp_size`` so the
cooperative-load fan-out covers all CTA lanes, not just the warp
count. ``_collect_candidates`` and ``_process_scope`` recurse
transparently into ``AtomTile`` — its axes parameterize the mma.sync
instruction shape and contribute as ``block`` multipliers rather than
register iteration.

**Masked-tile ``Cond`` passthrough.** When ``010_partition_loops`` wraps
the parallel-tile body in a boundary ``Cond(decoded_coord < bound, ...)``
(non-divisor / symbolic axis), this pass descends transparently and emits
the ``StageBundle`` inside the rewritten Cond body. The follow-on pass
``021_hoist_staged_loads_above_mask`` lifts the K-pipeline above the Cond
so the cooperative load fires for every thread (TMA's elected issuer /
cp.async's full-CTA fan-out would otherwise be gated out).
"""

from __future__ import annotations

import os
from typing import NamedTuple

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AtomTile,
    GridTile,
    RegisterTile,
    SerialTile,
    Source,
    StageBundle,
    StagePolicy,
    StridedTile,
    TemplateAddressing,
    ThreadTile,
    TileOp,
    WarpTile,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    parallel_tile_of,
    replace_parallel_tile_body,
    single_tile,
)

PATTERN = [Pattern("root", TileOp)]

STAGE = Knob("STAGE", KnobType.BINMASK, help="Bitmask over ranked candidate buffers (char i = buffer i)")


def _tile_is_cooperative(tt: ThreadTile) -> bool:
    """True iff some Accum inside ``tt`` reduces over one of ``tt``'s
    thread axes — i.e. the kernel is cooperative-K. Replaces the legacy
    ``bool(tt.cooperative_axes)`` check now that cooperativity is derived
    from ``Accum.axes`` (see ``ir/tile/escape_analysis.py``)."""
    tt_axis_names = frozenset(ax.name for ax in tt.axes)
    for s in tt.body.iter():
        if isinstance(s, Accum) and tt_axis_names & frozenset(s.axes):
            return True
    return False


class _Slab(NamedTuple):
    """Slab geometry derived from one Load's index."""

    origin: tuple[Expr, ...]
    cache_axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]
    template: tuple[Expr, ...] | None
    n_bytes: int
    # Per-cache-axis structural block multiplier extracted from σ literal
    # coefficients. ``()`` = all-1s (the scalar / pre-M3 case); a non-
    # trivial tuple aligns with ``cache_axes`` and records the per-axis
    # factor (e.g. ``atom_M`` / ``atom_K`` for an MMA-strided load). Fed
    # straight into ``AffineAddressing.block`` at Source construction.
    block: tuple[int, ...] = ()


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    """Emit one TileOp option per subset of stage-able input buffers,
    ordered most-staged first. Option-0 stages every qualifying buffer
    (best perf when smem fits). Subsequent options progressively drop
    buffers. The final option stages nothing.

    Idempotence is gated on the ``STAGE`` knob (stamped on every
    emitted variant) rather than on body structure, so the no-staging
    variant still has a distinct ``op_cache_key``.
    """
    if STAGE.name in root.op.knobs:
        raise RuleSkipped("stage already applied (idempotence via knob)")
    budget = ctx.max_dynamic_smem
    atom_kind = root.op.knobs.get("MMA")
    # Per-buffer bytes-per-elem so ``_classify``'s slab-cap check sizes
    # fp16 slabs at 2 B/elem instead of the BYTES_PER_ELEM=4 hardcoded
    # over-count. Without this the half-precision MMA path rejects every
    # admissable staged variant (the block-scaled slab gets over-counted
    # 2× and trips ``slab_cap``).
    bytes_per_elem = {buf: int(t.dtype.nbytes) for buf, t in root.op.inputs.items() if t.dtype is not None}
    variants = _enumerate_variants(
        root.op.body,
        slab_cap=budget,
        scope_budget=budget,
        parent_op=root.op,
        warp_size=ctx.warp_size,
        atom_kind=atom_kind,
        bytes_per_elem=bytes_per_elem,
    )
    if not variants:
        raise RuleSkipped("no Load qualifies for staging")
    return variants


def _forced_stage_mask(n: int) -> int | None:
    raw = os.environ.get(STAGE.env)
    if raw is None or raw == "":
        return None
    return STAGE.parse(raw, width=n)


def _enumerate_variants(
    body: Body,
    *,
    slab_cap: int,
    scope_budget: int,
    parent_op: TileOp,
    warp_size: int,
    atom_kind: str | None = None,
    bytes_per_elem: dict[str, int] | None = None,
) -> list[TileOp]:
    candidates = _candidate_buffers(body, warp_size=warp_size)
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda kv: -kv[1])
    bufs_ranked = [b for b, _ in ranked]
    n = len(bufs_ranked)
    forced = _forced_stage_mask(n)
    if forced is not None:
        masks = [forced]
    elif atom_kind is not None:
        # MMA operands feed ``ldmatrix`` (smem→register only — no gmem-direct
        # path), so every staged candidate of an atom kernel MUST land in smem.
        # The no-stage / partial masks the scalar path enumerates would leave an
        # operand gmem-direct and crash in ``kernel/005_lower_atom_tile``; the
        # greedy picker, ranking by prior alone, would otherwise be free to pick
        # one. Offer only the fully-staged variant for the atom path.
        masks = [(1 << n) - 1]
    else:
        masks = sorted(range(1 << n), key=lambda m: (-bin(m).count("1"), m))
    variants: list[TileOp] = []
    for mask in masks:
        allow = frozenset(b for i, b in enumerate(bufs_ranked) if mask & (1 << i))
        new_body = _maybe_rewrite(
            body,
            slab_cap=slab_cap,
            scope_budget=scope_budget,
            allowed_bufs=allow,
            warp_size=warp_size,
            atom_kind=atom_kind,
            bytes_per_elem=bytes_per_elem,
        )
        if new_body is None:
            continue
        knobs = {**parent_op.knobs, STAGE.name: STAGE.pretty(mask, width=n)}
        variants.append(TileOp(body=new_body, name=parent_op.name, knobs=knobs))
    return variants


def _candidate_buffers(body: Body, *, warp_size: int) -> list[tuple[str, int]]:
    idx, outer = single_tile(body)
    tt = parallel_tile_of(outer)
    if any(isinstance(s, StageBundle) for s in tt.body.iter()):
        return []
    if not tt.axes:
        return []
    n_thread = 1
    for ax in tt.axes:
        n_thread *= ax.extent.as_static()
    # WarpTile axes count warps, not lanes; multiply by warp width to get
    # the CTA thread count the cooperative load actually has to fan out
    # across (one cooperative slab fill per CTA, all 4 × 32 = 128 lanes
    # participate even though the per-warp mma.sync atom is the consumer).
    if isinstance(tt, WarpTile):
        n_thread *= warp_size
    if n_thread <= warp_size:
        return []
    block_axes = outer.axes if isinstance(outer, GridTile) else ()
    block_axis_names = frozenset(ax.name for ax in block_axes)
    is_cooperative = _tile_is_cooperative(tt)
    all_axes = tuple(block_axes) + tuple(tt.axes)
    found: dict[str, int] = {}
    _collect_candidates(tt, tt.axes, all_axes, block_axis_names, found, slab_cap=10**12, is_cooperative=is_cooperative)
    return list(found.items())


def _collect_candidates(
    scope,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    found: dict[str, int],
    *,
    slab_cap: int,
    is_cooperative: bool = False,
    register_axes: tuple[Axis, ...] = (),
) -> None:
    """Preflight buffer-walk: record candidate buffers' slab sizes."""
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]] = {}
    for s in scope.body:
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "serial_outer":
            inner_stage = _peel_to_stage_inner(s)
            if inner_stage is not None:
                scope_axes = (*in_scope_axes, s.axis, inner_stage.axis)
                extra = (s.axis,) + register_axes
                for stmt in inner_stage.body:
                    if isinstance(stmt, Load):
                        loads_by_buf.setdefault(stmt.input, []).append((stmt, inner_stage.axis, scope_axes, extra))
                continue
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "stage_inner":
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
            continue
        if isinstance(s, RegisterTile):
            new_register_axes = register_axes + tuple(s.axes)
            _collect_candidates(
                s,
                thread_axes,
                tuple(in_scope_axes) + tuple(s.axes),
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=new_register_axes,
            )
            continue
        if isinstance(s, AtomTile):
            # MMA fragment-shape wrapper: ``005_lower_atom_tile`` consumes
            # this and re-emits the body as an Mma* chain whose
            # ``MmaLoad``s read from the smem slab we're about to admit
            # here. ``AtomTile.axes`` parameterize the mma.sync instruction
            # shape, not loop iteration — they appear in the σ as
            # ``cache_var * atom_M``-style stride multipliers and become
            # ``AffineAddressing.block`` factors, so they don't get
            # added to ``register_axes``.
            _collect_candidates(
                s,
                thread_axes,
                in_scope_axes,
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
            )
            continue
        if isinstance(s, Cond):
            # Masked-tile boundary guard wrapping the K-loop chain (FM/FN
            # × BM/BN non-divisor of E_M/E_N → per-cell ``if (row < M)``).
            # Treat as transparent for candidate scanning — the
            # cooperative load that ``_process_scope`` later emits will
            # be hoisted ABOVE the Cond at rewrite time so all threads
            # participate regardless of which cells the mask excludes.
            _collect_candidates(
                s,
                thread_axes,
                in_scope_axes,
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
            )
            continue
        if isinstance(s, SerialTile) and not s.is_reduce:
            _collect_candidates(
                s,
                thread_axes,
                (*in_scope_axes, s.axis),
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
            )
            continue
        if isinstance(s, (SerialTile, StridedTile)):
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
    for buf, items in loads_by_buf.items():
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _, _ in items):
            continue
        partitions_count: dict[tuple, int] = {}
        for load, _, _, _ in items:
            key = tuple(e.pretty() for e in load.index)
            partitions_count[key] = partitions_count.get(key, 0) + 1
        for load, reduce_axis, scope_axes, extra in items:
            key = tuple(e.pretty() for e in load.index)
            allow_no_fan_in = partitions_count[key] >= 2
            slab = _classify(
                load,
                thread_axes,
                reduce_axis,
                scope_axes,
                slab_cap=slab_cap,
                extra_candidates=extra,
                allow_no_fan_in=allow_no_fan_in,
            )
            if slab is None:
                continue
            found[buf] = found.get(buf, 0) + slab.n_bytes
            break


def _maybe_rewrite(
    body: Body,
    *,
    slab_cap: int,
    scope_budget: int,
    allowed_bufs: frozenset[str] | None = None,
    warp_size: int,
    atom_kind: str | None = None,
    bytes_per_elem: dict[str, int] | None = None,
) -> Body | None:
    idx, outer = single_tile(body)
    tt = parallel_tile_of(outer)
    if not tt.axes:
        if allowed_bufs is None:
            raise RuleSkipped("ThreadTile has no axes — no reuse to stage")
        return None

    n_thread = 1
    for ax in tt.axes:
        n_thread *= ax.extent.as_static()
    # Same WarpTile → CTA-thread expansion as `_candidate_buffers`. Keeps
    # the two preflight checks in lockstep so an MMA matmul body that
    # _candidate_buffers admits doesn't bail later inside _maybe_rewrite.
    if isinstance(tt, WarpTile):
        n_thread *= warp_size
    if n_thread <= warp_size:
        if allowed_bufs is None:
            raise RuleSkipped(f"warp-only cooperative tile (n_threads={n_thread} ≤ {warp_size}); register-resident, no smem stage")
        return None

    used_names: set[str] = set()
    block_axes = outer.axes if isinstance(outer, GridTile) else ()
    block_axis_names = frozenset(ax.name for ax in block_axes)
    is_cooperative = _tile_is_cooperative(tt)
    all_axes = tuple(block_axes) + tuple(tt.axes)
    new_tile_body = _process_scope(
        tt,
        tt.axes,
        all_axes,
        block_axis_names,
        used_names,
        slab_cap=slab_cap,
        scope_budget=scope_budget,
        allowed_bufs=allowed_bufs,
        is_cooperative=is_cooperative,
        atom_kind=atom_kind,
        bytes_per_elem=bytes_per_elem,
    )
    if new_tile_body == tt.body:
        if allowed_bufs is None:
            raise RuleSkipped("no Load qualifies for staging")
        return body
    rebuilt = replace_parallel_tile_body(outer, new_tile_body)
    return body[:idx] + (rebuilt,) + body[idx + 1 :]


def _peel_to_stage_inner(outer):
    """Descend single-stmt SerialTile chains until a stage_inner SerialTile
    is found. Returns ``None`` if the chain branches before reaching one.
    """
    cur = tuple(outer.body)
    while len(cur) == 1 and isinstance(cur[0], (SerialTile, StridedTile, RegisterTile)):
        s = cur[0]
        if isinstance(s, SerialTile) and s.kind == "stage_inner":
            return s
        cur = tuple(s.body)
    return None


def _process_scope(
    scope,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
    allowed_bufs: frozenset[str] | None = None,
    is_cooperative: bool = False,
    register_axes: tuple[Axis, ...] = (),
    atom_kind: str | None = None,
    bytes_per_elem: dict[str, int] | None = None,
) -> Body:
    """Walk scope.body; recurse into non-reduce free tiles; collect Loads
    from reduce tiles into per-buffer buckets. Per buffer, build a Source
    if all Loads agree on slab geometry. Admit Sources under budget;
    emit one ``StageBundle`` wrapping the contiguous range of consumer
    stmts that contain rewritten Loads. Source name + index rewrites are
    applied inside the bundle's body."""
    rewritten_inner: list[Stmt] = []
    # For each top-level stmt, the list of (load, reduce_axis, scope_axes, extra_cache_axes) bucketed by buf.
    # We keep a per-stmt map (by index in rewritten_inner) so we know which top-level stmts must end up
    # inside the wrapping Stage's body.
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]] = {}
    # Track which indices in rewritten_inner contain Loads that may need rewriting.
    stmt_contains_loads_idx: list[int] = []

    for s in scope.body:
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "serial_outer":
            inner_stage = _peel_to_stage_inner(s)
            if inner_stage is not None:
                scope_axes = (*in_scope_axes, s.axis, inner_stage.axis)
                extra = (s.axis,) + register_axes
                for stmt in inner_stage.body:
                    if isinstance(stmt, Load):
                        loads_by_buf.setdefault(stmt.input, []).append((stmt, inner_stage.axis, scope_axes, extra))
                rewritten_inner.append(s)
                stmt_contains_loads_idx.append(len(rewritten_inner) - 1)
                continue
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "stage_inner":
            pass  # fall through to the collection branch
        elif isinstance(s, RegisterTile):
            new_register_axes = register_axes + tuple(s.axes)
            new_body = _process_scope(
                s,
                thread_axes,
                tuple(in_scope_axes) + tuple(s.axes),
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=new_register_axes,
                atom_kind=atom_kind,
                bytes_per_elem=bytes_per_elem,
            )
            rewritten_inner.append(s.with_bodies((new_body,)))
            continue
        elif isinstance(s, AtomTile):
            # MMA fragment wrapper — walk transparently; ``register_axes``
            # stays untouched (atom dims are baked into the σ literal
            # multipliers and become ``AffineAddressing.block``).
            new_body = _process_scope(
                s,
                thread_axes,
                in_scope_axes,
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
                atom_kind=atom_kind,
                bytes_per_elem=bytes_per_elem,
            )
            rewritten_inner.append(s.with_bodies((new_body,)))
            continue
        elif isinstance(s, SerialTile) and not s.is_reduce:
            new_body = _process_scope(
                s,
                thread_axes,
                (*in_scope_axes, s.axis),
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
                atom_kind=atom_kind,
                bytes_per_elem=bytes_per_elem,
            )
            rewritten_inner.append(s.with_bodies((new_body,)))
            continue
        elif isinstance(s, Cond):
            # Masked-tile boundary guard from ``010_partition_loops``.
            # Descend transparently into the Cond's body so any reuse
            # inside admits Sources, then re-wrap the rewritten body in
            # the Cond. Lifting the cooperative-load K-pipeline above
            # the Cond (so TMA's elected issuer / cp.async's full-CTA
            # fan-out aren't gated out by the boundary mask) is a
            # downstream concern handled by
            # ``021_hoist_staged_loads_above_mask``.
            new_body = _process_scope(
                s,
                thread_axes,
                in_scope_axes,
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
                atom_kind=atom_kind,
                bytes_per_elem=bytes_per_elem,
            )
            rewritten_inner.append(Cond(cond=s.cond, body=Body(new_body), else_body=s.else_body))
            continue
        if isinstance(s, (SerialTile, StridedTile)):
            scope_axes = (*in_scope_axes, s.axis)
            had_load = False
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
                    had_load = True
            rewritten_inner.append(s)
            if had_load:
                stmt_contains_loads_idx.append(len(rewritten_inner) - 1)
            continue
        rewritten_inner.append(s)

    if allowed_bufs is not None:
        loads_by_buf = {b: items for b, items in loads_by_buf.items() if b in allowed_bufs}
    sources, name_rewrites = _build_sources(
        loads_by_buf,
        thread_axes,
        block_axis_names,
        used_names,
        slab_cap=slab_cap,
        scope_budget=scope_budget,
        atom_kind=atom_kind,
        bytes_per_elem=bytes_per_elem,
    )
    if not sources:
        return tuple(rewritten_inner)

    # Rewrite Loads inside the affected stmts to read from staged smem.
    rewritten = list(rewritten_inner)
    for i in stmt_contains_loads_idx:
        rewritten[i] = _rewrite_loads(rewritten[i], name_rewrites)

    # Wrap the contiguous range of stmt_contains_loads_idx in a single Stage.
    # If the affected stmts are not contiguous, wrap from the first to the last
    # (interleaved sibling stmts come along for the ride — they're typically
    # init/setup that's fine to live inside the staged scope).
    if not stmt_contains_loads_idx:
        return tuple(rewritten)
    lo, hi = stmt_contains_loads_idx[0], stmt_contains_loads_idx[-1]
    # Emit a single-policy SYNC bundle holding the multi-source transport
    # operands directly; the bundle owns the consumer body. Downstream passes
    # (030 hoist, 040 ring-buffer promotion, 050/060 TMA/async promotion)
    # operate on bundles.
    bundle = StageBundle(
        sources=tuple(sources),
        body=Body(tuple(rewritten[lo : hi + 1])),
        policy=StagePolicy.SYNC,
    )
    return tuple([*rewritten[:lo], bundle, *rewritten[hi + 1 :]])


def _build_sources(
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]],
    thread_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
    atom_kind: str | None = None,
    bytes_per_elem: dict[str, int] | None = None,
) -> tuple[list[Source], dict[str, tuple[str, tuple[Expr, ...]]]]:
    """Per buffer, partition Loads by index equality; per partition, derive
    slab geometry; admit Sources under budget. Returns (sources,
    name_rewrites) where name_rewrites maps original Load SSA names to
    (smem_buf_name, smem_index)."""
    sources: list[Source] = []
    name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    used_bytes = 0

    for buf, items in loads_by_buf.items():
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _, _ in items):
            continue

        partitions: list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...], list[Load]]] = []
        for load, reduce_axis, scope_axes, extra_cache_axes in items:
            for rep_load, _, _, _, members in partitions:
                if load.index == rep_load.index:
                    members.append(load)
                    break
            else:
                partitions.append((load, reduce_axis, scope_axes, extra_cache_axes, [load]))

        for rep_load, rep_reduce, rep_scope, rep_extra, members in partitions:
            allow_no_fan_in = len(members) >= 2
            slab = _classify(
                rep_load,
                thread_axes,
                rep_reduce,
                rep_scope,
                slab_cap=slab_cap,
                extra_candidates=rep_extra,
                allow_no_fan_in=allow_no_fan_in,
                atom_kind=atom_kind,
                bytes_per_elem=(bytes_per_elem or {}).get(rep_load.input, BYTES_PER_ELEM) if bytes_per_elem else BYTES_PER_ELEM,
            )
            if slab is None:
                continue
            if used_bytes + slab.n_bytes > scope_budget:
                continue
            smem_name = _gen_name(buf, used_names)
            # Pick addressing eagerly: template when ``_classify`` flagged a
            # collapsed-reshape (slab.template is the verbatim source-dim Exprs),
            # affine otherwise (dims = the slab's per-cache-axis source dims).
            # Source's ``__post_init__`` builds an identity-dims affine default
            # when ``addressing`` is omitted, but stamping it explicitly here
            # carries the real (possibly non-identity / collapsed) source-dim
            # mapping. ``slab.block`` is non-empty when ``_classify`` extracted
            # a non-unit literal coefficient on at least one cache var (MMA atom
            # σ is the M3-driving case); it folds into ``AffineAddressing.block``.
            addressing: AffineAddressing | TemplateAddressing
            if slab.template is not None:
                addressing = TemplateAddressing(exprs=slab.template)
            else:
                addressing = AffineAddressing(
                    dims=tuple(slab.slab_dims),
                    block=slab.block,
                )
            src = Source(
                name=smem_name,
                buf=buf,
                cache_axes=slab.cache_axes,
                origin=slab.origin,
                pad=(),
                addressing=addressing,
            )
            sources.append(src)
            used_bytes += slab.n_bytes
            smem_index = tuple(Var(ax.name) for ax in slab.cache_axes)
            for load in members:
                name_rewrites[load.name] = (smem_name, smem_index)
    return sources, name_rewrites


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
    extra_candidates: tuple[Axis, ...] = (),
    allow_no_fan_in: bool = False,
    atom_kind: str | None = None,
    bytes_per_elem: int = BYTES_PER_ELEM,
) -> _Slab | None:
    candidates = (*thread_axes, reduce_axis, *extra_candidates)
    ctx = SimplifyCtx({ax.name: Interval(0, ax.extent.as_static() - 1) for ax in scope_axes if ax.extent.is_static})
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

    if not var_to_dim:
        return None
    if len(var_to_dim) == len(candidates) and not allow_no_fan_in:
        return None

    cache_axes_unsorted = tuple(ax for ax in candidates if ax.name in var_to_dim)
    cache_axes = tuple(sorted(cache_axes_unsorted, key=lambda ax: var_to_dim[ax.name]))
    slab_dims = tuple(var_to_dim[ax.name] for ax in cache_axes)
    # A symbolic cache extent (e.g. seq_len) makes the slab size unbounded
    # at compile time. We can't compare against ``slab_cap``, and a worst-case
    # bound would force-disable staging on every symbolic load anyway —
    # so skip the candidate. Drops the smem-caching optimization on
    # seq_len-bearing loads but the load still works via direct global access.
    if any(not ax.extent.is_static for ax in cache_axes):
        return None
    # Per-buffer dtype size from ``parent_op.inputs`` plumbed through
    # the rewrite chain — fp16 was previously over-counted 2× because
    # the slab-cap check assumed fp32 (BYTES_PER_ELEM=4). For an MMA
    # matmul with K=2048 and a 64×64 block-scaled A slab, the actual
    # fp16 footprint is 8 KB; the legacy check rejected at 16 KB and
    # locked out the staged variant on shapes where the picker would
    # otherwise have admitted it.
    n_bytes = bytes_per_elem
    for ax in cache_axes:
        n_bytes *= ax.extent.as_static()
    if n_bytes > slab_cap:
        return None

    # Per-source-dim composite check: a Source is AffineAddressing when
    # each source dim ``d`` is reached by a clean composite of its cache
    # axes — i.e. the cache axes mapping to ``d`` form a positional
    # product matching
    # ``load.index[d] = origin[d] + ax_0·(e_1·e_2·…) + ax_1·(e_2·…) + … + ax_{k-1}``,
    # where the cache axes are ordered most-to-least significant (leftmost
    # mapping to ``d`` carries the largest stride). Admits two patterns:
    #
    # - Single-axis-per-dim with stride 1 (the legacy matmul case): one
    #   cache axis maps to ``d``, composite collapses to ``ax_0·1``.
    # - Multi-axis-per-dim with composite strides (the
    #   ``BN_thread × FN_register`` matmul-N case): cache axes
    #   ``(a_thread, a_reg)`` mapping to N reconstruct
    #   ``load.index[N] = origin + a_thread·FN + a_reg`` — composite
    #   stride ``FN`` for the thread axis, ``1`` for the register axis.
    #
    # The matching source-index reconstruction lives in
    # ``ir.tile.ir.affine_decode_per_dim`` and is shared across every
    # post-staging consumer (``_stage_expand``,
    # ``025_unify_sibling_stages._reconstruct_global_index``,
    # ``_source_decl_line``) — together they guarantee the composite
    # stride round-trips through smem-stage → revert-to-gmem → cuda
    # emission. The legacy ``DEPLODOCK_AFFINE_COLLAPSE`` opt-in gate
    # was removed: the per-axis unit-stride check it defaulted to was
    # strictly less powerful (rejected every multi-axis case as
    # template), and the SDPA-style false-positive that motivated the
    # gate was an unrelated bug in the unify-pass revert path
    # (overwriting per-dim entries instead of composite-decoding).
    # Two-tier affine recognition:
    #
    # 1. **Legacy coef-1 composite** (block=(1,…,1)): the σ output reduces
    #    exactly to ``origin[d] + ax_0·(e_1·e_2·…) + … + ax_{k-1}·1``. Covers
    #    scalar matmul / RMSNorm / softmax; the byte-clean gate for M2.
    # 2. **Block-stamped composite**: enabled only when ``atom_kind`` is
    #    set on the parent TileOp (MMA fragment factorization path). The σ
    #    output reduces to ``origin[d] + ax_0·(e_1·b_1·…·b_0) + … +
    #    ax_{k-1}·b_{k-1}`` where ``b_i`` is the per-axis literal stride
    #    factor (e.g. ``atom_M`` = 16). The slab grows by ``b_i`` per axis
    #    so the mma.sync fragment load can read a full ``atom_M × atom_K`` cell.
    #
    # For scalar paths, tier 2 is skipped — a non-1 coef on the cache var
    # (typically from a register-axis sitting between origin and cache,
    # e.g. ``a3·4 + a5`` where a5 is a per-thread var) means the slab
    # would need extra positions filled by some axis that ISN'T part of
    # the cooperative iteration. That's exactly what TemplateAddressing
    # is for; admitting it as affine-with-block would mis-size the slab
    # and let the cooperative load under-fill it.
    needs_template = False
    block_per_axis: dict[str, int] = {}
    for d in sorted(set(var_to_dim.values())):
        axes_for_dim = tuple(ax for ax in cache_axes if var_to_dim[ax.name] == d)
        other_zeros = {n: Literal(0, "int") for n in candidate_names if var_to_dim.get(n) != d}
        actual = sorted(t.pretty() for t in _flatten_add(Sigma(other_zeros).reduce(load.index[d], ctx)))
        # Tier 1: try coef-1 composite first. Same shape as the pre-M3
        # check; admits scalar matmul byte-identically.
        composite_unit: Expr = Literal(0, "int")
        suffix_product = 1
        for ax in reversed(axes_for_dim):
            term: Expr = Var(ax.name) if suffix_product == 1 else BinaryExpr("*", Var(ax.name), Literal(suffix_product, "int"))
            composite_unit = (
                term if isinstance(composite_unit, Literal) and composite_unit.value == 0 else BinaryExpr("+", composite_unit, term)
            )
            suffix_product *= ax.extent.as_static()
        expected_unit = sorted(t.pretty() for t in _flatten_add((origin[d] + composite_unit).simplify(ctx)))
        if actual == expected_unit:
            continue
        # Tier 2: try block-stamped composite, only under MMA. Per-
        # axis coef probe: zero every cache var EXCEPT the one we're
        # probing, σ-reduce, pluck the literal multiplier on ``Var(ax)``
        # via ``_extract_var_coef``.
        if atom_kind is None:
            needs_template = True
            break
        axis_coefs: dict[str, int] = {}
        for ax in axes_for_dim:
            probe_zeros = {n: Literal(0, "int") for n in candidate_names if n != ax.name}
            coef = _extract_var_coef(Sigma(probe_zeros).reduce(load.index[d], ctx), ax.name)
            if coef is None or coef < 1:
                needs_template = True
                break
            axis_coefs[ax.name] = coef
        if needs_template:
            break
        # Solve for per-axis block: walk right-to-left, each axis's σ coef
        # equals ``suffix_product_so_far × block_ax``.
        derived_block: dict[str, int] = {}
        suffix_product = 1
        for ax in reversed(axes_for_dim):
            coef = axis_coefs[ax.name]
            if suffix_product == 0 or coef % suffix_product != 0:
                needs_template = True
                break
            block_ax = coef // suffix_product
            if block_ax < 1:
                needs_template = True
                break
            derived_block[ax.name] = block_ax
            suffix_product *= ax.extent.as_static() * block_ax
        if needs_template:
            break
        composite_blocked: Expr = Literal(0, "int")
        suffix_product = 1
        for ax in reversed(axes_for_dim):
            stride = suffix_product * derived_block[ax.name]
            term = Var(ax.name) if stride == 1 else BinaryExpr("*", Var(ax.name), Literal(stride, "int"))
            composite_blocked = (
                term
                if isinstance(composite_blocked, Literal) and composite_blocked.value == 0
                else BinaryExpr("+", composite_blocked, term)
            )
            suffix_product *= ax.extent.as_static() * derived_block[ax.name]
        expected_blocked = sorted(t.pretty() for t in _flatten_add((origin[d] + composite_blocked).simplify(ctx)))
        if actual != expected_blocked:
            needs_template = True
            break
        block_per_axis.update(derived_block)
    template = tuple(load.index) if needs_template else None
    # Build the slab.block tuple aligned to cache_axes ordering. Default to
    # ``()`` when every derived block is 1 — keeps the affine path byte-
    # identical to pre-M3 for scalar matmul.
    if template is None and block_per_axis:
        block_tuple = tuple(block_per_axis.get(ax.name, 1) for ax in cache_axes)
        if all(b == 1 for b in block_tuple):
            block_tuple = ()
    else:
        block_tuple = ()
    # Re-check the budget against the block-scaled slab size: the σ stride
    # bakes a per-axis multiplier (atom_M = 16 for MMA m16n16k16) that
    # multiplies the actual smem footprint. The pre-block ``n_bytes``
    # check above passes anything that fits under ``slab_cap`` ignoring
    # the multiplier; a 16×16 MMA slab grows 256× and would otherwise
    # silently overflow.
    if block_tuple:
        block_product = 1
        for b in block_tuple:
            block_product *= b
        n_bytes *= block_product
        if n_bytes > slab_cap:
            return None

    return _Slab(
        origin=origin,
        cache_axes=cache_axes,
        slab_dims=slab_dims,
        template=template,
        n_bytes=n_bytes,
        block=block_tuple,
    )


def _load_free_vars(load: Load) -> frozenset[str]:
    out: set[str] = set()
    for e in load.index:
        out |= e.free_vars()
    return frozenset(out)


def _flatten_add(e: Expr) -> list[Expr]:
    if isinstance(e, BinaryExpr) and e.op == "+":
        return _flatten_add(e.left) + _flatten_add(e.right)
    return [e]


def _extract_var_coef(sig: Expr, var_name: str) -> int | None:
    """Return the literal multiplier on ``Var(var_name)`` in a flat-affine
    σ expression, or ``None`` if it doesn't appear / has an unrecognized
    shape.

    Walks the additive terms of ``sig``. For each term:

    - ``Var(var_name)`` → coef contribution 1.
    - ``Var(var_name) * Literal(c)`` or ``Literal(c) * Var(var_name)`` →
      coef contribution ``c``.
    - Term doesn't reference ``var_name`` → ignored (it's part of the
      origin / outer-axis carry).
    - Anything else (``Var(var_name)`` nested in a non-affine form,
      e.g. divide / modulo / multi-var product) → return None.

    Sums the contributions across additive terms. M3's `_classify` calls
    this on the σ output with every other cache var pinned to 0; the
    expected shape is a single ``Var * Literal`` term plus an origin
    that may carry free outer-axis vars (e.g. the per-CTA K_o anchor).
    """
    total = 0
    found = False
    for term in _flatten_add(sig):
        if var_name not in term.free_vars():
            continue
        if isinstance(term, Var) and term.name == var_name:
            total += 1
            found = True
            continue
        if isinstance(term, BinaryExpr) and term.op == "*":
            left, right = term.left, term.right
            if isinstance(left, Var) and left.name == var_name and isinstance(right, Literal) and isinstance(right.value, int):
                total += right.value
                found = True
                continue
            if isinstance(right, Var) and right.name == var_name and isinstance(left, Literal) and isinstance(left.value, int):
                total += left.value
                found = True
                continue
        return None
    return total if found else None


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
