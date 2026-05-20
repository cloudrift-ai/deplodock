"""Register-tile cooperative-reduce kernels by replicating along nest levels.

One slot per tilable nest level — sibling loops at the same level
share a factor (one knob). Today two shapes produce slots:

- **Matmul** body (matmul-shape reduce ``Loop``): two slots, one per
  THREAD axis. M-axis (outer, smaller extent) gets ``FM``; N-axis
  (inner, larger extent) gets ``FN``. Splitting a THREAD axis by F
  shrinks the launch's thread count and lets each thread own F output
  cells with independent accumulators.
- **Cooperative reduce** body (reduce ``StridedLoop``s with literal
  step): one slot covering all reduce loops at the innermost level,
  shared factor ``FN``. Unrolling the strided loop by F breaks the
  serial dep chain on the running accumulator by giving each thread F
  independent partial sums that get folded after the loop. The same
  slot also unrolls the post-reduce normalize ``StridedLoop`` wrapper
  produced by ``005_cooperative_reduce`` Phase 3 (non-reduce, same axis
  name as a reduce loop) so the epilogue Loads / Writes pick up the
  same per-thread register tile — without this the normalize loop
  stays scalar and ``003_vectorize_loads`` / the upcoming vec-store
  pass have nothing to fold.

**Axis-aware replication.** For each slot, every stmt whose value
transitively depends on the slot's axis is replicated ``factor`` times
with σ-substituted indices and SSA names suffixed by the replica
index. Non-dependent stmts pass through. Block stmts recurse into
their bodies. Dependency analysis is one :meth:`Body.fold` with a
bound-axis filter; see :func:`replicate_along_axis`.

**Stages stay singleton across F.** Because ``stage_inputs`` runs
before this pass, the body already references staged smem via
cache-local Loads. Replication σ-substitutes the cache-axis Vars
inside those Loads so per-thread cells decode different cache-local
offsets into the same slab. Stage stmts themselves aren't replicated
(cache-axis names are excluded from their dependency contribution).

Idempotence: ``FN`` is the universal post-008 marker. Both paths stamp
it; on a second visit, the pass skips immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile
from deplodock.compiler.tuning import register_tile_shape

PATTERN = [Pattern("root", TileOp)]

# Per-nest-level factor choices. Each slot picks one; factors must
# divide the slot's extent.
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Factor for the next-outer tilable nest level")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Factor for the innermost tilable nest level")
# Cap on total per-thread replication (∏ factors). NVRTC compile time
# explodes on more-unrolled bodies; ``TileOp.validate`` is the
# second-line filter on post-tile thread count.
_MAX_CELLS_PER_THREAD = 128


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)

    # Idempotence: every variant stamps ``FN``, so its presence in
    # ``knobs`` is the post-008 marker. Without this gate, F=1 leaves
    # axes unchanged and ``register_tile_shape`` would re-fire on every
    # subsequent rule pass.
    if FN.name in root.op.knobs:
        raise RuleSkipped("register-tile already applied (FN set in knobs)")

    slots = _find_slots(tile)
    if not slots:
        raise RuleSkipped("no tilable slots in Tile body")

    heuristic = tuple(s.heuristic for s in slots) if all(s.heuristic is not None for s in slots) else None
    combos = _enumerate_combos(slots, heuristic=heuristic, max_product=_MAX_CELLS_PER_THREAD)
    if not combos:
        raise RuleSkipped("no viable factor combo")

    variants: list[TileOp] = []
    for combo in combos:
        new_tile = tile
        for slot, f in zip(slots, combo, strict=True):
            for site in slot.make_sites(f):
                new_tile = site.apply(new_tile)
        knobs = {slot.knob: f for slot, f in zip(slots, combo, strict=True)}
        variants.append(TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=root.op.name, knobs=knobs))
    if len(variants) == 1:
        return variants[0]
    return variants


@dataclass(frozen=True)
class _Site:
    """One axis-replication site. ``apply(tile)`` returns the rewritten
    Tile (wrapper change + body replication + any post stmts)."""

    axis: str
    factor: int
    apply: Callable[[Tile], Tile]


@dataclass(frozen=True)
class _Slot:
    """One tilable nest level — a slot in the factor cross-product.
    Sibling loops at the same level share this slot (one ``knob``, one
    factor across them); each level is a distinct slot.

    ``label`` / ``extent`` drive divisibility; ``choices`` is the
    per-slot factor search space (every choice must divide ``extent``);
    ``heuristic`` is an optional cap-exempt pre-pick. ``make_sites(f)``
    returns one ``_Site`` per loop at this level."""

    knob: str
    label: str
    extent: int
    choices: tuple[int, ...]
    heuristic: int | None
    make_sites: Callable[[int], list[_Site]]


def _split_axis(axes: tuple[BoundAxis, ...], target: str, factor: int) -> tuple[tuple[BoundAxis, ...], Axis]:
    """Replace ``BoundAxis(target:E, THREAD)`` with ``BoundAxis(target_o:E/F, THREAD)``.
    Returns (new_axes, outer_axis)."""
    new_axes: list[BoundAxis] = []
    outer: Axis | None = None
    for ba in axes:
        if ba.axis.name == target:
            ext = int(ba.axis.extent)
            outer = Axis(f"{target}_o", ext // factor)
            new_axes.append(BoundAxis(axis=outer, bind=BIND_THREAD))
        else:
            new_axes.append(ba)
    assert outer is not None
    return tuple(new_axes), outer


def _thread_split_site(axis_name: str, factor: int) -> _Site:
    """Split a THREAD axis by ``factor``; replicate the body with
    ``σ: axis → axis_o * factor + i``."""

    def apply(tile: Tile) -> Tile:
        new_axes, axis_o = _split_axis(tile.axes, axis_name, factor)
        body = replicate_along_axis(tile.body, axis_name, factor, _sigma_split(axis_name, axis_o.name, factor))
        return Tile(axes=new_axes, body=body)

    return _Site(axis=axis_name, factor=factor, apply=apply)


def _normalize_unroll_site(axis_name: str, factor: int) -> _Site:
    """Widen the post-reduce *normalize* ``StridedLoop`` (non-reduce
    wrapper produced by ``005_cooperative_reduce`` Phase 3) on
    ``axis_name`` by ``factor``; replicate its body with
    ``σ: axis → axis + i * step``. No accumulator-fold chain: this
    wrapper has no ``Accum`` — its body is Load / Assign / Write that
    computes and emits one normalized output cell per replica."""

    def apply(tile: Tile) -> Tile:
        new_body: list[Stmt] = []
        for s in tile.body:
            if isinstance(s, StridedLoop) and not s.is_reduce and s.axis.name == axis_name and isinstance(s.step, Literal):
                step = int(s.step.value)
                if int(s.axis.extent) % (step * factor) != 0:
                    new_body.append(s)
                    continue
                inner = replicate_along_axis(s.body, axis_name, factor, _sigma_offset(axis_name, step))
                new_body.append(StridedLoop(axis=s.axis, start=s.start, step=Literal(step * factor, "int"), body=inner, unroll=s.unroll))
            else:
                new_body.append(s)
        return Tile(axes=tile.axes, body=Body(new_body))

    return _Site(axis=axis_name, factor=factor, apply=apply)


def _reduce_unroll_site(axis_name: str, factor: int) -> _Site:
    """Widen the reduce ``StridedLoop`` on ``axis_name`` by ``factor``;
    replicate its body with ``σ: axis → axis + i * step``; fold
    ``acc_0..acc_{F-1}`` back into the original Accum name so the
    downstream Combine sees a single value."""

    def apply(tile: Tile) -> Tile:
        new_body: list[Stmt] = []
        for s in tile.body:
            if isinstance(s, StridedLoop) and s.is_reduce and s.axis.name == axis_name and isinstance(s.step, Literal):
                step = int(s.step.value)
                if int(s.axis.extent) % (step * factor) != 0:
                    new_body.append(s)
                    continue
                accums = [(a.name, a.op) for a in s.body if isinstance(a, Accum)]
                inner = replicate_along_axis(s.body, axis_name, factor, _sigma_offset(axis_name, step))
                new_body.append(StridedLoop(axis=s.axis, start=s.start, step=Literal(step * factor, "int"), body=inner, unroll=s.unroll))
                for orig_name, op in accums:
                    # Chain binary folds: acc_0 op acc_1 → t1; t1 op acc_2 → t2; … last → orig_name.
                    cur = f"{orig_name}_0"
                    for i in range(1, factor):
                        t_name = orig_name if i == factor - 1 else f"{orig_name}_fold{i}"
                        new_body.append(Assign(name=t_name, op=op, args=(cur, f"{orig_name}_{i}")))
                        cur = t_name
            else:
                new_body.append(s)
        return Tile(axes=tile.axes, body=Body(new_body))

    return _Site(axis=axis_name, factor=factor, apply=apply)


def _divisors(n: int, cap: int) -> tuple[int, ...]:
    """All divisors of ``n`` up to ``cap``, ascending."""
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _enumerate_combos(
    slots: list[_Slot],
    *,
    heuristic: tuple[int, ...] | None = None,
    max_product: int | None = None,
) -> list[tuple[int, ...]]:
    """Cross-product of per-slot ``choices``, filtered by divisibility
    (``slot.extent % f == 0``) and an optional cap on ∏factors.
    ``heuristic`` (when given) is emitted first cap-exempt so
    deterministic compiles pick option 0; the rest follow in fixed
    iteration order. All-1 combos are dropped as no-ops.
    """
    n = len(slots)
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []

    def _add(combo: tuple[int, ...], *, cap: bool = True) -> None:
        if combo in seen:
            return
        if all(f <= 1 for f in combo):
            return
        if cap and max_product is not None:
            prod = 1
            for f in combo:
                prod *= f
            if prod > max_product:
                return
        for slot, f in zip(slots, combo, strict=True):
            if slot.extent % f != 0:
                return
        seen.add(combo)
        ordered.append(combo)

    if heuristic is not None and len(heuristic) == n:
        _add(tuple(int(f) for f in heuristic), cap=False)

    def rec(prefix: tuple[int, ...]) -> None:
        if len(prefix) == n:
            _add(prefix)
            return
        for f in slots[len(prefix)].choices:
            rec(prefix + (f,))

    rec(())
    return ordered


def _find_slots(tile: Tile) -> list[_Slot]:
    """Emit one ``_Slot`` per tilable nest level. Matmul body → two
    slots (M-axis outer → ``FM``, N-axis inner → ``FN``), one site
    each. Cooperative-reduce body → one slot at the innermost level
    (``FN``), one site per reduce ``StridedLoop`` with shared factor.
    Returns ``[]`` when neither shape matches or no factor > 1 fits."""
    if any(is_matmul_reduce(s) for s in tile.body.iter() if isinstance(s, Loop)):
        heuristic = register_tile_shape(tile)
        if heuristic[0] <= 1 and heuristic[1] <= 1:
            return []
        thread_axes = [ba for ba in tile.axes if ba.bind == BIND_THREAD]
        if len(thread_axes) != 2:
            return []
        # Smaller-extent axis = M (outer level → FM); larger = N (inner level → FN).
        sorted_ba = sorted(thread_axes, key=lambda ba: int(ba.axis.extent))
        knob_names = (FM.name, FN.name)
        return [
            _Slot(
                knob=knob_names[i],
                label=ba.axis.name,
                extent=int(ba.axis.extent),
                choices=_TUNE_F_CHOICES,
                heuristic=int(h),
                make_sites=(lambda f, name=ba.axis.name: [_thread_split_site(name, f)]),
            )
            for i, (ba, h) in enumerate(zip(sorted_ba, heuristic, strict=True))
        ]

    from math import gcd  # noqa: PLC0415

    reduce_loops = [s for s in tile.body if isinstance(s, StridedLoop) and s.is_reduce and isinstance(s.step, Literal)]
    if not reduce_loops:
        return []
    per_thread_iters = []
    for sl in reduce_loops:
        ext = int(sl.axis.extent)
        step = int(sl.step.value)
        if step <= 0 or ext % step != 0:
            return []
        per_thread_iters.append(ext // step)
    # The post-reduce normalize ``StridedLoop`` (005 Phase 3, non-reduce,
    # axis name aliased to a reduce axis) participates in the same slot:
    # unrolling it by the same F gives the epilogue Load/Write chain F
    # consecutive copies that 009b can permute and 003_vectorize_loads /
    # 005_vectorize_stores can fold. Include its per-thread iter count
    # in the gcd so F always divides cleanly.
    reduce_axis_set = {sl.axis.name for sl in reduce_loops}
    normalize_loops = [
        s
        for s in tile.body
        if isinstance(s, StridedLoop) and not s.is_reduce and isinstance(s.step, Literal) and s.axis.name in reduce_axis_set
    ]
    for sl in normalize_loops:
        ext = int(sl.axis.extent)
        step = int(sl.step.value)
        if step <= 0 or ext % step != 0:
            return []
        per_thread_iters.append(ext // step)
    # gcd of per-loop iter counts → factor dividing it divides them all.
    slot_extent = per_thread_iters[0]
    for it in per_thread_iters[1:]:
        slot_extent = gcd(slot_extent, it)
    if slot_extent < 2:
        return []
    reduce_axis_names = tuple(sl.axis.name for sl in reduce_loops)
    normalize_axis_names = tuple(sl.axis.name for sl in normalize_loops)
    # All divisors of ``slot_extent`` (not just powers of two) — Qwen-style
    # hidden dims like 3584 give per-thread iters = 14, whose only useful
    # factors are {2, 7, 14}; restricting to powers of two would leave
    # only FN=2 on the table.
    #
    # ``heuristic`` = the largest valid factor (most aggressive unroll —
    # best ILP for the accumulator chain) so deterministic compiles pick
    # FN=14 over FN=2 on rmsnorm.qwen.s32 (saves ~33%). Autotune sweeps
    # the smaller factors after.
    divisors = _divisors(slot_extent, _MAX_CELLS_PER_THREAD)
    return [
        _Slot(
            knob=FN.name,
            label="reduce",
            extent=slot_extent,
            choices=divisors,
            heuristic=max(divisors),
            make_sites=(
                lambda f, ra=reduce_axis_names, na=normalize_axis_names: [
                    *(_reduce_unroll_site(a, f) for a in ra),
                    *(_normalize_unroll_site(a, f) for a in na),
                ]
            ),
        )
    ]


def replicate_along_axis(body: Body, axis: str, factor: int, sigma_for: Callable[[int], Sigma]) -> Body:
    """F× replicate every stmt whose value transitively depends on
    ``axis``. Each such stmt is emitted ``factor`` times with σ given
    by ``sigma_for(i)`` and SSA names suffixed ``_<i>``. Stmts that
    don't depend on ``axis`` pass through. Block stmts recurse into
    their bodies and rebuild via :meth:`Stmt.with_bodies`; a wrapper
    that shadows ``axis`` isn't itself replicated (the fold's
    bound-axis filter keeps shadowed references local).

    Dependency analysis is one :meth:`Body.fold` over the def-use DAG
    with bound-axis filtering. ``keep[name]`` records which SSA names
    must carry the suffix vs. pass through unchanged (Tile-input
    buffers, constants, axis-free producers)."""

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        # Stage cache-axis Vars are smem-local — they don't vary per replica.
        # Mark them bound here so Stages aren't tagged for replication; only
        # the consumer Loads (which σ-rewrite cache-axis Vars) multiply.
        local_bound = bound | frozenset(ax.name for ax in s.axes) if isinstance(s, Stage) else bound
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in local_bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in s.defines()}

    def rename_for(i: int):
        def _rename(name: str) -> str:
            return f"{name}_{i}" if keep.get(name, False) else name

        return _rename

    def go(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            nested = s.nested()
            if nested:
                out.append(s.with_bodies(tuple(go(child) for child in nested)))
            elif axis in deps.get(id(s), frozenset()):
                for i in range(factor):
                    out.append(s.rewrite(rename_for(i), sigma_for(i)))
            else:
                out.append(s)
        return Body(out)

    return go(body)


def _sigma_split(axis: str, axis_o: str, factor: int) -> Callable[[int], Sigma]:
    """σ for THREAD-axis split: ``axis → axis_o * factor + i``."""

    def _f(i: int) -> Sigma:
        return Sigma({axis: Var(axis_o) * Literal(factor, "int") + Literal(i, "int")})

    return _f


def _sigma_offset(axis: str, step: int) -> Callable[[int], Sigma]:
    """σ for reduction-axis unroll: ``axis → axis + i * step`` (identity for i=0)."""

    def _f(i: int) -> Sigma:
        if i == 0:
            return Sigma.IDENTITY
        return Sigma({axis: Var(axis) + Literal(i * step, "int")})

    return _f
