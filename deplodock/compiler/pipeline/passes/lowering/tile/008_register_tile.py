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
from deplodock.compiler.ir.stmt import Loop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    MAX_CELLS_PER_THREAD as _MAX_CELLS_PER_THREAD_SHARED,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    TUNE_F_CHOICES as _TUNE_F_CHOICES_SHARED,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    is_matmul_reduce,
    replicate_along_axis,
    single_tile,
)
from deplodock.compiler.tuning import BodyInfo, register_tile_shape

PATTERN = [Pattern("root", TileOp)]


def _logical_output_extents(tile: Tile) -> tuple[int, ...]:
    """Walk ``tile.axes`` folding adjacent BLOCK-then-THREAD pairs into
    a single extent. Returns sorted descending."""
    from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD  # noqa: PLC0415

    extents: list[int] = []
    i = 0
    while i < len(tile.axes):
        ba = tile.axes[i]
        ext = int(ba.axis.extent)
        if ba.bind == BIND_BLOCK and i + 1 < len(tile.axes) and tile.axes[i + 1].bind == BIND_THREAD:
            extents.append(ext * int(tile.axes[i + 1].axis.extent))
            i += 2
            continue
        extents.append(ext)
        i += 1
    return tuple(sorted(extents, reverse=True))


# Per-nest-level factor choices. Each slot picks one; factors must
# divide the slot's extent. Single source of truth in ``_helpers.py``
# so the planner-driven fork (``000_partition_planner``) and this
# legacy fork stay in sync.
_TUNE_F_CHOICES = _TUNE_F_CHOICES_SHARED

FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Factor for the next-outer tilable nest level")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Factor for the innermost tilable nest level")
# Cap on total per-thread replication (∏ factors). NVRTC compile time
# explodes on more-unrolled bodies; ``TileOp.validate`` is the
# second-line filter on post-tile thread count.
_MAX_CELLS_PER_THREAD = _MAX_CELLS_PER_THREAD_SHARED


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)

    # Idempotence: every variant stamps ``FN``, so its presence in
    # ``knobs`` is the post-008 marker. Without this gate, F=1 leaves
    # axes unchanged and ``register_tile_shape`` would re-fire on every
    # subsequent rule pass.
    if FN.name in root.op.knobs:
        # Idempotence marker. Also fires after ``006a_register_tile_planned``
        # has handled the planner-driven path pre-staging — this rule is
        # the legacy split-and-replicate fork for non-planner kernels.
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
    slots (M-axis outer → ``FM``, N-axis inner → ``FN``), one site each.

    Returns ``[]`` when the body isn't matmul-shaped or no factor > 1
    fits. The cooperative-reduce slot (``StridedLoop`` unroll) has been
    removed alongside the cooperative-reduce pipeline.

    The matmul branch is the fallback path for kernels where the
    planner's pre-blockify ``register_tile_shape`` heuristic returned
    ``(1, 1)`` (small / fused matmuls — SDPA's 32×64 inner matmul);
    those kernels arrive here without an FN stamp and this rule does
    the post-blockify register-tile split."""
    if not any(is_matmul_reduce(s) for s in tile.body.iter() if isinstance(s, Loop)):
        return []
    body_info = BodyInfo.of(tile.body)
    output_extents = _logical_output_extents(tile)
    thread_axes = [ba for ba in tile.axes if ba.bind == BIND_THREAD]
    thread_extents = tuple(int(ba.axis.extent) for ba in thread_axes)
    heuristic = register_tile_shape(output_extents, thread_extents, body_info)
    if heuristic[0] <= 1 and heuristic[1] <= 1:
        return []
    if len(thread_axes) != 2:
        return []
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


def _sigma_split(axis: str, axis_o: str, factor: int) -> Callable[[int], Sigma]:
    """σ for THREAD-axis split: ``axis → axis_o * factor + i``."""

    def _f(i: int) -> Sigma:
        return Sigma({axis: Var(axis_o) * Literal(factor, "int") + Literal(i, "int")})

    return _f
