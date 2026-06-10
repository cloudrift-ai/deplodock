"""Per-source ``+1`` smem padding to break bank conflicts on body reads.

After ``020_stage_inputs`` lays out each ``Source`` as one smem slab, body
Loads read it through thread-decoded coords. When the slab's per-thread-
axis stride is a multiple of 32 floats (the bank count × 4 bytes / 4
bytes per fp32), every lane in a warp hits the same bank — a 32-way
conflict that serializes 32 LDS instructions into 32 single-bank reads.

Per-source ``+1`` padding on one cache extent shifts every higher-stride
row by one float and breaks the alias. ``pad`` lives on each
:class:`Source`; the materializer reads it via :attr:`Source.alloc_extents`
when sizing the ``Smem`` decl, so body Loads' index expressions stay
unchanged and only the smem stride contract shifts.

**Autotune fork.** ``PAD_SMEM`` is a BOOL knob with both polarities
emitted whenever any source is pad-eligible: ``True`` applies the
greedy ``+1`` pad fix where it drives ``max_way`` to 1; ``False``
leaves every source unpadded. The autotuner picks the winner from
measured perf — padding costs smem footprint (each ``+1`` row × the
slab stride below) and a 1-byte ``ld.shared`` misalignment penalty
that sometimes outweighs the bank-conflict reduction on small slabs.
Tiles whose validate() check rejects the padded variant (smem
overflow) get dropped silently by the search.

**Per-source pad selection (PAD_SMEM=True).**

1. Descend the Tile body until a ``ThreadTile`` is in scope (its axes
   drive lane-decode for bank analysis).
2. For each ``BufferedStage`` / ``AsyncBufferedStage`` (``TmaBufferedStage``
   skipped — TMA box copies + hardware swizzle don't tolerate ``+1`` pad
   and the class asserts pad-empty), walk every ``Source``.
3. Per source, collect body Loads reading its smem name. Compute the
   worst-case ``max_way`` across them at the un-padded extents using
   :func:`lane_bank_distribution`. If already ≤ 1, skip.
4. Try ``+1`` pad candidates (1-dim then 2-dim, innermost-first). Pick
   the first that drives every Load's max-way to 1, or fall back to
   the best partial fix.

The innermost cache dim is padded only when the body Loads against this
source can't be vectorized by ``050_vectorize_loads``. ``+1`` on the
innermost adds 4 bytes to the next-outer stride (= 4 mod 16 in bytes),
which mis-aligns any LDS.128 emitted with an outer-dim offset. We detect
the vectorizable shape ahead of time: ≥ 2 consecutive Loads against the
same source sharing every outer-index expression and differing only on
the last index by exactly 1 per step. If any such run exists, the inner
dim is unsafe to pad and we restrict candidates to outer dims only.
Sources with disparate per-Load outer indices (e.g. matmul A-side where
the next-outer K-fragment dim is a per-load literal) are safe to inner-pad.

Idempotence: stamps ``PAD_SMEM`` on every emitted variant so re-entry
self-skips.
"""

from __future__ import annotations

import logging
from dataclasses import replace as _replace

from deplodock.compiler.context import Context
from deplodock.compiler.diagnostics.bank_conflicts import lane_bank_distribution
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Source,
    StageBundle,
    StagePolicy,
    ThreadTile,
    TileOp,
    WarpTile,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

PAD_SMEM = Knob(
    "PAD_SMEM",
    KnobType.BOOL,
    hints=(False, True),
    help=(
        "Apply per-source ``+1`` smem pad to break bank conflicts on body Loads. "
        "Autotune fork — True trades smem footprint for fewer LDS replays; False "
        "leaves the slab dense. Variants are only emitted when at least one Source "
        "actually benefits."
    ),
    off=False,
)


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    if PAD_SMEM.name in root.op.knobs:
        raise RuleSkipped("pad_smem already applied (idempotence via knob)")

    pad_plan = _plan_pad(root.op.body)
    if pad_plan is None:
        # No fixable bank conflict — record the off decision (body unchanged) so
        # the realized config keeps a uniform knob set instead of leaving it absent.
        return [TileOp(body=root.op.body, name=root.op.name, knobs={**root.op.knobs, PAD_SMEM.name: False})]

    # Emission order: pad-on first as the greedy default (matches the
    # pre-refactor behavior where padding fired whenever it could).
    # ``PAD_SMEM.narrow`` filters to the env-pinned polarity if
    # ``DEPLODOCK_PAD_SMEM`` is set; otherwise both fire.
    polarities = PAD_SMEM.narrow((True, False))
    variants: list[TileOp] = []
    for polarity in polarities:
        if polarity:
            padded_body, applied = _apply_pad(root.op.body, pad_plan)
            if not applied:
                continue
            variants.append(TileOp(body=padded_body, name=root.op.name, knobs={**root.op.knobs, PAD_SMEM.name: True}))
        else:
            variants.append(TileOp(body=root.op.body, name=root.op.name, knobs={**root.op.knobs, PAD_SMEM.name: False}))
    if not variants:
        raise RuleSkipped("PAD_SMEM env pin produced no matching variants")
    return variants


def _plan_pad(body: Body) -> dict[int, dict[str, tuple[int, ...]]] | None:
    """First pass: walk the body and decide what pad each ``BufferedStage`` /
    ``AsyncBufferedStage`` source would receive under PAD_SMEM=True.

    Returns ``None`` if no source has a fixable conflict. Otherwise returns
    a dict keyed by ``id(stage)`` mapping source-name → pad tuple. Stages
    not in the dict pass through unchanged; sources not in a stage's inner
    dict pass through unchanged.
    """
    plan: dict[int, dict[str, tuple[int, ...]]] = {}
    _plan_walk(body, plan, thread_axes=())
    return plan or None


def _plan_walk(body: Body, plan: dict[int, dict[str, tuple[int, ...]]], *, thread_axes: tuple[Axis, ...]) -> None:
    for s in body:
        if isinstance(s, (ThreadTile, WarpTile)):
            _plan_walk(s.body, plan, thread_axes=s.axes)
            continue
        # Pad applies to BUFFERED / ASYNC bundles; SYNC has no buffer
        # rotation that needs the conflict break, TMA forbids padding.
        if isinstance(s, StageBundle) and s.policy in (StagePolicy.BUFFERED, StagePolicy.ASYNC):
            _plan_for_bundle(s, plan, thread_axes=thread_axes)
        for b in s.nested():
            _plan_walk(b, plan, thread_axes=thread_axes)


def _plan_for_bundle(bundle: StageBundle, plan: dict[int, dict[str, tuple[int, ...]]], *, thread_axes: tuple[Axis, ...]) -> None:
    if not thread_axes:
        return
    per_src: dict[str, tuple[int, ...]] = {}
    for src in bundle.sources:
        if src.pad and any(src.pad):
            continue
        # M7: a +1 pad on the inner cache axis breaks the mma.sync
        # ``ldmatrix`` 16-byte alignment, so block-stamped Sources
        # (MMA atom-strided σ) must skip padding entirely. A future
        # MMA-friendly swizzle (XOR-style row shuffle matching
        # ldmatrix.x4) is the right answer; until then, leave the
        # bundle un-padded.
        addressing = src.addressing
        if isinstance(addressing, AffineAddressing) and addressing.block:
            continue
        loads = [s for s in bundle.body.iter() if isinstance(s, Load) and s.input == src.name]
        if not loads:
            continue
        inner_safe = not _has_vectorizable_run(loads)
        pad = _pick_pad(loads, src, thread_axes=thread_axes, inner_safe=inner_safe)
        if pad is None:
            continue
        per_src[src.name] = pad
    if per_src:
        plan[id(bundle)] = per_src


def _has_vectorizable_run(loads: list[Load]) -> bool:
    """True iff ``loads`` (in source order) contain ≥ 2 consecutive Loads
    sharing every outer-index expression and differing in the last index
    by exactly 1 per step. This is the structural precondition for
    ``050_vectorize_loads`` to emit ``ld.shared.v4`` against this source —
    we use it as the gate for innermost-pad eligibility (a ``+1`` on the
    innermost would otherwise misalign the vectorized load)."""
    if len(loads) < 2:
        return False
    from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx  # noqa: PLC0415

    run = 1
    for prev, cur in zip(loads, loads[1:], strict=False):
        if len(prev.index) != len(cur.index) or len(prev.index) == 0:
            run = 1
            continue
        if any(p.pretty() != c.pretty() for p, c in zip(prev.index[:-1], cur.index[:-1], strict=True)):
            run = 1
            continue
        diff = BinaryExpr("-", cur.index[-1], prev.index[-1]).simplify(SimplifyCtx.empty())
        if not (isinstance(diff, Literal) and isinstance(diff.value, int) and diff.value == 1):
            run = 1
            continue
        run += 1
        if run >= 2:
            return True
    return False


def _apply_pad(body: Body, plan: dict[int, dict[str, tuple[int, ...]]]) -> tuple[Body, bool]:
    """Second pass: rebuild the body, applying the planned pads. Returns
    ``(new_body, applied)`` where ``applied`` is True iff any pad landed."""
    out: list[Stmt] = []
    applied = False
    for s in body:
        if isinstance(s, StageBundle) and id(s) in plan:
            per_src = plan[id(s)]
            new_sources: list[Source] = []
            for src in s.sources:
                if src.name in per_src:
                    new_sources.append(src.with_pad(per_src[src.name]))
                    applied = True
                else:
                    new_sources.append(src)
            s = _replace(s, sources=tuple(new_sources))
        nested = s.nested()
        if nested:
            new_bodies: list[Body] = []
            sub_applied = False
            for b in nested:
                nb, ca = _apply_pad(b, plan)
                new_bodies.append(nb)
                sub_applied = sub_applied or ca
            if sub_applied:
                s = s.with_bodies(tuple(new_bodies))
                applied = True
        out.append(s)
    return Body(tuple(out)), applied


def _pick_pad(loads: list[Load], src: Source, *, thread_axes: tuple[Axis, ...], inner_safe: bool) -> tuple[int, ...] | None:
    base_extents = tuple(ax.extent.as_static() for ax in src.cache_axes)
    if not base_extents:
        return None
    base_conflict = _max_conflict(loads, base_extents, thread_axes)
    if base_conflict is None or base_conflict <= 1:
        return None

    n = len(base_extents)
    # The innermost dim is included in the candidate set only when no body
    # Load forms a vectorizable run (see _has_vectorizable_run) — otherwise
    # the ``+1`` there mis-aligns nvcc's ``ld.shared.v4`` and crashes at
    # runtime with ``CUDA_ERROR_MISALIGNED_ADDRESS``.
    if n == 1:
        if not inner_safe:
            return None
        candidates: list[tuple[int, ...]] = [(1,)]
    else:
        candidates = []
        inner_dims = range(n - 1, -1, -1) if inner_safe else range(n - 2, -1, -1)
        for dim in inner_dims:
            pad = [0] * n
            pad[dim] = 1
            candidates.append(tuple(pad))
        pair_max = n if inner_safe else n - 1
        for d1 in range(pair_max):
            for d2 in range(d1 + 1, pair_max):
                pad = [0] * n
                pad[d1] = 1
                pad[d2] = 1
                candidates.append(tuple(pad))

    best_pad: tuple[int, ...] | None = None
    best_c = base_conflict
    for pad in candidates:
        padded = tuple(e + p for e, p in zip(base_extents, pad, strict=True))
        c = _max_conflict(loads, padded, thread_axes)
        if c is None:
            continue
        if c <= 1:
            return pad
        if c < best_c:
            best_c = c
            best_pad = pad
    return best_pad


def _max_conflict(loads: list[Load], extents: tuple[int, ...], thread_axes: tuple[Axis, ...]) -> int | None:
    """Worst-case ``max_way`` across body ``loads`` at the hypothetical
    ``extents`` — i.e. evaluating row-major strides as if the slab were
    sized to ``extents``. Loads in a wrap-body ``BufferedStage`` carry a
    leading phase dim prepended by ``040_use_ring_buffers``; drop it before
    the rank check."""
    worst = 1
    for ld in loads:
        cache_idx = ld.index[1:] if len(ld.index) == len(extents) + 1 else ld.index
        if len(cache_idx) != len(extents):
            return None
        dist = lane_bank_distribution(tuple(cache_idx), extents, thread_axes)
        if dist is None:
            return None
        worst = max(worst, dist.max_way)
    return worst
