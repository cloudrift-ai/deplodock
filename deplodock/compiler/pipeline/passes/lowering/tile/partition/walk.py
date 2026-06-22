"""One nest walk in place of three regime template-matchers.

`walk_nest` walks the loop nest once and tags it algebraically — the free
(`PARALLEL`) outer chain plus the reduce (`REDUCE`) axes, keyed by each reduce
loop's `Loop.algebra_kind` (`MAP` / `MONOID` / `SEMIRING` / `TWISTED_MONOID`,
computed bottom-up from the carrier). It produces the regime skeleton the shared
`_assemble` emitter consumes, replacing the rigid whole-kernel envelope checks
the old `lift_*` functions performed. See
`plans/move-composer-axis-walk-scheduler.md`.

The walk is the *recognition front-end* only — it enumerates the legal per-axis
moves' inputs; the Fork tree (`tree.py`) still *picks* the schedule (tier,
split-K, knob ranges). Symbolic axes and `TWISTED_MONOID` carriers stay on the
legacy path (returned as `None`).

Phase 1 keeps the three regimes' envelopes byte-identical; later phases relax the
prologue/epilogue/multi-reduce checks so those shapes compose instead of falling
through to legacy.
"""

from __future__ import annotations

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Loop, Stmt, Write
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import (
    CoopReduceSkeleton,
    MatmulSkeleton,
    PointwiseSkeleton,
    _map_axis,
    _split_leading_non_loops,
)

Skeleton = PointwiseSkeleton | MatmulSkeleton | CoopReduceSkeleton


def _free_chain(body: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], list[Loop]]:
    """Split leading non-loop stmts, then walk the outer single-stmt chain of
    free (non-reduce) loops — the `PARALLEL` axes, outermost-first."""
    leading, rest = _split_leading_non_loops(body)
    chain: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    return leading, chain


def _all_static(chain: list[Loop]) -> bool:
    return all(lp.axis.extent.is_static for lp in chain)


def walk_nest(loop_op: LoopOp, *, warp_size: int = 32) -> Skeleton | None:
    """Tag the nest and return the regime skeleton, or `None` for shapes the
    composer leaves to the legacy planner (symbolic axes, `TWISTED_MONOID`
    flash, mixed/multi-different-axis reduces, chain-less bodies)."""
    body = tuple(loop_op.body)
    reduce_loops = [lp for lp in loop_op.body.iter_of_type(Loop) if lp.is_reduce]
    algebras = {lp.algebra_kind for lp in reduce_loops}
    if AlgebraKind.TWISTED_MONOID in algebras:
        return None  # flash / coupled carry — not yet an axis transform
    leading, chain = _free_chain(body)
    if not chain:
        return None
    inner_loop = chain[-1]
    inner_body = tuple(inner_loop.body)

    # PARALLEL-only — pointwise.
    if not reduce_loops:
        return PointwiseSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]) if len(chain) >= 2 else None,
            extra_outer=tuple(chain[:-2]) if len(chain) >= 2 else tuple(chain[:-1]),
            inner_body=inner_body,
            leading=leading,
        )

    if not _all_static(chain):
        return None
    k_loop = reduce_loops[0]
    if not k_loop.axis.extent.is_static:
        return None

    if algebras == {AlgebraKind.SEMIRING}:
        # Matmul — canonical envelope (multi-accumulator + fused prologue are
        # later phases, so keep the single-reduce, no-extra-stmts shape).
        if len(reduce_loops) != 1 or len(chain) < 2:
            return None
        loops_in = [s for s in inner_body if isinstance(s, Loop)]
        if loops_in != [k_loop] or not any(isinstance(s, Write) for s in inner_body):
            return None
        if any(not isinstance(s, (Loop, Write)) for s in inner_body):
            return None  # fused prologue — later phase
        return MatmulSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]),
            extra_outer=tuple(chain[:-2]),
            k_loop=k_loop,
            k_name=k_loop.axis.name,
            k_extent=k_loop.axis.extent.as_static(),
            inner_body=inner_body,
            leading=leading,
        )

    if algebras == {AlgebraKind.MONOID}:
        # Cooperative reduce. The reduces (one for sum/max; two for softmax) and
        # any second-pass map loop all iterate the SAME K axis — `upstream
        # unify_sibling_reduce_axes` collapses the sibling reduce + map axes to
        # one canonical name, and `_replace_k_coop` rewrites every K-named loop.
        # So the epilogue MAP scalar stmts (RMSNorm rsqrt, softmax log) and the
        # normalize map loop just ride the row tile — no rigid envelope.
        k_extent = k_loop.axis.extent.as_static()
        if k_extent < warp_size:
            return None
        if {lp.axis.extent.as_static() for lp in reduce_loops} != {k_extent}:
            return None  # reduces over different-extent K axes — multi-axis is a later phase
        # Every body loop must be a K-extent loop (a reduce or the second-pass
        # map) — those are what get cooperatively split. A loop of any other
        # extent is an unexpected shape; leave it to legacy.
        body_loops = [s for s in inner_body if isinstance(s, Loop)]
        if any(not lp.axis.extent.is_static or lp.axis.extent.as_static() != k_extent for lp in body_loops):
            return None
        return CoopReduceSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]) if len(chain) >= 2 else None,
            extra_outer=tuple(chain[:-2]) if len(chain) >= 2 else tuple(chain[:-1]),
            k_loop=k_loop,
            k_name=k_loop.axis.name,
            k_extent=k_extent,
            inner_body=inner_body,
            leading=leading,
            target_names=frozenset(lp.axis.name for lp in body_loops),
        )

    return None


# Thin type-filtered wrappers — the regime-specific entry points (tests +
# back-compat). The single recognizer is `walk_nest`.
def lift_pointwise(loop_op: LoopOp) -> PointwiseSkeleton | None:
    nest = walk_nest(loop_op)
    return nest if isinstance(nest, PointwiseSkeleton) else None


def lift_matmul(loop_op: LoopOp) -> MatmulSkeleton | None:
    nest = walk_nest(loop_op)
    return nest if isinstance(nest, MatmulSkeleton) else None


def lift_coop_reduce(loop_op: LoopOp, *, warp_size: int = 32) -> CoopReduceSkeleton | None:
    nest = walk_nest(loop_op, warp_size=warp_size)
    return nest if isinstance(nest, CoopReduceSkeleton) else None
