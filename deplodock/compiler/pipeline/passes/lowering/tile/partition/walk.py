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
    FlashSkeleton,
    MatmulSkeleton,
    PointwiseSkeleton,
    _map_axis,
    _split_leading_non_loops,
)

Skeleton = PointwiseSkeleton | MatmulSkeleton | CoopReduceSkeleton | FlashSkeleton


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


def walk_nest(loop_op: LoopOp, *, warp_size: int = 32) -> Skeleton | None:
    """Tag the nest and return the regime skeleton, or `None` for shapes the
    composer leaves to the legacy planner (symbolic axes, `TWISTED_MONOID`
    flash, mixed/multi-different-axis reduces, chain-less bodies)."""
    body = tuple(loop_op.body)
    reduce_loops = [lp for lp in loop_op.body.iter_of_type(Loop) if lp.is_reduce]
    algebras = {lp.algebra_kind for lp in reduce_loops}
    leading, chain = _free_chain(body)
    if not chain:
        return None
    inner_loop = chain[-1]
    inner_body = tuple(inner_loop.body)

    if AlgebraKind.TWISTED_MONOID in algebras:
        # Fused flash attention (the FlashCombine streaming-softmax carrier).
        # Tile the free output axes; serial-transform the streaming KV reduce +
        # its nested QK^T reduce; the carriers render their own rescale. A
        # symbolic ``seq_len`` lands on the free q-rows axis (masked tile) AND the
        # streaming KV reduce (masked stream — `k_bounds`); the nested QK^T reduce
        # (head_dim) must stay static, since masked-K matmul is a separate tier.
        if len(chain) < 2:
            return None
        twisted = [lp for lp in reduce_loops if lp.algebra_kind == AlgebraKind.TWISTED_MONOID]
        others = [lp for lp in reduce_loops if lp.algebra_kind != AlgebraKind.TWISTED_MONOID]
        if any(not lp.axis.extent.is_static for lp in others):
            return None
        k_bounds = {lp.axis.name: lp.axis.extent.expr for lp in twisted if not lp.axis.extent.is_static}
        return FlashSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]),
            extra_outer=tuple(chain[:-2]),
            target_names=frozenset(lp.axis.name for lp in reduce_loops),
            inner_body=inner_body,
            leading=leading,
            k_bounds=k_bounds,
        )

    # PARALLEL-only — pointwise.
    if not reduce_loops:
        return PointwiseSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]) if len(chain) >= 2 else None,
            extra_outer=tuple(chain[:-2]) if len(chain) >= 2 else tuple(chain[:-1]),
            inner_body=inner_body,
            leading=leading,
        )

    # Symbolic *free* axes are fine — they tile as masked (ceil-div grid + a
    # `< extent` store guard), the same path pointwise uses. A symbolic *K*
    # (reduce) is handled only on the cooperative (MONOID) path below (masked-K
    # fill with the carrier identity); the matmul path requires it static.
    k_loop = reduce_loops[0]

    if algebras == {AlgebraKind.SEMIRING}:
        # Matmul. A MAP epilogue after the reduce — the QK^T `* 1/√d` scale, a
        # `matmul_add` residual — rides the output tile, so allow non-Loop stmts.
        # Multiple same-K contractions (a gated MLP's gate + up matmuls, with a
        # fused epilogue between/after) also compose: each is its own reduce over
        # the shared K, split together by extent (scalar tier — `is_atom_eligible`
        # declines a multi-accum body, so warp gates off in build_matmul_tree).
        # A leading Load (the `matmul_add` residual / a bias operand the epilogue
        # adds) may precede the reduce — what we reject is a MAP-loop PROLOGUE
        # (a non-reduce loop feeding the contraction), so require every body loop
        # to BE a reduce over the (static) K extent.
        k_dim = k_loop.axis.extent
        if not k_dim.is_static or len(chain) < 2:
            return None
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None  # reduces over different-extent K axes
        body_loops = [s for s in inner_body if isinstance(s, Loop)]
        if any(lp.axis.extent != k_dim or not lp.is_reduce for lp in body_loops):
            return None  # a non-K / non-reduce loop (a map prologue) is not a plain matmul
        if not body_loops:
            return None
        if not any(isinstance(s, Write) for s in inner_body):
            return None
        return MatmulSkeleton(
            inner_n=_map_axis(inner_loop),
            outer_m=_map_axis(chain[-2]),
            extra_outer=tuple(chain[:-2]),
            k_loop=k_loop,
            k_name=k_loop.axis.name,
            k_extent=k_dim.as_static(),
            inner_body=inner_body,
            leading=leading,
            target_names=frozenset(lp.axis.name for lp in body_loops),
        )

    if algebras == {AlgebraKind.MONOID}:
        # Cooperative reduce. The reduces (one for sum/max; two for softmax) and
        # any second-pass map loop all iterate the SAME K axis — `upstream
        # unify_sibling_reduce_axes` collapses the sibling reduce + map axes to
        # one canonical name, and `_replace_k_coop` rewrites every K-named loop.
        # So the epilogue MAP scalar stmts (RMSNorm rsqrt, softmax log) and the
        # normalize map loop just ride the row tile — no rigid envelope.
        k_dim = k_loop.axis.extent
        # Tiling extent: static size, or the Dim hint for a symbolic K (masked).
        k_extent = k_dim.as_static() if k_dim.is_static else (k_dim.hint or 0)
        if k_extent < warp_size:
            return None
        # Match the reduce(s) and the second-pass map loop by K *Dim* (so static
        # and symbolic match uniformly). A body loop of any other extent is an
        # unexpected shape; leave it to legacy.
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None  # reduces over different K axes — multi-axis is a later phase
        body_loops = [s for s in inner_body if isinstance(s, Loop)]
        if any(lp.axis.extent != k_dim for lp in body_loops):
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
            k_bound=None if k_dim.is_static else k_dim.expr,
        )

    return None
