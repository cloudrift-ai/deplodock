"""Lift a hardware-free algebraic skeleton from a ``LoopOp``.

Phase 1 covers the pointwise (``MAP``) regime: a loop nest of free (non-reduce)
axes ending in a write, with no reduce carrier anywhere. The skeleton names the
innermost free axis ``N`` and the next-out one ``M`` (matching the legacy
planner's ``outer_n`` / ``outer_m``), plus any extra outer free loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Loop, Stmt, Write


@dataclass(frozen=True)
class MapAxis:
    """One free (map) axis to tile. ``extent`` is the static size, or the
    ``Dim`` hint for a symbolic axis (``symbolic=True``)."""

    loop: Loop
    symbolic: bool
    extent: int


@dataclass(frozen=True)
class PointwiseSkeleton:
    """Pointwise kernel shape: free axes + the body to tile."""

    inner_n: MapAxis
    outer_m: MapAxis | None
    extra_outer: tuple[Loop, ...]
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]


def _split_leading_non_loops(body: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _map_axis(loop: Loop) -> MapAxis:
    ext = loop.axis.extent
    if ext.is_static:
        return MapAxis(loop=loop, symbolic=False, extent=ext.as_static())
    return MapAxis(loop=loop, symbolic=True, extent=ext.hint or 0)


@dataclass(frozen=True)
class MatmulSkeleton:
    """Plain scalar-matmul shape: free output axes ``M`` (outer) / ``N`` (inner)
    + one ``SEMIRING`` reduce axis ``K``, with the canonical body
    ``[Loop(k, [Load a, Load b, Assign(multiply), Accum]), Write(o[m,n])]``.

    Phase 2a covers the scalar tier with no split-K / cooperative-K / fused
    prologue and a static ``K`` extent; everything else returns ``None`` from
    :func:`lift_matmul` (legacy fallthrough)."""

    inner_n: MapAxis
    outer_m: MapAxis
    extra_outer: tuple[Loop, ...]
    k_loop: Loop
    k_name: str
    k_extent: int
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]


def lift_matmul(loop_op: LoopOp) -> MatmulSkeleton | None:
    """Lift a plain scalar-matmul skeleton, or ``None`` for anything outside the
    Phase-2a envelope (symbolic K, multi-accumulator, fused prologue, missing M
    axis, any non-``SEMIRING`` reduce → legacy fallthrough)."""
    reduce_loops = [lp for lp in loop_op.body.iter_of_type(Loop) if lp.is_reduce]
    if not reduce_loops or any(lp.algebra_kind is not AlgebraKind.SEMIRING for lp in reduce_loops):
        return None
    if len(reduce_loops) != 1:
        return None  # multi-accumulator matmul — defer to legacy
    k_loop = reduce_loops[0]
    if not k_loop.axis.extent.is_static:
        return None  # symbolic K stays on the legacy degenerate path for now

    leading, rest = _split_leading_non_loops(tuple(loop_op.body))
    chain: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    if len(chain) < 2:
        return None  # need both M and N free axes (outer_m required)

    inner_n_loop = chain[-1]
    inner_body = tuple(inner_n_loop.body)
    # Canonical body: exactly the K reduce loop + Write(s), no prologue siblings.
    loops_in = [s for s in inner_body if isinstance(s, Loop)]
    if loops_in != [k_loop]:
        return None
    if not any(isinstance(s, Write) for s in inner_body):
        return None
    if any(not isinstance(s, (Loop, Write)) for s in inner_body):
        return None  # leading assigns / extra stmts ⇒ fused prologue, not plain matmul

    return MatmulSkeleton(
        inner_n=_map_axis(inner_n_loop),
        outer_m=_map_axis(chain[-2]),
        extra_outer=tuple(chain[:-2]),
        k_loop=k_loop,
        k_name=k_loop.axis.name,
        k_extent=k_loop.axis.extent.as_static(),
        inner_body=inner_body,
        leading=leading,
    )


@dataclass(frozen=True)
class CoopReduceSkeleton:
    """Plain associative reduce (`MONOID`) over a static K axis ≥ warp_size, with
    free output rows. Phase 3b covers the whole-CTA cooperative form: each CTA
    reduces one row's K across `BR` threads, then a warp/tree combine folds the
    partials. The canonical body is `[Loop(k, [Load, Accum]), Write(o[rows])]`."""

    inner_n: MapAxis
    outer_m: MapAxis | None
    extra_outer: tuple[Loop, ...]
    k_loop: Loop
    k_name: str
    k_extent: int
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]


def lift_coop_reduce(loop_op: LoopOp, *, warp_size: int = 32) -> CoopReduceSkeleton | None:
    """Lift a plain cooperative-reduce skeleton, or `None` outside the Phase-3b
    envelope (non-`MONOID` reduce, symbolic / small K, multi-reduce, fused
    epilogue → legacy fallthrough)."""
    reduce_loops = [lp for lp in loop_op.body.iter_of_type(Loop) if lp.is_reduce]
    if len(reduce_loops) != 1:
        return None
    k_loop = reduce_loops[0]
    if k_loop.algebra_kind is not AlgebraKind.MONOID:
        return None  # SEMIRING → matmul; TWISTED_MONOID → flash (out of scope)
    if not k_loop.axis.extent.is_static or k_loop.axis.extent.as_static() < warp_size:
        return None  # small / symbolic reduce stays on the legacy / pointwise path

    leading, rest = _split_leading_non_loops(tuple(loop_op.body))
    chain: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    if not chain:
        return None

    inner_n_loop = chain[-1]
    inner_body = tuple(inner_n_loop.body)
    loops_in = [s for s in inner_body if isinstance(s, Loop)]
    if loops_in != [k_loop]:
        return None
    if not any(isinstance(s, Write) for s in inner_body):
        return None
    if any(not isinstance(s, (Loop, Write)) for s in inner_body):
        return None  # epilogue (e.g. RMSNorm rsqrt) — deferred

    return CoopReduceSkeleton(
        inner_n=_map_axis(inner_n_loop),
        outer_m=_map_axis(chain[-2]) if len(chain) >= 2 else None,
        extra_outer=tuple(chain[:-2]) if len(chain) >= 2 else tuple(chain[:-1]),
        k_loop=k_loop,
        k_name=k_loop.axis.name,
        k_extent=k_loop.axis.extent.as_static(),
        inner_body=inner_body,
        leading=leading,
    )


def lift_pointwise(loop_op: LoopOp) -> PointwiseSkeleton | None:
    """Lift a pointwise skeleton, or ``None`` if the kernel has any reduce
    carrier (not pointwise → the dispatcher falls through to the legacy
    planner). A chain-less body (a bare write, no free axis) also returns
    ``None`` — the phantom-axis case stays on the legacy path for now.
    """
    body = tuple(loop_op.body)
    # Any reduce loop anywhere disqualifies the pointwise regime.
    if any(lp.is_reduce for lp in loop_op.body.iter_of_type(Loop)):
        return None

    leading, rest = _split_leading_non_loops(body)
    chain: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    if not chain:
        return None

    inner_n = _map_axis(chain[-1])
    outer_m = _map_axis(chain[-2]) if len(chain) >= 2 else None
    extra_outer = tuple(chain[:-2]) if outer_m is not None else tuple(chain[:-1])
    return PointwiseSkeleton(
        inner_n=inner_n,
        outer_m=outer_m,
        extra_outer=extra_outer,
        inner_body=tuple(chain[-1].body),
        leading=leading,
    )
