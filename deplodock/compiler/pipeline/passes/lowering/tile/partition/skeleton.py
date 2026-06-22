"""Hardware-free algebraic skeletons + the shared nest helpers.

One skeleton per regime the composer covers — pointwise (`MAP`), matmul
(`SEMIRING`), cooperative reduce (`MONOID`) — each naming the innermost free
axis `N`, the next-out one `M`, the extra outer free loops, and (for reduces)
the `K` axis. The recognition that fills them lives in `walk.py` (`walk_nest`);
this module holds only the dataclasses + the free-axis-chain helpers they share.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.stmt import Loop, Stmt


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
    # Every K-extent loop name to cooperatively split — the reduce(s) PLUS any
    # second-pass map loop (RMSNorm normalize, softmax exp). Keyed by extent
    # (== ``k_extent``) like the legacy planner, since the map loop carries a
    # different axis name than the reduce (only sibling *reduces* are unified
    # upstream). Defaults to ``{k_name}`` for the plain reduce.
    target_names: frozenset[str] = frozenset()
