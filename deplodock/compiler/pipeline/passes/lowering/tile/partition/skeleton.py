"""Lift a hardware-free algebraic skeleton from a ``LoopOp``.

Phase 1 covers the pointwise (``MAP``) regime: a loop nest of free (non-reduce)
axes ending in a write, with no reduce carrier anywhere. The skeleton names the
innermost free axis ``N`` and the next-out one ``M`` (matching the legacy
planner's ``outer_n`` / ``outer_m``), plus any extra outer free loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.loop import LoopOp
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
