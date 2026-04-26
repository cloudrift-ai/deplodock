"""Axis primitives shared across all IR layers.

``Axis`` is the iteration-variable identity (name + extent) used by Loop
IR, Tile IR, and Kernel IR. ``BoundAxis`` pairs an axis with a parallel-
coord binding (``BIND_THREAD`` / ``BIND_BLOCK``), used by Tile-IR
``Block`` and Kernel-IR ``Enclosure`` to express how output axes map to
GPU coords.

Lives at ``ir/axis.py`` rather than inside any one IR package because
both concepts span every layer. Loop IR re-exports ``Axis`` for
back-compat (``from ir.loop import Axis`` continues to work); Tile and
Kernel IR import ``Axis`` and ``BoundAxis`` directly from here.

Future bindings (``BIND_SPLIT(factor, outer, inner)`` for matmul tile
splits, ``BIND_CHUNKED(factor)`` if K-axis chunking ever lives at the
output level) will land here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One named iteration variable.

    Referenced from ``Expr`` subtrees by ``Var(name)``. Extent is a
    static integer in v1; future revisions may allow an ``Expr`` for
    dynamic batch / sequence dims.
    """

    name: str
    extent: int


# BoundAxis.bind values — how an axis is treated by the surrounding
# scheduling context. ``Tile.axes`` / ``Enclosure.axes`` use the
# launch-geometry binds (``BIND_THREAD`` / ``BIND_BLOCK``);
# ``BoundLoop.axis`` uses only ``BIND_SERIAL``. Cooperative iteration
# is expressed by splitting an axis into ``(chunk, t)`` at strategy
# time, with the inner half marked ``BIND_THREAD`` in ``Tile.axes`` —
# no separate "strided body loop" binding is needed.

# --- Launch-geometry bindings (used in Tile.axes / Enclosure.axes) ---
# One thread per axis value (axis flattens into threadIdx.x).
BIND_THREAD = "THREAD"
# One CUDA block per axis value (axis flattens into blockIdx.x/y/z),
# threads inside cooperate.
BIND_BLOCK = "BLOCK"

# --- Body-loop binding (used on BoundLoop.axis) ---
# Each thread iterates the axis privately (renders to a plain serial
# ``for`` loop).
BIND_SERIAL = "SERIAL"


@dataclass(frozen=True)
class BoundAxis:
    """An axis paired with its GPU-coord binding.

    Used by Tile-IR ``Block`` and Kernel-IR ``Enclosure`` to express how
    each output axis maps to the parallel hierarchy. ``BIND_THREAD``
    means "one thread per axis value"; ``BIND_BLOCK`` means "one CUDA
    block per axis value, threads inside cooperate."
    """

    axis: Axis
    bind: str

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def extent(self) -> int:
        return self.axis.extent


def split_axis(ax: Axis, factor: int) -> tuple[Axis, Axis]:
    """Split ``ax`` into ``(outer, inner)`` for tile-style decomposition.

    Outer extent is ``ax.extent // factor``, inner extent is ``factor``.
    Names follow the ``f"{ax.name}_o"`` / ``f"{ax.name}_i"`` convention so
    tiled IR remains readable. v1 requires divisibility — non-divisible
    extents need a residue-tail story that no current rule wants.
    """
    if ax.extent % factor != 0:
        raise ValueError(f"split_axis: {ax.name} extent {ax.extent} not divisible by {factor}")
    return Axis(f"{ax.name}_o", ax.extent // factor), Axis(f"{ax.name}_i", factor)


__all__ = ["Axis", "BoundAxis", "BIND_THREAD", "BIND_BLOCK", "BIND_SERIAL", "split_axis"]
