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


# BoundAxis.bind values — how an axis maps to GPU parallel coords.
BIND_THREAD = "THREAD"
BIND_BLOCK = "BLOCK"


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


__all__ = ["Axis", "BoundAxis", "BIND_THREAD", "BIND_BLOCK"]
