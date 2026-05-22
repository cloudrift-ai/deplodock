"""Axis primitives shared across all IR layers.

``Axis`` is the iteration-variable identity (name + extent) used by Loop
IR, Tile IR, and Kernel IR. ``BoundAxis`` pairs an axis with a parallel-
coord binding (``BIND_THREAD`` / ``BIND_BLOCK``), used by Tile-IR
``Tile`` (across Tile and Kernel IRs) to express how output axes map
to GPU coords.

Lives at ``ir/axis.py`` rather than inside any one IR package because
both concepts span every layer. Loop IR re-exports ``Axis`` for
convenience (lifting passes use ``from ir.loop import ...`` to grab the
full Loop-body vocabulary in one import); Tile and Kernel IR import
``Axis`` and ``BoundAxis`` directly from here.

Future bindings (``BIND_SPLIT(factor, outer, inner)`` for matmul tile
splits, ``BIND_CHUNKED(factor)`` if K-axis chunking ever lives at the
output level) will land here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Axis:
    """One named iteration variable.

    Referenced from ``Expr`` subtrees by ``Var(name)``. Extent is a
    static integer in v1; future revisions may allow an ``Expr`` for
    dynamic batch / sequence dims.

    ``source_axis`` is the original (pre-split) axis this one was carved
    out of. Top-level axes (the ones the frontend traces) have
    ``source_axis = None``; every sub-axis the partition planner produces
    (e.g. ``M_b``, ``M_t``, ``M_r`` from ``M``) points to the original.
    Used by downstream passes (MMA factorization, scope-walk origin
    derivation) to group surrounding axes by source-axis identity instead
    of name-suffix convention. Equality and hashing exclude ``source_axis``
    so Var-rename invariance is preserved — two Axes with the same name
    and extent are the same axis regardless of where they came from.
    """

    name: str
    extent: int
    source_axis: Axis | None = field(default=None, compare=False, hash=False)

    def split(self, factor: int) -> tuple[Axis, Axis]:
        """Split this axis into ``(outer, inner)`` for tile-style decomposition.

        Outer extent is ``self.extent // factor``, inner extent is ``factor``.
        Names follow the ``f"{self.name}_o"`` / ``f"{self.name}_i"`` convention
        so tiled IR remains readable. v1 requires divisibility — non-divisible
        extents need a residue-tail story that no current rule wants.

        Children inherit ``source_axis = self.source_axis or self`` — top-level
        axes become their own source on first split; further splits chain to
        the same original.
        """
        if self.extent % factor != 0:
            raise ValueError(f"Axis.split: {self.name} extent {self.extent} not divisible by {factor}")
        src = self.source_axis or self
        return (
            Axis(f"{self.name}_o", self.extent // factor, source_axis=src),
            Axis(f"{self.name}_i", factor, source_axis=src),
        )


# BoundAxis.bind values — used on ``Tile.axes`` / ``Tile.axes``
# to describe launch-geometry. Body loops carry no bind; the loop
# construct itself (``Loop`` for serial, ``StridedLoop`` for cooperative
# striding) encodes iteration shape.

# One thread per axis value (axis flattens into threadIdx.x).
BIND_THREAD = "THREAD"
# One CUDA block per axis value (axis flattens into blockIdx.x/y/z),
# threads inside cooperate.
BIND_BLOCK = "BLOCK"


@dataclass(frozen=True)
class BoundAxis:
    """An axis paired with its GPU-coord binding.

    Used by the legacy ``Tile`` class (now Loop-IR-only after the
    tile-flavor refactor — the new ``GridTile`` / ``ThreadTile`` carry
    bare ``Axis`` tuples instead). ``BIND_THREAD`` means "one thread per
    axis value"; ``BIND_BLOCK`` means "one CUDA block per axis value,
    threads inside cooperate."
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
