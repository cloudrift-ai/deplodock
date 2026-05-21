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

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """Planner-assigned role tag on a body ``Loop`` / ``StridedLoop``.

    Set by ``000_partition_planner`` to communicate axis-structure decisions
    to downstream materialization passes. Excluded from ``Body.structural_key``
    (not rendered by ``pretty()``) so adding a role doesn't invalidate
    autotune cache entries.

    Roles assigned to body loops:

    - ``BLOCK``: output axis (or split sub-axis) that ``001_launch_geometry`` lifts
      into ``Tile.axes`` with ``BIND_BLOCK``. Stamped by the planner.
    - ``THREAD``: output axis (or split sub-axis) that ``001_launch_geometry`` lifts
      into ``Tile.axes`` with ``BIND_THREAD``. Stamped by the planner.
    - ``SPLITK_BLOCK``: cross-CTA split-K outer axis. Lifted by
      ``001_launch_geometry`` into ``Tile.axes`` with ``BIND_BLOCK``;
      the planner also rewrites the epilogue Write to be atomic when
      this role is present.
    - ``REGISTER``: inner axis of a thread-axis split. ``001_launch_geometry`` stops
      lifting at this axis (keeps it as a body Loop instead of pulling into
      ``Tile.axes``); ``006a_register_tile_planned`` replicates dependent
      stmts along it.
    - ``STAGE_INNER``: inner reduce axis after a K split. Slab dimension for
      ``007_stage_inputs``.
    - ``SERIAL_OUTER``: outer serial chunk loop (e.g. K_o). Pipeline / double-
      buffer targets.
    - ``PIPELINE``: serial outer loop marked for pipelining by ``015_pipeline_k_outer``.
    """

    BLOCK = "block"
    THREAD = "thread"
    SPLITK_BLOCK = "splitk_block"
    REGISTER = "register"
    STAGE_INNER = "stage_inner"
    SERIAL_OUTER = "serial_outer"
    PIPELINE = "pipeline"


@dataclass(frozen=True)
class Axis:
    """One named iteration variable.

    Referenced from ``Expr`` subtrees by ``Var(name)``. Extent is a
    static integer in v1; future revisions may allow an ``Expr`` for
    dynamic batch / sequence dims.
    """

    name: str
    extent: int

    def split(self, factor: int) -> tuple[Axis, Axis]:
        """Split this axis into ``(outer, inner)`` for tile-style decomposition.

        Outer extent is ``self.extent // factor``, inner extent is ``factor``.
        Names follow the ``f"{self.name}_o"`` / ``f"{self.name}_i"`` convention
        so tiled IR remains readable. v1 requires divisibility — non-divisible
        extents need a residue-tail story that no current rule wants.
        """
        if self.extent % factor != 0:
            raise ValueError(f"Axis.split: {self.name} extent {self.extent} not divisible by {factor}")
        return Axis(f"{self.name}_o", self.extent // factor), Axis(f"{self.name}_i", factor)


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

    Used by ``Tile`` (across Tile and Kernel IRs) to express how each
    output axis maps to the parallel hierarchy. ``BIND_THREAD``
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


__all__ = ["Axis", "BoundAxis", "BIND_THREAD", "BIND_BLOCK", "Role"]
