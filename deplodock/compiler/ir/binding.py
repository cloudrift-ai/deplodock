"""Shared axis-binding type used by Tile IR (``Block``) and Kernel IR
(``Enclosure``).

A ``BoundAxis`` pairs an iteration ``Axis`` with a binding kind that
says how its values map to GPU coords:

- ``BIND_THREAD`` — one thread per axis value (output axes flattened
  into ``threadIdx.x``).
- ``BIND_BLOCK`` — one CUDA block per axis value (axis decoded from
  ``blockIdx.x/y/z``).

Future bindings (``BIND_CHUNKED``, ``BIND_SPLIT(factor=…)``) attach
additional fields here without changing how axes are listed.

``Block`` and ``Enclosure`` both carry ``axes: tuple[BoundAxis, ...]``
as their primary representation. The legacy ``thread_axes`` /
``block_axes`` tuples are exposed as derived properties for the
renderer and launch-geometry code (which still partition by binding).
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.loop import Axis

# Binding values for output / parallel axes.
BIND_THREAD = "THREAD"
BIND_BLOCK = "BLOCK"


@dataclass(frozen=True)
class BoundAxis:
    """An axis paired with its GPU-coord binding.

    ``axis`` carries the iteration variable + extent. ``bind`` says
    where it lives in the launch geometry (thread vs CUDA block).
    """

    axis: Axis
    bind: str

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def extent(self) -> int:
        return self.axis.extent


__all__ = ["BoundAxis", "BIND_THREAD", "BIND_BLOCK"]
