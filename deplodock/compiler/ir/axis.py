"""Axis primitives shared across all IR layers.

``Axis`` is the iteration-variable identity (name + extent) used by Loop
IR, Tile IR, and Kernel IR.

Lives at ``ir/axis.py`` rather than inside any one IR package because the
concept spans every layer. Loop IR re-exports ``Axis`` for convenience
(lifting passes use ``from ir.loop import ...`` to grab the full Loop-body
vocabulary in one import); Tile and Kernel IR import ``Axis`` directly
from here.

The pre-refactor ``BoundAxis`` / ``BIND_BLOCK`` / ``BIND_THREAD`` triple
that packaged "axis plus its launch-coord binding" has been deleted: the
typed tile flavors (``GridTile`` / ``ThreadTile`` / ``RegisterTile``)
carry bare ``Axis`` tuples and encode the binding in the flavor's type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.dim import Dim, to_dim


@dataclass(frozen=True)
class Axis:
    """One named iteration variable.

    Referenced from ``Expr`` subtrees by ``Var(name)``. ``extent`` is a
    :class:`Dim` — static (``Dim(32)``) in most cases today, symbolic
    (``Dim("seq_len")``) for dynamic dims. Construction coerces a bare
    ``int`` / ``str`` to ``Dim`` for ergonomics, so ``Axis("m", 32)``
    keeps working.

    ``source_axis`` is the original (pre-split) axis this one was carved
    out of. Top-level axes (the ones the frontend traces) have
    ``source_axis = None``; every sub-axis the partition planner produces
    (e.g. ``M_b``, ``M_t``, ``M_r`` from ``M``) points to the original.
    Used by downstream passes (MMA factorization, scope-walk origin
    derivation) to group surrounding axes by source-axis identity instead
    of name-suffix convention. Equality and hashing exclude ``source_axis``
    so Var-rename invariance is preserved — two Axes with the same name
    and extent are the same axis regardless of where they came from.

    ``real_extent`` is the original static N (or M) extent BEFORE ceil-div
    rounding for masked tiles. Set on the block axis (``*_b``) by the
    partition planner when the underlying source axis has no divisor in
    ``_TUNE_AXIS_CHOICES``: the ``extent`` becomes ``ceil_div(real, BN·FN)``
    so the grid covers a partial last tile, and ``real_extent`` carries
    the bound used to gate boundary lanes. Materializer emits
    ``if (decoded_src_coord < real_extent) { ... }`` around the per-thread
    body. Excluded from equality / hashing for the same Var-rename-invariance
    reason as ``source_axis``.
    """

    name: str
    extent: Dim
    source_axis: Axis | None = field(default=None, compare=False, hash=False)
    real_extent: int | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.extent, Dim):
            object.__setattr__(self, "extent", to_dim(self.extent))

    def split(self, factor: int) -> tuple[Axis, Axis]:
        """Split this axis into ``(outer, inner)`` for tile-style decomposition.

        Outer extent is ``self.extent // factor``, inner extent is ``factor``.
        Names follow the ``f"{self.name}_o"`` / ``f"{self.name}_i"`` convention
        so tiled IR remains readable. v1 requires divisibility — non-divisible
        extents need a residue-tail story that no current rule wants. Symbolic
        extents refuse to split (M3 of the dynamic-shapes plan).

        Children inherit ``source_axis = self.source_axis or self`` — top-level
        axes become their own source on first split; further splits chain to
        the same original.
        """
        ext = self.extent.as_static()
        if ext % factor != 0:
            raise ValueError(f"Axis.split: {self.name} extent {ext} not divisible by {factor}")
        src = self.source_axis or self
        return (
            Axis(f"{self.name}_o", ext // factor, source_axis=src),
            Axis(f"{self.name}_i", factor, source_axis=src),
        )


__all__ = ["Axis"]
