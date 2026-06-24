"""Regime classification — tag a nest by its reduce axes' carrier algebra.

The recognition predicate the enumeration passes dispatch on. There is **no shape
matching** — the regime is purely the reduce axes' ``Loop.algebra_kind`` (``MAP``
no contraction, ``SEMIRING`` a contraction, ``MONOID`` a plain reduce,
``TWISTED_MONOID`` an online stream), read off the derived iteration DAG.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Write
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import AxisRole, IterDag


@dataclass(frozen=True)
class _Regime:
    """The classification handoff: the nest's algebra + the contraction-axis names
    a reduce decomposition rewrites (``target_names``)."""

    algebra: AlgebraKind  # MAP | SEMIRING | MONOID | TWISTED_MONOID
    target_names: frozenset[str] = frozenset()


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG — purely the reduce axes'
    ``Loop.algebra_kind``, or ``None`` for a shape the moves don't cover."""
    if not dag.parallel:
        return None
    reduce_loops = [n.loop for n in dag.reduce]
    algebras = dag.algebras
    inner_body = dag.inner_body
    nested_reduce = any(n.parent is not None and n.parent.role is AxisRole.REDUCE for n in dag.reduce)
    coop_monoid = AlgebraKind.TWISTED_MONOID in algebras and not nested_reduce

    if AlgebraKind.TWISTED_MONOID in algebras and nested_reduce:
        if len(dag.parallel) < 2:
            return None
        if any(not lp.axis.extent.is_static for lp in reduce_loops if lp.algebra_kind != AlgebraKind.TWISTED_MONOID):
            return None
        return _Regime(AlgebraKind.TWISTED_MONOID, frozenset(lp.axis.name for lp in reduce_loops))

    if not reduce_loops:  # no contraction — a MAP nest.
        return _Regime(AlgebraKind.MAP)

    k_dim = reduce_loops[0].axis.extent
    body_loops = [s for s in inner_body if isinstance(s, Loop)]

    if algebras == {AlgebraKind.SEMIRING}:
        # A symbolic (masked) K tiles at the ``Dim`` hint and zero-fills the partial
        # final K tile past the runtime bound (the masked-K mma tier — the warp build
        # ceil-divides ``K_o`` and the smem slab / ldmatrix load zero-fills the overhang,
        # so the mma accumulates 0 past ``k_bound``). A static K must be a real extent.
        k_extent = k_dim.as_static() if k_dim.is_static else (k_dim.hint or 0)
        if k_extent < 1 or len(dag.parallel) < 2:
            return None
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None
        if not body_loops or any(lp.axis.extent != k_dim or not lp.is_reduce for lp in body_loops):
            return None
        if not any(isinstance(s, Write) for s in inner_body):
            return None
        return _Regime(AlgebraKind.SEMIRING, frozenset(lp.axis.name for lp in body_loops))

    if algebras == {AlgebraKind.MONOID} or coop_monoid:
        if coop_monoid and not k_dim.is_static:
            return None
        k_extent = k_dim.as_static() if k_dim.is_static else (k_dim.hint or 0)
        if k_extent < 1:
            return None
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None
        if any(lp.axis.extent != k_dim for lp in body_loops):
            return None
        return _Regime(AlgebraKind.MONOID, frozenset(lp.axis.name for lp in body_loops))

    return None
