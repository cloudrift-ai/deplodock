"""Regime classification — tag a nest by its reduce axes' carrier algebra.

The recognition predicate the enumeration passes dispatch on. There is **no shape
matching** — the regime is purely the reduce axes' ``Loop.algebra_kind`` (``MAP``
no contraction, ``SEMIRING`` a contraction, ``MONOID`` an associative reduce),
read off the derived iteration DAG.

The streaming-flash schedule is **not** a distinct algebra (a twisted monoid is a
monoid — transport of structure): it is a *structural* property of a ``MONOID``
nest — a tuple `Monoid` carrier streaming over a *nested* contraction (flash's
QK^T reduce inside the KV stream). ``classify`` reads that off the DAG and flags
``_Regime.streaming`` so the streaming fork (``080_streaming``) and the knob-pin
validator select the streaming tier; a plain `Accum` / non-nested `Monoid` reduce
is the cooperative ``MONOID`` regime. See ``plans/twisted-monoid-carrier-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Monoid, Write
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import AxisRole, IterDag


@dataclass(frozen=True)
class _Regime:
    """The classification handoff: the nest's algebra + the contraction-axis names
    a reduce decomposition rewrites (``target_names``). ``streaming`` flags the
    flash schedule — a ``MONOID`` nest whose carrier is a tuple `Monoid` streaming
    over a nested contraction (selects the streaming tier, not the coop reduce)."""

    algebra: AlgebraKind  # MAP | SEMIRING | MONOID
    target_names: frozenset[str] = frozenset()
    streaming: bool = False


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG — the reduce axes'
    ``Loop.algebra_kind`` + the structural streaming flag, or ``None`` for a shape
    the moves don't cover."""
    if not dag.parallel:
        return None
    reduce_loops = [n.loop for n in dag.reduce]
    algebras = dag.algebras
    inner_body = dag.inner_body
    nested_reduce = any(n.parent is not None and n.parent.role is AxisRole.REDUCE for n in dag.reduce)
    # A tuple `Monoid` carrier (online-softmax LSE) reads as MONOID — it *is* a
    # monoid. What makes flash *stream* rather than coop-reduce is the nested
    # contraction (the QK^T reduce inside the KV stream): MONOID + nested + Monoid.
    has_monoid = any(isinstance(n.carrier, Monoid) for n in dag.reduce)
    streaming = nested_reduce and has_monoid

    if streaming:
        if len(dag.parallel) < 2:
            return None
        if any(not n.loop.axis.extent.is_static for n in dag.reduce if not isinstance(n.carrier, Monoid)):
            return None
        return _Regime(AlgebraKind.MONOID, frozenset(lp.axis.name for lp in reduce_loops), streaming=True)

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

    if algebras == {AlgebraKind.MONOID}:
        # A non-nested `Monoid` carrier (a per-row LSE with no inner contraction)
        # keeps its symbolic axis degenerate — the cooperative reduce can't tile a
        # symbolic K, so require a static extent (a plain `Accum` reduce may be symbolic).
        if has_monoid and not k_dim.is_static:
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
