"""Regime classification — tag a nest by its reduce axes' carrier algebra.

The recognition predicate the enumeration passes dispatch on. There is **no shape
matching** — the regime is purely the reduce axes' ``Loop.algebra_kind`` (``MAP``
no contraction, ``SEMIRING`` a contraction, ``MONOID`` an associative reduce),
read off the derived iteration DAG.

The streaming-flash schedule is **not** a distinct algebra (a twisted monoid is a
monoid — transport of structure): it is a *structural* property of a ``MONOID``
nest — a tuple `Monoid` carrier streaming over a *nested* contraction (flash's
QK^T reduce inside the KV stream). It is **derived on demand** (``IterDag.streaming``),
never stored on the regime: the streaming fork (``080_streaming``) and the knob-pin
validator query the DAG when they need the distinction; a plain `Accum` / non-nested
`Monoid` reduce is the cooperative ``MONOID`` regime.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Monoid, Write
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag


@dataclass(frozen=True)
class _Regime:
    """The classification handoff: the nest's algebra + the contraction-axis names
    a reduce decomposition rewrites (``target_names``). The streaming-flash schedule
    is **not** carried here — it is a derived property of the DAG (``IterDag.streaming``)
    queried on demand by the moves that need it.

    ``inner_algebra`` carries the **compositional** algebra of a twisted carrier:
    a streaming-flash
    ``Monoid`` carrier is ``MONOID(SEMIRING)`` — its online-softmax combine is a
    SEMIRING accumulation (the embedded P@V ``O += p·v`` over the hinge ``kv``) twisted
    by the MONOID rescale (``α``). ``inner_algebra = SEMIRING`` surfaces that embedded
    contraction so the shared-axis ``reduce_decomp`` (Phase 1c) can tile the hinge as
    BOTH a carrier reduce and a P@V reduce. ``None`` for a flat (non-compositional)
    carrier — a plain reduce is just its ``algebra``."""

    algebra: AlgebraKind  # MAP | SEMIRING | MONOID
    target_names: frozenset[str] = frozenset()
    inner_algebra: AlgebraKind | None = None  # the embedded contraction's algebra for a twisted carrier (flash: SEMIRING P@V on the hinge)


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG — the reduce axes'
    ``Loop.algebra_kind`` (the streaming-flash schedule is derived separately via
    ``dag.streaming``), or ``None`` for a shape the moves don't cover."""
    if not dag.parallel:
        return None
    reduce_loops = [n.loop for n in dag.reduce]
    algebras = dag.algebras
    inner_body = dag.inner_body

    if dag.streaming:
        if len(dag.parallel) < 2:
            return None
        if any(not n.loop.axis.extent.is_static for n in dag.reduce if not isinstance(n.carrier, Monoid)):
            return None
        # The twisted carrier is compositional: ``MONOID(SEMIRING)`` when the DAG exposes
        # the carried contraction chain (the embedded P@V on the hinge ``kv``). The chain
        # is a structural property (``dag.chain``), so this stays a derived read.
        inner = AlgebraKind.SEMIRING if dag.chain is not None else None
        return _Regime(AlgebraKind.MONOID, frozenset(lp.axis.name for lp in reduce_loops), inner_algebra=inner)

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
        has_monoid = any(isinstance(n.carrier, Monoid) for n in dag.reduce)
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
