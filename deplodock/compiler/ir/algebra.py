"""Structural semiring recognition over a reduce loop.

The algebra of a kernel is **in the body** — the fold ⊕ is the carrier (``Accum``,
or ``Monoid`` + ``Twist``), the lift is the partial. There is no stored algebra
tag to keep in sync; passes read the structure directly.

A **semiring contraction** (matmul) is the one structural shape the schedule must
recognize to make tiling decisions: ``reduce(⊕) ∘ map(⊗)`` — for each output cell,
fold (⊕) over the reduce axis the product (⊗) of two operands. :class:`Semiring`
is that recognized view, computed on demand by :meth:`Semiring.match` (never
stored). It is distinguished from a *plain* reduce not by "two loads" but by the
genuine algebra: the lift ⊗ must **distribute over** the fold ⊕ (``multiply`` over
``add`` is a semiring; ``add`` over ``add`` — a sum of two operands — is not), and
there must be ≥ 2 distinct contracted operands (``x·x`` is a squared reduce, not a
contraction).

The fold's monoid laws (assoc + comm + identity) license blocking / cooperative-K
/ split-K, via the carrier's own ``combine_partials()`` — the same cross-partition
machinery any reduce uses. The lift + operand layouts license tiling with operand
reuse and, for the ``(×, +)`` semiring (:attr:`Semiring.is_additive`), the
tensor-core mma atom.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load


@dataclass(frozen=True)
class Operand:
    """One ⊗ input of a contraction — a buffer read plus its index exprs. The
    index is what the scheduler reads to derive operand reuse (which free axis the
    operand is invariant in) and layout (for smem staging / the mma fragment)."""

    buf: str
    index: tuple[Expr, ...]


@dataclass(frozen=True)
class Semiring:
    """Structural view of a semiring contraction in a reduce loop —
    ``reduce(⊕) ∘ map(⊗)``. Built by :meth:`match` from the loop body, never
    stored (the body is the source of truth).

    - ``fold`` — the ⊕ monoid carrier (an additive ``Accum``: identity 0, assoc + comm).
    - ``lift`` — the ⊗ product op; distributes over ⊕ and has a multiplicative identity.
    - ``operands`` — the contracted inputs (≥ 2 distinct buffers).
    - ``reduce_axis`` — the contracted (K) axis.
    """

    fold: Accum
    lift: ElementwiseImpl
    operands: tuple[Operand, ...]
    reduce_axis: Axis

    @property
    def is_additive(self) -> bool:
        """The ``(×, +)`` semiring the tensor-core mma implements — the gate for the
        mma atom (a tropical / min-plus contraction is still a semiring, still tiles,
        but has no hardware atom)."""
        return self.lift.name == "multiply" and self.fold.op.reduce_canon == "add"

    @staticmethod
    def match(loop) -> Semiring | None:
        """Recognize a semiring contraction on ``loop``, or ``None``. Duck-typed on
        ``.is_reduce`` / ``.axis`` / ``.body`` so it serves a Loop-IR ``Loop`` /
        ``StridedLoop`` alike (the caller restricts the type).

        A reduce loop is a contraction iff its single ``Accum`` fold's partial is
        produced by a lift that **distributes over** the fold op, contracting ≥ 2
        distinct operands over the reduce axis."""
        if not getattr(loop, "is_reduce", False):
            return None
        body = loop.body
        accs = [s for s in body if isinstance(s, Accum)]
        if len(accs) != 1:
            return None
        fold = accs[0]
        lift = next((s for s in body if isinstance(s, Assign) and s.name == fold.value), None)
        if lift is None or not lift.op.distributes_over(fold.op):
            return None
        k = loop.axis.name
        operands = tuple(
            Operand(ld.input, ld.index) for ld in body if isinstance(ld, Load) and k in {v for e in ld.index for v in e.free_vars()}
        )
        if len({o.buf for o in operands}) < 2:
            return None
        return Semiring(fold=fold, lift=lift.op, operands=operands, reduce_axis=loop.axis)
