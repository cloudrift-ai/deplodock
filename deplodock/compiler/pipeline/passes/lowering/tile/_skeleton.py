"""Build the recognized :class:`~deplodock.compiler.ir.tile.skeleton.Skeleton` from a kernel.

Walks the op tree (``ir/stmt/algebra``) once at recognition, mirroring ``ops.lower``'s recursion,
and records the structural facts a schedule is chosen over: per-scope parallel axes, the single
reduce axis with its fold carrier, cooperative eligibility, and (for a contraction) the
operand→role binding. The scheduler then reads these instead of re-deriving them.

**The K-as-reduce normalization.** A ``Semiring`` contributes a reduce axis whose carrier is
``semiring.as_monoid()`` — so a contraction's K is structurally identical to any ``Monoid``
reduce axis and the Semiring↔Monoid duality never reaches scheduling.

The walk is **total**: it never raises. An unbindable contraction (scalar / 1-D output, or a
computed-cone operand — the cases recognition lifts for the per-cell fallback) stores
``binding=None`` rather than failing. Leading ``_`` so the pass loader skips this module (it is a
helper called from ``010_recognize`` / ``020_schedule``, not a standalone rule).
"""

from __future__ import annotations

from deplodock.compiler.ir.stmt.algebra import Semiring
from deplodock.compiler.ir.tile import MonoidKernel, ReduceAxis, Scope, Skeleton
from deplodock.compiler.pipeline.passes.lowering.tile._atomize import semiring_binding
from deplodock.compiler.pipeline.pipeline import LoweringError


def build_skeleton(kernel, free) -> Skeleton:
    """The :class:`Skeleton` for ``kernel`` whose root scope owns the output-ordered ``free``
    (parallel) axes. ``coop_eligible`` on the root reduce mirrors the scheduler's old
    ``_coop_carrier`` predicate (a ``MonoidKernel`` over a ``Monoid`` carrier with an axis)."""
    return Skeleton(root=_scope(kernel.op, tuple(free), root_kernel=kernel))


def _scope(node, parallel=(), root_kernel=None) -> Scope:
    """One carrier scope for ``node``. ``root_kernel`` is the owning ``*Kernel`` at the root (for
    the cooperative-eligibility test) and ``None`` for nested partial scopes."""
    inner = node.reduce_node  # Monoid | Semiring | None
    if inner is None:
        return Scope(node=node, parallel=parallel)
    if isinstance(inner, Semiring):
        red = ReduceAxis(
            axis=inner.reduce_axis,
            carrier=inner.as_monoid(),  # the K-as-reduce normalization
            contraction=True,
            coop_eligible=False,  # a contraction's K is not cooperative-eligible this cut
            binding=_try_bind(node, parallel),
        )
        return Scope(node=node, parallel=parallel, reduce=red)
    # A Monoid carrier (a reduction, or flash's online-softmax over nested contractions).
    coop = isinstance(root_kernel, MonoidKernel) and inner.axis is not None
    red = ReduceAxis(axis=inner.axis, carrier=inner, contraction=False, coop_eligible=coop)
    children = tuple(_scope(p) for p in inner.partial if _is_carrier(p))
    return Scope(node=node, parallel=parallel, reduce=red, children=children)


def _is_carrier(partial) -> bool:
    """True iff a ``Monoid`` partial source is itself a reduction (a nested carrier scope — flash's
    score ``Semiring``), vs a plain pointwise contribution (flash's value ``Map``) or a bound name."""
    return not isinstance(partial, str) and getattr(partial, "reduce_node", None) is not None


def _try_bind(node, grid):
    """The contraction operand→role :class:`AtomBinding`, or ``None`` when it can't be bound
    (scalar / 1-D output, a computed-cone operand, or a nested scope without a resolved grid)."""
    try:
        return semiring_binding(node, grid)
    except LoweringError:
        return None


__all__ = ["build_skeleton"]
