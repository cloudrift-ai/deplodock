"""High-level algebraic op tree — the geometry-free compute layer.

The algebraic vocabulary itself lives in :mod:`deplodock.compiler.ir.stmt.algebra` —
the lift :class:`~deplodock.compiler.ir.stmt.algebra.Map` (re-exported here), the
carrier ``Monoid`` + ``Twist``, and the ``Semiring`` contraction view. This module is
the **op tree** built on top of them: the fold node :class:`Reduce`, the operand
descriptor :class:`TensorRef`, and :func:`lower`.

A kernel's compute is built from two things:

- :class:`Reduce` — a fold over one axis through a carrier (``Monoid`` + ``Twist``),
  whose **partials are nested**: a ``Map`` (a stmt body), a :class:`TensorRef`
  (a direct operand load), or another ``Reduce``. A contraction (matmul / the
  SEMIRING) is ``Reduce(⊕=+)`` over ``Map(⊗=·)``; flash is ``Reduce(lse)`` over the
  scaled ``Σ Q·K`` and ``V``.
- ``Map`` — a pointwise body: a typed sequence of loop-IR stmts (the operand
  ``Load``\\ s, the lift ``Assign``\\ s, an optional masking ``Select``, and — at the
  kernel root — the output ``Write``) that binds a value name as its last defining
  stmt. It **is** a :class:`Body`, so it carries Body's analysis helpers and there is
  nothing to "lower" — the stmts are the lowering. A ``Map`` plugged into a carrier
  partial supplies that partial's stmts (last-binding the partial's name); an opaque
  recovered subgraph — e.g. a fused-RoPE score whose Q/K are computed SSA, not loads —
  is just a ``Map``.

The tree is **geometry-free**: ``TensorRef`` (buffer + index exprs) is the only place
layout lives; the tree names the axes it folds and the operands it reads but says
nothing about threads / tiling (that is the separate ``Tile(op, placement)`` layer).

:func:`lower` emits loop-IR stmts: a ``Reduce`` generates the structure (an ``Init``
per carried state ← the carrier identity, the streaming ``Loop`` + the carrier fold,
then the carrier's ``finalize`` φ — the post-loop projection of the final state to the
output value, e.g. flash's ``O/l``), a ``Map`` is already stmts (returned as-is), and a
``TensorRef`` is a ``Load``. So a kernel *is* the lowered tree — no per-kernel builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt import Body, Init, Load, Loop, Map  # Map re-exported from ir.stmt.algebra
from deplodock.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class TensorRef:
    """An operand read: a buffer plus the index exprs that address it (over the
    iteration axes). The index is the *only* place layout lives — staging / the mma
    fragment read it; nothing is duplicated into the carrier."""

    buf: str
    index: tuple[Expr, ...]


# A partial source: a Map (stmt body), a direct operand load (TensorRef), or a nested Reduce.
Source = "Map | TensorRef | Reduce"


@dataclass(frozen=True)
class Reduce:
    """Fold over ``axis`` through ``carrier`` (a ``Monoid`` — its ``Twist`` is the
    combine). ``partials`` produce the carrier's ``partial`` contributions (one source
    per ``carrier.partial`` name, in order — a ``Map`` whose stmts last-bind that name,
    a ``TensorRef`` loaded into it, or a nested ``Reduce``). ``init_ops`` gives the
    per-state identity-bearing op for the enclosing ``Init`` (one per ``carrier.state``).
    ``out`` is the carried state read after the fold (the carrier's primary state)."""

    out: str
    axis: Axis
    carrier: object  # Monoid
    partials: tuple
    init_ops: tuple[ElementwiseImpl, ...]
    dtype: DataType = field(default=F32)


def _lower_source(src, name: str) -> list[Stmt]:
    """Emit stmts that bind ``name`` to the value produced by ``src``: a ``TensorRef``
    (→ a ``Load`` into ``name``), a ``Map`` (→ its stmts verbatim — they already bind
    ``name`` as their last defining stmt), or a nested ``Reduce`` (→ ``lower``)."""
    if isinstance(src, TensorRef):
        return [Load(name=name, input=src.buf, index=src.index)]
    if isinstance(src, Map):
        return list(src)
    if isinstance(src, Reduce):
        return lower(src)
    raise TypeError(f"_lower_source: unsupported partial source {type(src).__name__}")


def lower(op) -> list[Stmt]:
    """Lower an op-tree node to loop-IR stmts. A ``Map`` is already stmts (returned
    as-is); a ``Reduce`` generates its ``Init``\\ s + streaming ``Loop``. One ``lower``
    call on the root ``Map`` emits a whole kernel's per-cell body."""
    if isinstance(op, Map):
        return list(op)
    if isinstance(op, Reduce):
        return _lower_reduce(op)
    raise TypeError(f"lower: expected Map / Reduce root, got {type(op).__name__}")


def _lower_reduce(r: Reduce) -> list[Stmt]:
    """An ``Init`` per carried state (the carrier's identity), then the streaming
    ``Loop`` whose body computes the partials and applies the carrier fold, then the
    carrier's ``finalize`` φ (the post-loop projection of the final state to the output
    value — empty for a plain reduce / matmul, ``O/l`` for flash). Pure
    structure-from-carrier — no per-kernel assembly."""
    out: list[Stmt] = []
    for s, op in zip(r.carrier.state, r.init_ops, strict=True):
        out.append(Init(name=s, op=op, dtype=r.dtype))
    body: list[Stmt] = []
    for src, pname in zip(r.partials, r.carrier.partial, strict=True):
        body += _lower_source(src, pname)
    body.append(r.carrier)  # the Monoid fold (its Twist is the combine)
    out.append(Loop(axis=r.axis, body=Body(tuple(body))))
    out.extend(getattr(r.carrier, "finalize", ()))  # the carrier's φ: final state → output value (post-loop)
    return out


__all__ = ["TensorRef", "Map", "Reduce", "lower"]
