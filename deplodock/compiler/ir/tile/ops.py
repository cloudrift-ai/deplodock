"""High-level algebraic op tree — the geometry-free compute layer.

The algebraic vocabulary itself lives in :mod:`deplodock.compiler.ir.stmt.algebra` —
the lift :class:`~deplodock.compiler.ir.stmt.algebra.Map` (re-exported here), the fold
carrier ``Monoid`` (+ ``Twist``), and the contraction ``Semiring``. This module is the
**lowering** of that tree to loop IR (:func:`lower`) plus the structural pretty-printer.

A kernel's compute is a tree of three node kinds, each of whose children
(``Monoid.partial`` / ``Semiring.operands``) is itself one of the three:

- ``Map`` — a pointwise body: a typed sequence of loop-IR stmts (operand ``Load``\\ s, the
  lift ``Assign``\\ s, an optional masking ``Select``, and — at the kernel root — the
  output ``Write``) that binds a value name as its last defining stmt. It **is** a
  :class:`~deplodock.compiler.ir.stmt.body.Body`, so there is nothing to "lower" — the
  stmts are the lowering. A bare operand load is a one-``Load`` ``Map``.
- ``Monoid`` — a fold over its ``axis`` through the carrier (its ``Twist`` is the
  combine), folding the ``partial`` sources. Flash is the ``(m, l, O)`` monoid whose
  score partial is a ``Map`` (or a nested ``Semiring``) and value partial a one-``Load``
  ``Map``.
- ``Semiring`` — a contraction ``reduce(⊕) ∘ map(⊗)`` (the matmul): the ``lift`` ⊗ over
  its ``operands``, folded by the additive ``fold`` ⊕ over ``reduce_axis``.

:func:`lower` emits loop-IR stmts: a ``Monoid`` generates the streaming ``Loop`` (its
partials expanded as sibling stmts + the carrier fold) then the carrier's ``finalize`` φ
— and leaves an in-loop carrier with its partials reduced to bound names; the carried
states are seeded by ``Loop.render`` from ``state.identity`` (no ``Init`` stmts, the same
prelude it uses for ``Accum``\\ s). A ``Semiring`` generates its contraction ``Loop`` in
the matmul-recognizable ``Accum``-in-``Loop`` form; a ``Map`` is already stmts (returned
as-is). So a kernel *is* the lowered tree — no per-kernel builder.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.stmt import Assign, Body, Loop, Map, Monoid, Semiring
from deplodock.compiler.ir.stmt.algebra import _partial_name
from deplodock.compiler.ir.stmt.base import Stmt, pretty_body


def lower(op) -> list[Stmt]:
    """Lower an op-tree node to loop-IR stmts. A ``Map`` is already stmts (returned
    as-is); a ``Monoid`` / ``Semiring`` generates its reduce ``Loop`` (states seeded by
    ``Loop.render``). One ``lower`` call on the root node emits the kernel's per-cell body."""
    if isinstance(op, Map):
        return list(op)
    if isinstance(op, Monoid):
        return _lower_monoid(op)
    if isinstance(op, Semiring):
        return _lower_semiring(op)
    raise TypeError(f"lower: expected Map / Monoid / Semiring node, got {type(op).__name__}")


def _lower_partial(p) -> list[Stmt]:
    """Emit the stmts a partial / operand source contributes: a node (``Map`` →
    its stmts, ``Monoid`` / ``Semiring`` → ``lower``), or nothing for a ``str`` (an
    already-bound sibling name — a loop-IR carrier shape, not an op-tree source)."""
    return [] if isinstance(p, str) else lower(p)


def _lower_monoid(m: Monoid) -> list[Stmt]:
    """The streaming ``Loop`` whose body expands the partial sources (siblings) and
    applies the carrier fold, then the carrier's ``finalize`` φ (the post-loop projection
    of the final state to the output — empty for a plain reduce, ``O/l`` for flash). No
    ``Init`` stmts: ``Loop.render`` seeds each carried state from ``state.identity`` in
    the same pre-loop prelude it uses for ``Accum``\\ s. The in-loop carrier is this
    ``Monoid`` with its partial sources reduced to their bound names and its
    self-contained ``axis`` / ``finalize`` stripped — the loop-IR carrier stmt the
    renderer / passes consume. Pure structure-from-carrier."""
    body: list[Stmt] = []
    for p in m.partial:
        body += _lower_partial(p)
    carrier = replace(m, partial=m.partial_names(), axis=None, finalize=())
    body.append(carrier)
    return [Loop(axis=m.axis, body=Body(tuple(body))), *m.finalize]


def _lower_semiring(s: Semiring) -> list[Stmt]:
    """The contraction ``Loop`` in the matmul-recognizable ``Accum``-in-``Loop`` form
    (``Semiring.match`` reads it back): expand each operand source (siblings), the lift
    ⊗ ``Assign`` (``fold.value = lift(operands…)``), and the additive ``fold`` ⊕ (its
    identity init is the ``Loop``'s immediate-``Accum`` prelude — no explicit ``Init``)."""
    body: list[Stmt] = []
    names: list[str] = []
    for opnd in s.operands:
        body += _lower_partial(opnd)
        names.append(_partial_name(opnd))
    body.append(Assign(name=s.fold.value, op=s.lift, args=tuple(names)))
    body.append(s.fold)
    return [Loop(axis=s.reduce_axis, body=Body(tuple(body)))]


def pretty(op, indent: str = "") -> list[str]:
    """Structurally pretty-print an op tree (for dumps) — WITHOUT lowering. A ``Map`` is
    its stmt body; a ``Monoid`` / ``Semiring`` is a header (the carrier / contraction
    over its axis, projecting to ``out``) above its named children, with the ``Monoid``'s
    ``finalize`` φ; any other stmt prints itself."""
    if isinstance(op, Map):
        return list(pretty_body(op, indent))
    if isinstance(op, Monoid):
        carrier = op.pretty()[0].strip()
        ax = op.axis.name if op.axis is not None else "?"
        lines = [f"{indent}reduce[{ax}] {carrier} -> {op.out}"]
        for src, pname in zip(op.partial, op.partial_names(), strict=True):
            lines.append(f"{indent}  partial {pname}:")
            lines += pretty(src, indent + "    ")
        for a in op.finalize:
            lines += a.pretty(indent + "  ")
        return lines
    if isinstance(op, Semiring):
        lines = [f"{indent}contract[{op.reduce_axis.name}] {op.lift.name} / {op.fold.op.name} -> {op.out}"]
        for opnd in op.operands:
            lines.append(f"{indent}  operand {_partial_name(opnd)}:")
            lines += pretty(opnd, indent + "    ")
        return lines
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = ["Map", "Monoid", "Semiring", "lower", "pretty"]
