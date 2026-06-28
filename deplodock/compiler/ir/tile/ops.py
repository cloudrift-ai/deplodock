"""High-level algebraic op tree — the geometry-free compute layer.

The algebraic vocabulary itself lives in :mod:`deplodock.compiler.ir.stmt.algebra` —
the lift :class:`~deplodock.compiler.ir.stmt.algebra.Map` (re-exported here), the fold
carrier ``Monoid`` (+ ``Twist``), and the contraction ``Semiring``. This module is the
**lowering** of that tree to loop IR (:func:`lower`) plus the structural pretty-printer.

A kernel's compute is a tree of three node kinds, each of whose children
(``Monoid.partial`` / ``Semiring.operands``) is itself one of the three:

- ``Map`` — a pointwise lift: a typed ``Body`` of loop-IR stmts (operand ``Load``\\ s, the
  lift ``Assign``\\ s, an optional masking ``Select``, and — at the kernel root — the
  output ``Write``) binding a value as its last def, optionally applied OVER a nested
  ``source`` node (``project ∘ reduce``). A bare operand load is a one-``Load`` ``Map``.
- ``Monoid`` — a fold over its ``axis`` through the carrier (its ``Twist`` is the
  combine), folding the ``partial`` sources. Flash is the ``(m, l, O)`` monoid whose
  score partial is a ``Map`` (or a nested ``Semiring``) and value partial a one-``Load``
  ``Map``; its ``O/l`` projection is a ``Map`` *over* the monoid.
- ``Semiring`` — a contraction ``reduce(⊕) ∘ map(⊗)`` (the matmul): the ``lift`` ⊗ over
  its ``operands``, folded by the additive ``fold`` ⊕ over ``reduce_axis``.

:func:`lower` emits loop-IR stmts: a ``Monoid`` generates an explicit ``Init`` per carried
state (``<f32> state = identity;``, via ``State.inits``) then the streaming ``Loop`` (its
partials expanded as sibling stmts + the carrier fold), and leaves an in-loop carrier with
its partials cleared. The ``Init`` stmts make the seed explicit IR so ``Loop.render`` stays
generic — it never reads ``state``. A ``Semiring`` generates its contraction ``Loop`` in
the matmul-recognizable ``Accum``-in-``Loop`` form; a ``Map`` emits its ``source``'s
lowering (if any) then its pointwise body. So a kernel *is* the lowered tree — no
per-kernel builder.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.stmt import Assign, Body, Loop, Map, Monoid, Semiring
from deplodock.compiler.ir.stmt.algebra import _partial_name
from deplodock.compiler.ir.stmt.base import INDENT, Stmt, pretty_body


def lower(op) -> list[Stmt]:
    """Lower an op-tree node to loop-IR stmts. A ``Map`` emits its ``source``'s lowering
    (if any) then its pointwise ``body``; a ``Monoid`` / ``Semiring`` generates its reduce
    ``Loop``. One ``lower`` call on the root node emits the kernel's per-cell body."""
    if isinstance(op, Map):
        return (lower(op.source) if op.source is not None else []) + list(op.body)
    if isinstance(op, Monoid):
        return _lower_monoid(op)
    if isinstance(op, Semiring):
        return _lower_semiring(op)
    raise TypeError(f"lower: expected Map / Monoid / Semiring node, got {type(op).__name__}")


def _lower_partial(p) -> list[Stmt]:
    """Emit the stmts a partial / operand source node contributes: a ``Map`` → its
    source + body, a ``Monoid`` / ``Semiring`` → its reduce ``Loop``. (Op-tree sources are
    always nodes; a loop-IR carrier has no source nodes — its partials are siblings.)"""
    return lower(p)


def _lower_monoid(m: Monoid) -> list[Stmt]:
    """The streaming ``Loop`` whose body expands the partial sources (siblings) and applies
    the carrier fold.

    A **degenerate** carrier (plain reduce — the identity twist) lowers to bare ``Accum``
    folds (``m.as_accums()``): the seed is derived from each ``Accum``'s ``op.identity`` by
    ``Loop.render``, so no explicit ``Init`` is emitted and the accumulator dtype rides on
    the ``Accum`` (stamped by ``030_stamp_types``). A **twisted** carrier (online softmax /
    flash) keeps the streaming merge: its state components are coupled (the ψ rescale reads
    sibling components), so it can't collapse to independent ``Accum``\\ s — its carried
    state is seeded by explicit ``Init`` stmts (``m.state.inits()``) before the ``Loop``.
    The in-loop carrier keeps its ``axis`` for the cooperative-axis analysis. The φ
    projection, if any, is a :class:`Map` *over* this Monoid — emitted by ``lower(Map)``,
    not here. Pure structure-from-carrier."""
    body: list[Stmt] = []
    for p in m.partial:
        body += _lower_partial(p)
    accums = m.as_accums()
    if accums is not None:
        body += accums
        return [Loop(axis=m.axis, body=Body(tuple(body)))]
    body.append(replace(m, partial=()))
    return [*m.state.inits(), Loop(axis=m.axis, body=Body(tuple(body)))]


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
    """Structurally pretty-print an op tree (for dumps) — WITHOUT lowering. A pure
    pointwise ``Map`` (no source) is just its body; a projection ``Map`` (``source`` set) is
    a ``map:`` node with an ``over:`` block (the nested node it maps over) and a ``body:``
    block (its pointwise body / the φ projection) — both nested under the map, so the body
    reads as part of it. A ``Monoid`` / ``Semiring`` is a header (the carrier / contraction
    over its axis, projecting to ``out``) above its named children; any other stmt prints
    itself."""
    # Every nesting level steps by one ``INDENT`` (a ``label:`` header sits at +INDENT under
    # its parent node, its content at +INDENT under the label) so the whole tree lines up.
    sub = indent + INDENT
    sub2 = sub + INDENT
    if isinstance(op, Map):
        if op.source is None:
            return list(pretty_body(op.body, indent))  # pure pointwise — the body IS the map
        # A projection Map: its ``body`` (the φ projection) and the ``source`` it maps over
        # are BOTH the map's — nest them under one ``map:`` so the body reads as part of it.
        lines = [f"{indent}map:", f"{sub}over:"]
        lines += pretty(op.source, sub2)
        lines.append(f"{sub}body:")
        lines += list(pretty_body(op.body, sub2))
        return lines
    if isinstance(op, Monoid):
        carrier = op.pretty()[0].strip()
        ax = op.axis.name if op.axis is not None else "?"
        lines = [f"{indent}monoid[{ax}] {carrier} -> {op.out}"]
        for src, pname in zip(op.partial, op.partial_names(), strict=True):
            lines.append(f"{sub}partial {pname}:")
            lines += pretty(src, sub2)
        return lines
    if isinstance(op, Semiring):
        lines = [f"{indent}semiring[{op.reduce_axis.name}] {op.lift.name} / {op.fold.op.name} -> {op.out}"]
        for opnd in op.operands:
            lines.append(f"{sub}operand {_partial_name(opnd)}:")
            lines += pretty(opnd, sub2)
        return lines
    if isinstance(op, Stmt):
        return list(op.pretty(indent))
    return [f"{indent}{op!r}"]


__all__ = ["Map", "Monoid", "Semiring", "lower", "pretty"]
