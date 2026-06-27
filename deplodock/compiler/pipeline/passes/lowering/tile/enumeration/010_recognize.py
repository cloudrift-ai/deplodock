"""Recognize the reduce carrier's algebra → normalize it to a twisted ``Monoid``.

First of the two enumeration steps — recognition here, scheduling in
``020_schedule``. It rewrites a ``LoopOp`` in place so every ``MONOID`` reduce
carrier is expressed as a :class:`Monoid`, the one unified twisted-monoid
representation:

- a scalar :class:`Accum` (plain sum / max / mean) becomes its **degenerate**
  monoid (``Accum.as_monoid`` — the identity twist, no rescale), seeded by an
  enclosing :class:`Init`;
- an already-twisted ``Monoid`` (the online-softmax recognizer's carrier) is
  left untouched — it is the same representation, with the max-rescale twist;
- a ``SEMIRING`` contraction (matmul) keeps its ``Accum`` — a different algebra
  (two operations ⊗ / ⊕), recognized by the matmul tier later. (Converting it
  would flip ``classify_algebra`` to ``MONOID`` and mis-schedule it as a naive
  serial reduce.)

After this pass a plain reduction and online softmax share ONE representation,
so the scheduler and the kernel lowering never branch on which — the twist ψ
(the ``Monoid.merge`` program) is the only thing that differs.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind, classify_algebra
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Body, Init, Loop
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def _normalize(stmts: list[Stmt]) -> tuple[list[Stmt], bool]:
    """Rewrite each ``MONOID`` reduce ``Loop``'s ``Accum``\\ s to ``Init`` +
    degenerate ``Monoid`` (deep). Returns ``(stmts, changed)``; ``changed`` is
    ``False`` when nothing needed normalizing (so the rule can skip)."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and classify_algebra(s) is AlgebraKind.MONOID:
            accums = [x for x in s.body if isinstance(x, Accum)]
            if accums:
                for acc in accums:
                    out.append(Init(name=acc.name, op=acc.op, dtype=F32))
                new_body = [x.as_monoid() if isinstance(x, Accum) else x for x in s.body]
                out.append(Loop(axis=s.axis, body=Body(tuple(new_body)), unroll=s.unroll))
                changed = True
                continue
        if s.nested():
            subs = []
            for b in s.nested():
                nb, ch = _normalize(list(b))
                subs.append(Body(tuple(nb)))
                changed = changed or ch
            s = s.with_bodies(tuple(subs))
        out.append(s)
    return out, changed


def rewrite(match: Match, root: Node) -> LoopOp | None:
    loop: LoopOp = root.op
    new, changed = _normalize(list(loop.body))
    if not changed:
        raise RuleSkipped("no scalar reduce carrier to normalize to a monoid")
    return replace(loop, body=Body(tuple(new)))
