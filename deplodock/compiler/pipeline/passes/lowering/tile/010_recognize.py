"""Recognize a ``LoopOp``'s algebraic structure → normalize it to a twisted ``Monoid``.

First of the two tile-lowering steps — recognition here, scheduling in
``020_schedule``. It rewrites a ``LoopOp`` so every reduce carrier is the one
unified twisted ``Monoid`` representation. Three recognitions, tried in order
(each is unconditional now — no knobs):

1. **Flash attention** — a softmax-then-P@V kernel (+ its scaled-QK producer) is
   the online-softmax *monoid* over a streaming KV reduce; rewrite the pair to one
   fused flash ``TileOp`` (the ``(m, l, O)`` twisted monoid). Graph rewrite —
   consumes the score producer. Recognition + construction live in ``_flash``
   (``try_flash``).
2. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same
   input fuses into one streaming online-softmax ``Monoid`` (the ``(m, d)`` twist).
   Recognition + construction live in ``_softmax`` (``try_online_softmax``).
3. **Normalize** — any remaining scalar ``Accum`` (a plain sum / max / mean) becomes
   its **degenerate** monoid (``Accum.as_monoid`` — the identity twist, no rescale);
   each carried state is seeded by an explicit ``Seed`` stmt before the loop
   (``Monoid.seeds()``). A semiring contraction (``Semiring.match``)
   keeps its ``Accum`` — degenerate-monoidizing it would lose the contraction
   structure the matmul tier reads off the body. This generic step stays here.

Flash must precede online-softmax which must precede normalize: each later step
consumes the ``Accum``\\ s an earlier one pattern-matches. After this pass a plain
reduction, online softmax, and flash all share ONE representation — only the twist ψ
(the ``Monoid``'s ``merge`` program) differs — so the scheduler and the kernel
lowering never branch on which.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Body, Loop, Monoid
from deplodock.compiler.ir.stmt.algebra import Semiring
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._flash import try_flash
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import try_online_softmax

PATTERN = [Pattern("root", LoopOp)]


def _normalize(stmts: list[Stmt]) -> tuple[list[Stmt], bool]:
    """Rewrite each plain reduce ``Loop``'s ``Accum``\\ s to their degenerate
    ``Monoid`` (deep; each carried state seeded by an explicit ``Seed`` stmt before
    the loop, ``Monoid.seeds()``). A semiring contraction (``Semiring.match``) keeps its
    ``Accum`` — degenerate-monoidizing it would lose the contraction structure the
    matmul tier reads. Returns ``(stmts, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and Semiring.match(s) is None:
            if any(isinstance(x, Accum) for x in s.body):
                # Each Accum becomes its degenerate Monoid; each carried state is seeded by
                # an explicit ``Seed`` stmt emitted before the loop (from ``state.identity``).
                new_body = [x.as_monoid() if isinstance(x, Accum) else x for x in s.body]
                out.extend(seed for x in new_body if isinstance(x, Monoid) for seed in x.seeds())
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


def rewrite(match: Match, root: Node) -> LoopOp | Graph | None:
    # Order matters: flash consumes the Accums online-softmax would match, which in
    # turn consumes the Accums normalize would convert.
    flash = try_flash(match, root)
    if flash is not None:
        return flash
    fused = try_online_softmax(root)
    if fused is not None:
        return fused
    new, changed = _normalize(list(root.op.body))
    if not changed:
        raise RuleSkipped("no reduce carrier to recognize or normalize")
    return replace(root.op, body=Body(tuple(new)))
