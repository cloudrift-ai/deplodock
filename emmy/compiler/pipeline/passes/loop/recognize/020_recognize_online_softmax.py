"""Recognize a standalone two-pass softmax at Loop IR → fuse to one online-softmax pass.

A standalone ``softmax(x, dim=-1)`` decomposes into THREE loops over the reduce axis: a row-max
reduce, a ``Σ exp(x − max)`` reduce, then a normalize. This recognizer fuses the first two — the only
ones that re-read ``x`` to recompute the same `exp` — into a single streaming **online-softmax**
:class:`~emmy.compiler.ir.stmt.Monoid` pass (the flash softmax-stats trick applied standalone):

    for kv: m = max(m, x);  d = d·exp(m_old − m) + exp(x − m);   (one pass)

The carried states keep the original Accum names (``m`` = the rowmax acc, ``d`` = the sum acc), so the
downstream ``reciprocal(d)`` + normalize loop are untouched — 3 passes over ``x`` become 2. The
resulting flat ``Monoid`` flows through the MONOID regime like any reduce; its serial realization is
the carrier-generic :class:`~emmy.compiler.ir.twist.ScalarCombiner`.

A pattern-recognition pass (sibling of ``010_recognize_flash``): runs after the ``loop/fusion``
fixpoint, gated by the ``ONLINE_SOFTMAX`` knob (off → untouched). The matcher anchors on an adjacent
``(rowmax-reduce, sum-of-exp-reduce)`` loop pair over the same input + reduce extent where the sum
folds ``exp(x − rowmax_acc)`` — specific enough not to fire on unrelated reduce pairs.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.loop.ir import Accum, Assign, Body, Init, Load, Loop, LoopOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.recognize._flash import ONLINE_SOFTMAX, online_softmax_combine

PATTERN = [Pattern("root", LoopOp)]


def _enabled() -> bool:
    raw = ONLINE_SOFTMAX.raw()
    return raw is not None and ONLINE_SOFTMAX.parse(raw)


def _rowmax(loop: Loop) -> tuple[str, str, tuple] | None:
    """``(acc, input, index)`` if ``loop`` is a row-max reduce of a single ``Load`` — its body holds
    exactly one ``maximum`` ``Accum`` over a ``Load``'s value."""
    body = list(loop.body)
    maxes = [s for s in body if isinstance(s, Accum) and s.op.reduce_canon == "maximum"]
    if len(maxes) != 1:
        return None
    acc = maxes[0]
    ld = next((s for s in body if isinstance(s, Load) and s.name == acc.value), None)
    return (acc.name, ld.input, ld.index) if ld is not None else None


def _sumexp(loop: Loop, maxacc: str, input_buf: str) -> str | None:
    """The sum ``Accum`` name if ``loop`` is a ``Σ exp(x − maxacc)`` reduce over the same ``input_buf``
    — its body folds ``add`` over ``exp(subtract(load(input_buf, …), maxacc))``."""
    body = list(loop.body)
    sums = [s for s in body if isinstance(s, Accum) and s.op.reduce_canon == "add"]
    if len(sums) != 1:
        return None
    acc2 = sums[0]
    expa = next((s for s in body if isinstance(s, Assign) and s.name == acc2.value and s.op.name == "exp"), None)
    if expa is None:
        return None
    suba = next((s for s in body if isinstance(s, Assign) and s.name == expa.args[0] and s.op.name == "subtract"), None)
    if suba is None or maxacc not in suba.args:
        return None
    ld = next((s for s in body if isinstance(s, Load) and s.name == suba.args[0] and s.input == input_buf), None)
    return acc2.name if ld is not None else None


def _fuse(body: Body) -> tuple[Body, bool]:
    """Recurse into nested ``Loop`` bodies; fuse any adjacent ``(rowmax, sum-of-exp)`` reduce pair over
    the same input + reduce extent into one online-softmax ``Monoid`` loop (+ the carried-state seeds)."""
    stmts = list(body)
    out: list = []
    changed = False
    i = 0
    while i < len(stmts):
        s = stmts[i]
        if isinstance(s, Loop) and i + 1 < len(stmts) and isinstance(stmts[i + 1], Loop):
            nxt = stmts[i + 1]
            mx = _rowmax(s)
            if mx is not None and s.axis.extent == nxt.axis.extent:
                maxacc, input_buf, index = mx
                sumacc = _sumexp(nxt, maxacc, input_buf)
                if sumacc is not None:
                    src = f"{maxacc}__osin"
                    fused = Loop(
                        axis=s.axis,
                        body=Body.coerce(
                            (Load(name=src, input=input_buf, index=index), online_softmax_combine(maxacc, sumacc, src, axis=s.axis.name))
                        ),
                    )
                    out += [
                        Init(name=maxacc, op=ElementwiseImpl("maximum"), dtype="f32"),
                        Init(name=sumacc, op=ElementwiseImpl("add"), dtype="f32"),
                        fused,
                    ]
                    changed = True
                    i += 2
                    continue
        if isinstance(s, Loop):
            nb, ch = _fuse(s.body)
            if ch:
                s = replace(s, body=nb)
                changed = True
        out.append(s)
        i += 1
    return Body.coerce(out), changed


def rewrite(root: Node) -> Graph | None:
    if not _enabled():
        raise RuleSkipped("ONLINE_SOFTMAX off — two-pass softmax kept")
    new_body, changed = _fuse(root.op.body)
    if not changed:
        raise RuleSkipped("no two-pass softmax (rowmax + sum-of-exp) to fuse")
    return LoopOp(body=new_body)
