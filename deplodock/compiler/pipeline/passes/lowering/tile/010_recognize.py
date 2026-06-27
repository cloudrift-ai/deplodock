"""Recognize a ``LoopOp``'s algebraic structure → normalize it to a twisted ``Monoid``.

First of the two tile-lowering steps — recognition here, scheduling in
``020_schedule``. It rewrites a ``LoopOp`` so every reduce carrier is the one
unified twisted ``Monoid`` representation. Three recognitions, tried in order
(each is unconditional now — no knobs):

1. **Flash attention** — a softmax-then-P@V kernel (+ its scaled-QK producer) is
   the online-softmax *monoid* over a streaming KV reduce; rewrite the pair to one
   fused flash ``LoopOp`` (the ``(m, l, O)`` twisted monoid). Graph rewrite —
   consumes the score producer.
2. **Online softmax** — an adjacent ``(rowmax, Σ exp)`` reduce pair over the same
   input fuses into one streaming online-softmax ``Monoid`` (the ``(m, d)`` twist).
3. **Normalize** — any remaining scalar ``Accum`` (a plain sum / max / mean) becomes
   its **degenerate** monoid (``Accum.as_monoid`` — the identity twist, no rescale),
   seeded by an enclosing :class:`Init`. A semiring contraction (``Semiring.match``)
   keeps its ``Accum`` — degenerate-monoidizing it would lose the contraction
   structure the matmul tier reads off the body.

Flash must precede online-softmax which must precede normalize: each later step
consumes the ``Accum``\\ s an earlier one pattern-matches. After this pass a plain
reduction, online softmax, and flash all share ONE representation — only the twist ψ
(the ``Monoid``'s ``merge`` program) differs — so the scheduler and the kernel
lowering never branch on which.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.algebra import Semiring
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Select, Write
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._flash import (
    build_flash_frag,
    build_flash_recovered,
    flash_shape_eligible,
    gqa_group,
    online_softmax_combine,
)

PATTERN = [Pattern("root", LoopOp)]


# ---------------------------------------------------------------------------
# 1. Flash attention — softmax-then-P@V (+ scaled-QK producer) → one flash LoopOp
# ---------------------------------------------------------------------------


def _is_sum(accum: Accum) -> bool:
    """The accum is the semiring additive reduce ``⊕`` (``add`` / ``sum``)."""
    return accum.op.reduce_canon == "add"


def _is_rowmax(accum: Accum) -> bool:
    """The accum is the softmax rowmax reduce (``maximum`` / ``amax``)."""
    return accum.op.reduce_canon == "maximum"


def _accum_loops(op: LoopOp) -> list[Loop]:
    """Loops whose immediate body folds an ``Accum`` (the matmul / softmax-stat reduces)."""
    return [lp for lp in op.body.iter_of_type(Loop) if any(isinstance(s, Accum) for s in lp.body)]


def _var_at(index: tuple, pos: int) -> str | None:
    """The plain axis-var name at ``index[pos]``, or None (literal / affine)."""
    if abs(pos) > len(index):
        return None
    e = index[pos]
    return e.name if isinstance(e, Var) else None


def _extract_qk(xnode: Node) -> tuple[str, str, object] | None:
    """From the scaled-QK^T producer of the score buffer, return ``(q_id, k_id,
    head_dim_extent)``. Q vs K by index (fusion reorders the operands): the matmul
    operand whose seq index equals the score's row (M) axis is Q."""
    op = xnode.op
    if not isinstance(op, LoopOp):
        return None
    writes = [s for s in op.body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    m_var = _var_at(writes[0].index, -2)  # score [..., M (query), N (kv)] → row var
    if m_var is None:
        return None
    for lp in _accum_loops(op):
        loads = [s for s in lp.body if isinstance(s, Load)]
        accs = [s for s in lp.body if isinstance(s, Accum)]
        muls = [s for s in lp.body if isinstance(s, Assign) and s.op.semiring_product]
        if len(loads) == 2 and len(accs) == 1 and _is_sum(accs[0]) and muls:
            q_id = k_id = None
            for ld in loads:
                if _var_at(ld.index, -2) == m_var:
                    q_id = ld.input
                else:
                    k_id = ld.input
            if q_id is not None and k_id is not None:
                return q_id, k_id, lp.axis.extent
    return None


def _def(stmts: tuple[Stmt, ...], name: str) -> Stmt | None:
    """The statement in ``stmts`` (one loop body, flat) that defines SSA ``name``."""
    for s in stmts:
        if isinstance(s, Load) and name in s.names:
            return s
        if isinstance(s, (Assign, Select)) and s.name == name:
            return s
    return None


def _is_loopop(graph: Graph, buf: str) -> bool:
    node = graph.nodes.get(buf)
    return node is not None and isinstance(node.op, LoopOp)


def _classify_rowmax(graph: Graph, lp: Loop) -> tuple[str, str, str | None] | None:
    """For the rowmax reduce loop, return ``(score_buf, mask_kind, mask_buf)`` where
    ``mask_kind`` is ``"none"`` / ``"causal"`` / ``"additive"``; else None. The value
    folded by the ``maximum`` Accum is the bare score Load (no mask) or
    ``add(score, mask)`` — the mask a coord ``Select`` (causal) or a buffer ``Load``."""
    max_accs = [s for s in lp.body if isinstance(s, Accum) and _is_rowmax(s)]
    if len(max_accs) != 1:
        return None
    feed = _def(lp.body, max_accs[0].value)
    if isinstance(feed, Load):
        return feed.input, "none", None
    if isinstance(feed, Assign) and feed.op.name == "add":
        a, b = feed.args
        for sc, mk in ((a, b), (b, a)):
            sdef, mdef = _def(lp.body, sc), _def(lp.body, mk)
            if isinstance(sdef, Load) and _is_loopop(graph, sdef.input):
                if isinstance(mdef, Select):
                    return sdef.input, "causal", None
                if isinstance(mdef, Load):
                    return sdef.input, "additive", mdef.input
    return None


def _recognize(graph: Graph, node: Node) -> tuple[str, str, str, str | None] | None:
    """If ``node`` is a softmax-then-P@V kernel, return ``(x_buf, v_buf, mask_kind,
    mask_buf)`` — the score buffer the rowmax reduces, the P@V's V operand, and the
    softmax-side mask (if any). Q/K recovery is left to the caller."""
    op = node.op
    if not isinstance(op, LoopOp):
        return None
    body = op.body
    if not any(isinstance(s, Assign) and s.op.name == "exp" for s in body.iter()):
        return None
    writes = [s for s in body.iter() if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    out_write = writes[0]
    x_buf: str | None = None
    mask_kind = "none"
    mask_buf: str | None = None
    for lp in _accum_loops(op):
        cls = _classify_rowmax(graph, lp)
        if cls is not None:
            x_buf, mask_kind, mask_buf = cls
            break
    if x_buf is None:
        return None
    v_buf: str | None = None
    for lp in _accum_loops(op):
        if not any(isinstance(s, Accum) and s.name == out_write.value and _is_sum(s) for s in lp.body):
            continue
        others = {s.input for s in lp.body if isinstance(s, Load)} - {x_buf, mask_buf}
        if len(others) == 1:
            v_buf = next(iter(others))
    if v_buf is None:
        return None
    return x_buf, v_buf, mask_kind, mask_buf


def _try_flash(match: Match, root: Node) -> Graph | None:
    """Recognize SDPA on ``root`` and return the fused flash ``Graph`` fragment, or
    ``None`` if ``root`` is not a recognizable / eligible attention kernel."""
    found = _recognize(match.graph, root)
    if found is None:
        return None
    x_buf, v_id, mask_kind, mask_buf = found
    graph = match.graph
    operands = (x_buf, v_id, *((mask_buf,) if mask_buf is not None else ()))
    if any(nid not in graph.nodes for nid in operands):
        return None

    # Synthetic path: a clean scaled-QK producer (Q/K recoverable as plain Loads).
    qk = _extract_qk(graph.nodes[x_buf])
    if qk is not None:
        q_id, k_id, _head_dim = qk
        if q_id not in graph.nodes or k_id not in graph.nodes:
            return None
        q_shape = graph.nodes[q_id].output.shape
        k_shape = graph.nodes[k_id].output.shape
        v_shape = graph.nodes[v_id].output.shape
        group = gqa_group(q_shape, k_shape)
        if group is None:
            return None
        mask_shape = graph.nodes[mask_buf].output.shape if mask_buf is not None else None
        if not flash_shape_eligible(q_shape, k_shape, v_shape, group=group, mask_shape=mask_shape):
            return None
        mask = (mask_buf, mask_shape) if mask_kind == "additive" else None
        return build_flash_frag(
            q_id, k_id, v_id, q_shape, k_shape, v_shape, root.output, causal=(mask_kind == "causal"), group=group, mask=mask
        )

    # Recovery path: a fused score producer (RoPE / GQA index / mask inline). The
    # producer carries its own mask, so a softmax-side mask here would double-count.
    if mask_kind != "none":
        return None
    xnode = graph.nodes[x_buf]
    if not isinstance(xnode.op, LoopOp):
        return None
    return build_flash_recovered(graph, xnode.op, root.op, x_buf, v_id, root.output)


# ---------------------------------------------------------------------------
# 2. Online softmax — fuse a (rowmax, Σ exp) reduce pair into one Monoid pass
# ---------------------------------------------------------------------------


def _rowmax(loop: Loop) -> tuple[str, str, tuple] | None:
    """``(acc, input, index)`` if ``loop`` is a row-max reduce of a single ``Load``."""
    body = list(loop.body)
    maxes = [s for s in body if isinstance(s, Accum) and s.op.reduce_canon == "maximum"]
    if len(maxes) != 1:
        return None
    acc = maxes[0]
    ld = next((s for s in body if isinstance(s, Load) and s.name == acc.value), None)
    return (acc.name, ld.input, ld.index) if ld is not None else None


def _sumexp(loop: Loop, maxacc: str, input_buf: str) -> str | None:
    """The sum ``Accum`` name if ``loop`` is a ``Σ exp(x − maxacc)`` reduce over
    ``input_buf`` — folds ``add`` over ``exp(subtract(load(input_buf, …), maxacc))``."""
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
    """Recurse into nested ``Loop`` bodies; fuse any adjacent ``(rowmax, sum-of-exp)``
    reduce pair over the same input + reduce extent into one online-softmax ``Monoid``
    loop (+ the carried-state seeds)."""
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


def _try_online_softmax(root: Node) -> LoopOp | None:
    new_body, changed = _fuse(root.op.body)
    if not changed:
        return None
    return replace(root.op, body=new_body)


# ---------------------------------------------------------------------------
# 3. Normalize — any remaining scalar Accum → its degenerate (identity-twist) Monoid
# ---------------------------------------------------------------------------


def _normalize(stmts: list[Stmt]) -> tuple[list[Stmt], bool]:
    """Rewrite each plain reduce ``Loop``'s ``Accum``\\ s to ``Init`` + degenerate
    ``Monoid`` (deep). A semiring contraction (``Semiring.match``) keeps its
    ``Accum`` — degenerate-monoidizing it would lose the contraction structure the
    matmul tier reads. Returns ``(stmts, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and Semiring.match(s) is None:
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


def rewrite(match: Match, root: Node) -> LoopOp | Graph | None:
    # Order matters: flash consumes the Accums online-softmax would match, which in
    # turn consumes the Accums normalize would convert.
    flash = _try_flash(match, root)
    if flash is not None:
        return flash
    fused = _try_online_softmax(root)
    if fused is not None:
        return fused
    new, changed = _normalize(list(root.op.body))
    if not changed:
        raise RuleSkipped("no reduce carrier to recognize or normalize")
    return replace(root.op, body=Body(tuple(new)))
