"""Recognize the (fused) online-softmax attention at Loop IR → rewrite to flash.

This is a **pattern-recognition** pass (``loop/recognize``): it rewrites a generic
fused op-cluster into a specialized fused kernel. It runs AFTER the entire
``loop/fusion`` fixpoint has settled — not interleaved with the generic fuser. The
placement is load-bearing: the matmul's elementwise product (``qk_ew``) and the
score reduce start as separate ops out of decomposition, and ``010_merge_loop_ops``
fuses them (RoPE inline) over several fixpoint iterations. Firing recognition
*interleaved* catches the score producer mid-fusion (``qk_ew`` still a separate
kernel) and inlines that split form, re-materializing the RoPE'd product — defeating
flash's whole purpose. Recognizing only once fusion settles, the recovered score
body reads the projections directly (RoPE inline in the streaming reduce). Naming +
structural stamping then run in the shared ``loop/stamp`` pass (after both
``loop/fusion`` and ``loop/recognize``), so the minted flash kernel is named /
stamped like any other. NO modification to the decomposition stage; gated by the
``FLASH`` knob (off → untouched). Future recognizers (other attention variants,
fused-norm patterns) belong in this pass too.

A non-causal SDPA fuses to two ``LoopOp``s — the scaled scores ``X = (Σ_dd
Q·K)·scale`` and the softmax-then-P@V kernel ``out = Σ_kv softmax(X)·V`` (rowmax +
rowsum-of-exp + the normalized P@V all in one body). That second kernel IS the
online-softmax pattern in one place, which is the semantic layer to recognize it.

The matcher anchors on the softmax-P@V kernel (its body carries the tell-tale
``maximum`` rowmax Accum + an ``exp`` + a P@V sum feeding the output), reads ``X``
(the score buffer the rowmax reduces) and ``V`` (the P@V's non-``X`` operand). Two
score-recovery strategies: **synthetic** when ``X``'s producer is a clean scaled-QK
(``_extract_qk`` reads Q / K as plain Loads, disambiguated by index — the operand
whose seq index matches the score's row/M axis is Q), and **recovered**
(``build_flash_recovered``) when the producer is fused (RoPE / GQA index / scale /
mask inline, so Q/K are computed SSA values not Loads — real decoder layers): the
producer's score body is inlined wholesale and the consumer's V-load / output
indices are recovered. On a match the pair is rewritten to one fused flash
``LoopOp``; the scores kernel orphans and is removed.

Masking and GQA are recovered **structurally** from the fused body (no frontend
provenance to lean on): the score feeding the rowmax ``Accum`` is either the bare
score Load (no mask), ``add(score, Select(kv ≤ m))`` (causal — the lifted
``IndexMapOp`` bias), or ``add(score, Load(mask))`` (the HF ``(1,1,S,S)`` additive
bias); the GQA group is the ``q_heads // kv_heads`` shape ratio, deployed as a
``head // group`` K/V index. Detecting the mask is a correctness requirement — a
masked SDPA that matched the anchor but built an unmasked nest would be silently
wrong. Anything ineligible (symbolic non-seq, non-broadcastable mask, indivisible
heads) fails a check and the score-matrix path stands.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Select, Stmt, Write
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.recognize._flash import (
    build_flash_frag,
    build_flash_recovered,
    flash_enabled,
    flash_shape_eligible,
    gqa_group,
)

PATTERN = [Pattern("root", LoopOp)]


def _is_sum(accum: Accum) -> bool:
    """The accum is the semiring additive reduce ``⊕`` (``add`` / ``sum``)."""
    return accum.op.reduce_canon == "add"


def _is_rowmax(accum: Accum) -> bool:
    """The accum is the softmax rowmax reduce (``maximum`` / ``amax``)."""
    return accum.op.reduce_canon == "maximum"


def _reduce_loops(op: LoopOp) -> list[Loop]:
    """Loops whose immediate body folds an ``Accum`` (the matmul / softmax-stat
    reduces)."""
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
    for lp in _reduce_loops(op):
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


def _classify_rowmax(graph: Graph, lp: Loop) -> tuple[str, str, str | None] | None:
    """For the rowmax reduce loop, return ``(score_buf, mask_kind, mask_buf)``
    where ``mask_kind`` is ``"none"`` / ``"causal"`` / ``"additive"``; else None.

    The value folded by the ``maximum`` Accum is the bare score Load (no mask) or
    ``add(score, mask)`` — the mask being a coord ``Select`` (causal) or a buffer
    ``Load`` (the additive bias). The score side is the operand whose buffer is a
    ``LoopOp`` (the scaled-QK producer); the mask side is the other."""
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


def _is_loopop(graph: Graph, buf: str) -> bool:
    node = graph.nodes.get(buf)
    return node is not None and isinstance(node.op, LoopOp)


def _recognize(graph: Graph, node: Node) -> tuple[str, str, str, str | None] | None:
    """If ``node`` is a softmax-then-P@V kernel, return the anchor pieces
    ``(x_buf, v_buf, mask_kind, mask_buf)`` — the score buffer the rowmax reduces,
    the P@V's V operand, and the softmax-side mask (if any). Q/K recovery is left
    to the caller (``_extract_qk`` for the synthetic build, else recovery)."""
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

    # X = the score buffer the rowmax reduces; mask recovered alongside.
    x_buf: str | None = None
    mask_kind = "none"
    mask_buf: str | None = None
    for lp in _reduce_loops(op):
        cls = _classify_rowmax(graph, lp)
        if cls is not None:
            x_buf, mask_kind, mask_buf = cls
            break
    if x_buf is None:
        return None

    # P@V: the reduce whose sum-Accum feeds the output; V = its non-X/-mask operand.
    v_buf: str | None = None
    for lp in _reduce_loops(op):
        if not any(isinstance(s, Accum) and s.name == out_write.value and _is_sum(s) for s in lp.body):
            continue
        others = {s.input for s in lp.body if isinstance(s, Load)} - {x_buf, mask_buf}
        if len(others) == 1:
            v_buf = next(iter(others))
    if v_buf is None:
        return None
    return x_buf, v_buf, mask_kind, mask_buf


def _composer_wants_flash(root: Node) -> bool:
    """The move composer implies flash ONLY for symbolic-seq SDPA: static SDPA
    decomposes (the composer covers the static QK^T / softmax / P@V kernels, and
    the loop reference backend runs them), while a symbolic seq needs flash's
    masked streaming (the composer's matmul tier requires a static K). A symbolic
    seq shows up as a symbolic dim ANYWHERE in the attention output (the
    whole-model o_proj collapses the attn-out, so seq isn't always ``shape[-2]``);
    a fully-static output is a static SDPA."""
    return any(not getattr(d, "is_static", True) for d in root.output.shape)


def rewrite(match: Match, root: Node) -> Graph | None:
    if not flash_enabled() and not _composer_wants_flash(root):
        raise RuleSkipped("FLASH off (composer keeps static SDPA decomposed) — score-materializing path")
    found = _recognize(match.graph, root)
    if found is None:
        raise RuleSkipped("not a softmax-attention kernel")
    x_buf, v_id, mask_kind, mask_buf = found
    graph = match.graph
    operands = (x_buf, v_id, *((mask_buf,) if mask_buf is not None else ()))
    if any(nid not in graph.nodes for nid in operands):
        raise RuleSkipped("flash: score/V/mask operand not a graph node")

    # Synthetic path: a clean scaled-QK producer (Q/K recoverable as plain Loads).
    qk = _extract_qk(graph.nodes[x_buf])
    if qk is not None:
        q_id, k_id, _head_dim = qk
        if q_id not in graph.nodes or k_id not in graph.nodes:
            raise RuleSkipped("flash: Q/K operand not a graph node")
        q_shape = graph.nodes[q_id].output.shape
        k_shape = graph.nodes[k_id].output.shape
        v_shape = graph.nodes[v_id].output.shape
        group = gqa_group(q_shape, k_shape)
        if group is None:
            raise RuleSkipped("flash: GQA heads not statically divisible")
        mask_shape = graph.nodes[mask_buf].output.shape if mask_buf is not None else None
        if not flash_shape_eligible(q_shape, k_shape, v_shape, group=group, mask_shape=mask_shape):
            raise RuleSkipped("flash: SDPA shape not eligible (GQA / mask / symbolic non-seq)")
        mask = (mask_buf, mask_shape) if mask_kind == "additive" else None
        return build_flash_frag(
            q_id, k_id, v_id, q_shape, k_shape, v_shape, root.output, causal=(mask_kind == "causal"), group=group, mask=mask
        )

    # Recovery path: a fused score producer (RoPE / GQA index / mask inline) the
    # synthetic builder can't reconstruct. Inline the producer's score body and
    # recover the consumer's V / output indices. The producer carries its own
    # mask, so a softmax-side mask here would be double-counted — fall through.
    if mask_kind != "none":
        raise RuleSkipped("flash: softmax-side mask with a non-clean QK producer")
    xnode = graph.nodes[x_buf]
    if not isinstance(xnode.op, LoopOp):
        raise RuleSkipped("flash: score producer is not a LoopOp")
    frag = build_flash_recovered(graph, xnode.op, root.op, x_buf, v_id, root.output)
    if frag is None:
        raise RuleSkipped("flash: score producer not recoverable")
    return frag
