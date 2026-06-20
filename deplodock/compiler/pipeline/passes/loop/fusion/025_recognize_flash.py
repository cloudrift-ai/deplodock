"""Recognize the (fused) online-softmax attention at Loop IR → rewrite to flash.

Runs AFTER the generic fuser (``010_merge_loop_ops`` + ``020_dedup_loads``), so it
matches the **consolidated** form, not the scattered per-op chain: a non-causal
SDPA fuses to just two ``LoopOp``s — the scaled scores ``X = (Σ_dd Q·K)·scale``
and the softmax-then-P@V kernel ``out = Σ_kv softmax(X)·V`` (rowmax + rowsum-of-exp
+ the normalized P@V all in one body). That second kernel IS the online-softmax
pattern in one place, which is the semantic layer to recognize it. NO modification
to the decomposition stage; gated by the ``FLASH`` knob (off → untouched).

The matcher anchors on the softmax-P@V kernel (its body carries the tell-tale
``maximum`` rowmax Accum + an ``exp`` + a P@V sum feeding the output), reads ``X``
(the score buffer the rowmax reduces) and ``V`` (the P@V's non-``X`` operand), then
traces ``X`` to its producer (the scaled QK^T) to read Q / K. Operand order is NOT
preserved by fusion, so Q vs K is disambiguated by index: the QK operand whose seq
index matches the score's row (M / query) axis is Q. On a match the whole pair is
rewritten to the fused flash ``LoopOp`` (``_flash.build_flash_frag``); the scores
kernel orphans and is removed. Runs before ``991``/``992`` so the new kernel is
named + structurally stamped. Scope: non-causal, static or dynamic seq, no mask,
no GQA — anything else fails a check and the score-matrix path stands.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.loop.fusion._flash import build_flash_frag, flash_enabled, flash_shape_eligible

PATTERN = [Pattern("root", LoopOp)]

_SUM = ("add", "sum")
_MAX = ("maximum", "amax")


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
        muls = [s for s in lp.body if isinstance(s, Assign) and s.op.name == "multiply"]
        if len(loads) == 2 and len(accs) == 1 and accs[0].op.name in _SUM and muls:
            q_id = k_id = None
            for ld in loads:
                if _var_at(ld.index, -2) == m_var:
                    q_id = ld.input
                else:
                    k_id = ld.input
            if q_id is not None and k_id is not None:
                return q_id, k_id, lp.axis.extent
    return None


def _recognize(graph: Graph, node: Node) -> tuple[str, str, str] | None:
    """If ``node`` is a softmax-then-P@V kernel over a scaled QK^T, return
    ``(q_id, k_id, v_id)``; else None. Non-causal only."""
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

    # X = the score buffer the rowmax reduces (the softmax tell-tale).
    x_buf: str | None = None
    for lp in _reduce_loops(op):
        accs = [s for s in lp.body if isinstance(s, Accum)]
        loads = [s for s in lp.body if isinstance(s, Load)]
        if any(a.op.name in _MAX for a in accs) and len(loads) == 1:
            x_buf = loads[0].input
    if x_buf is None:
        return None

    # P@V: the reduce whose sum-Accum feeds the output; V = its non-X operand.
    v_buf: str | None = None
    for lp in _reduce_loops(op):
        if not any(isinstance(s, Accum) and s.name == out_write.value and s.op.name in _SUM for s in lp.body):
            continue
        others = {s.input for s in lp.body if isinstance(s, Load)} - {x_buf}
        if len(others) == 1:
            v_buf = next(iter(others))
    if v_buf is None:
        return None

    xnode = graph.nodes.get(x_buf)
    if xnode is None:
        return None
    qk = _extract_qk(xnode)
    if qk is None:
        return None
    q_id, k_id, _head_dim = qk
    return q_id, k_id, v_buf


def rewrite(match: Match, root: Node) -> Graph | None:
    if not flash_enabled():
        raise RuleSkipped("FLASH knob off — keep the score-materializing path")
    found = _recognize(match.graph, root)
    if found is None:
        raise RuleSkipped("not a (non-causal) softmax-attention kernel")
    q_id, k_id, v_id = found
    graph = match.graph
    if any(nid not in graph.nodes for nid in (q_id, k_id, v_id)):
        raise RuleSkipped("flash: Q/K/V operand not a graph node")
    q_shape = graph.nodes[q_id].output.shape
    k_shape = graph.nodes[k_id].output.shape
    v_shape = graph.nodes[v_id].output.shape
    if not flash_shape_eligible(q_shape, k_shape, v_shape, has_mask=False):
        raise RuleSkipped("flash: SDPA shape not eligible (GQA / symbolic non-seq)")
    return build_flash_frag(q_id, k_id, v_id, q_shape, k_shape, v_shape, root.output, causal=False)
