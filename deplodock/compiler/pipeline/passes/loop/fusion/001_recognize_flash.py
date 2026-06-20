"""Recognize the decomposed online-softmax attention at Loop IR → fuse to flash.

NO modification to the decomposition stage: ``010_sdpa`` decomposes SDPA into the
QK^T → scale → softmax → P@V tensor ops as always, lifting turns them into the
canonical per-op ``LoopOp`` chain, and THIS pass (running before the generic
fuser ``010_merge_loop_ops``) pattern-matches that chain and rewrites it into the
single fused streaming kernel built by ``_flash.build_flash_frag``. Gated by the
``FLASH`` knob; off → the score-materializing path is untouched.

The matched chain (canonical ``matmul_decompose`` / ``softmax_decompose`` output,
operands in A-then-B order)::

    out = squeeze( Σ_kv  probs ⊙ V )                         # P@V (the anchor)
    probs = exp(scores' − rowmax) / rowsum(exp(scores' − rowmax))   # softmax
    scores' = ( Σ_dd Q ⊙ K ) · scale                        # QK^T, scaled

Recognition walks BACKWARD from the terminal squeeze through structural
classifiers (``_as_copy`` / ``_as_reduce`` / ``_as_binary`` / ``_as_unary``),
peeling broadcast/unsqueeze copies, and reads Q / K / V off the two matmuls'
leaf operands. Scope: non-causal, static or dynamic seq, no explicit mask, no GQA
— anything else fails a classifier and falls through to the score-matrix path
(causal / mask / GQA are follow-ups). See ``_flash`` for the nest + scope.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.loop.fusion._flash import build_flash_frag, flash_enabled, flash_shape_eligible

PATTERN = [Pattern("root", LoopOp)]


# --- structural classifiers over a single LoopOp body ------------------------


def _leaves(op: LoopOp, ty: type) -> list:
    return [s for s in op.body.iter() if isinstance(s, ty)]


def _as_copy(op: LoopOp) -> str | None:
    """Source buffer if ``op`` is a pure copy / broadcast / squeeze — one Load,
    no Assign, every Write storing the loaded value verbatim."""
    loads, assigns, writes = _leaves(op, Load), _leaves(op, Assign), _leaves(op, Write)
    if assigns or len(loads) != 1 or not writes:
        return None
    if not all(w.is_scalar and w.value == loads[0].name for w in writes):
        return None
    return loads[0].input


def _as_unary(op: LoopOp) -> tuple[str, str] | None:
    """``(op_name, src_buf)`` if ``op`` is one elementwise Assign of a single
    loaded operand (e.g. ``exp``)."""
    loads, assigns = _leaves(op, Load), _leaves(op, Assign)
    by_name = {s.name: s.input for s in loads}
    if len(assigns) != 1 or len(assigns[0].args) != 1 or assigns[0].args[0] not in by_name:
        return None
    return assigns[0].op.name, by_name[assigns[0].args[0]]


def _as_binary(op: LoopOp) -> tuple[str, str, str] | None:
    """``(op_name, a_buf, b_buf)`` if ``op`` is one elementwise Assign over two
    loaded operands, in argument order (A then B)."""
    loads, assigns = _leaves(op, Load), _leaves(op, Assign)
    by_name = {s.name: s.input for s in loads}
    if len(assigns) != 1 or len(assigns[0].args) != 2:
        return None
    if any(a not in by_name for a in assigns[0].args):
        return None
    return assigns[0].op.name, by_name[assigns[0].args[0]], by_name[assigns[0].args[1]]


def _as_reduce(op: LoopOp) -> tuple[str, str, object] | None:
    """``(combine_op, src_buf, reduce_extent)`` if ``op``'s innermost reduce Loop
    folds a single loaded operand (a matmul-style ``Σ`` or a softmax stat)."""
    for lp in op.body.iter_of_type(Loop):
        accums = [s for s in lp.body if isinstance(s, Accum)]
        loads = [s for s in lp.body if isinstance(s, Load)]
        if len(accums) == 1 and len(loads) == 1:
            return accums[0].op.name, loads[0].input, lp.axis.extent
    return None


# --- backward walk -----------------------------------------------------------


def _prod(graph: Graph, buf: str) -> Node | None:
    """The node producing ``buf`` (buffer names == producer node ids)."""
    return graph.nodes.get(buf)


def _peel(graph: Graph, buf: str) -> str:
    """Follow copy / broadcast LoopOps back to the first non-copy producer."""
    seen: set[str] = set()
    while buf not in seen:
        seen.add(buf)
        node = _prod(graph, buf)
        if node is None or not isinstance(node.op, LoopOp):
            return buf
        src = _as_copy(node.op)
        if src is None:
            return buf
        buf = src
    return buf


def _reduce_node(graph: Graph, buf: str) -> tuple[str, str, object] | None:
    node = _prod(graph, buf)
    return _as_reduce(node.op) if node is not None and isinstance(node.op, LoopOp) else None


def _binary_node(graph: Graph, buf: str) -> tuple[str, str, str] | None:
    node = _prod(graph, buf)
    return _as_binary(node.op) if node is not None and isinstance(node.op, LoopOp) else None


def _unary_node(graph: Graph, buf: str) -> tuple[str, str] | None:
    node = _prod(graph, buf)
    return _as_unary(node.op) if node is not None and isinstance(node.op, LoopOp) else None


def _recognize(graph: Graph, terminal: Node) -> tuple[str, str, str] | None:
    """If ``terminal`` is the squeeze of a P@V whose probs come from a softmax over
    a scaled QK^T, return ``(q_id, k_id, v_id)``; else None. Non-causal only."""
    if not isinstance(terminal.op, LoopOp):
        return None
    # Anchor: the terminal must be a squeeze copy of the P@V reduce (this is what
    # disambiguates it from the P@V reduce itself, which carries a keepdim).
    pv_buf = _as_copy(terminal.op)
    if pv_buf is None:
        return None
    pv = _reduce_node(graph, pv_buf)  # P@V: Σ_kv (probs ⊙ V)
    if pv is None or pv[0] not in ("add", "sum"):
        return None
    pv_ew = _binary_node(graph, pv[1])  # probs ⊙ V, args (A=probs, B=V)
    if pv_ew is None or pv_ew[0] != "multiply":
        return None
    probs_buf, v_id = _peel(graph, pv_ew[1]), _peel(graph, pv_ew[2])

    # Softmax: probs = exp(shifted) / rowsum(exp(shifted)), shifted = scores' − max.
    sm = _binary_node(graph, probs_buf)
    if sm is None or sm[0] != "divide":
        return None
    exp = _unary_node(graph, _peel(graph, sm[1]))
    if exp is None or exp[0] != "exp":
        return None
    shifted = _binary_node(graph, exp[1])  # subtract(scores', max_bc)
    if shifted is None or shifted[0] != "subtract":
        return None
    scaled = _binary_node(graph, _peel(graph, shifted[1]))  # scores' = qk · scale
    if scaled is None or scaled[0] != "multiply":
        return None

    # QK^T: Σ_dd (Q ⊙ K), args (A=Q, B=K).
    qk = _reduce_node(graph, _peel(graph, scaled[1]))
    if qk is None or qk[0] not in ("add", "sum"):
        return None
    qk_ew = _binary_node(graph, qk[1])
    if qk_ew is None or qk_ew[0] != "multiply":
        return None
    q_id, k_id = _peel(graph, qk_ew[1]), _peel(graph, qk_ew[2])
    return q_id, k_id, v_id


def rewrite(match: Match, root: Node) -> Graph | None:
    if not flash_enabled():
        raise RuleSkipped("FLASH knob off — keep the score-materializing path")
    found = _recognize(match.graph, root)
    if found is None:
        raise RuleSkipped("not the terminal of a (non-causal) online-softmax attention")
    q_id, k_id, v_id = found
    graph = match.graph
    for nid in (q_id, k_id, v_id):
        if nid not in graph.nodes:
            raise RuleSkipped(f"flash: operand {nid!r} not a graph node")
    q_shape = graph.nodes[q_id].output.shape
    k_shape = graph.nodes[k_id].output.shape
    v_shape = graph.nodes[v_id].output.shape
    if not flash_shape_eligible(q_shape, k_shape, v_shape, has_mask=False):
        raise RuleSkipped("flash: SDPA shape not eligible (GQA / symbolic non-seq)")
    return build_flash_frag(q_id, k_id, v_id, q_shape, k_shape, v_shape, root.output, causal=False)
