"""Flash-attention helper — recognition and construction.

The ``010_recognize`` pass calls :func:`try_flash`; everything flash lives here. Two
halves:

- **Recognition** — :func:`try_flash` matches a softmax-then-P@V kernel (+ its clean
  scaled-QK producer) via ``_recognize`` / ``_extract_qk`` / ``_classify_rowmax``, and
  emits the fused fragment.
- **Construction** — the ``flash_combine`` carrier, the ``flash_shape_eligible`` /
  ``gqa_group`` predicates, and the fragment builder ``build_flash_frag``. It doesn't
  hand-assemble a kernel body — it builds the high-level **structural-node tree**
  (``ir/tile/ir``): flash is ``Map(body=[O/l projection], source=Reduction(role=TWISTED,
  axis=kv, source=Contraction(Σ Q·K)))`` — the ``(m,l,O)`` LSE streaming reduce over kv whose
  per-step partial folds a nested ``Σ_dd Q·K`` :class:`Contraction` node (the ``Reduction ⊃
  Contraction`` composition), projected ``O/l`` after the loop. ``build_flash_frag`` returns that
  ``Map`` UNLOWERED, on a ``TileOp`` with an UNMAPPED ``Placement`` — the free ``(batch…, m, d)``
  axes are the schedule's (like every recognizer), and ``_schedule`` maps them onto the grid;
  ``materialize`` lowers the nodes (``Reduction.loop`` splices the contraction's ``Σ Q·K`` loop
  ahead of the partial, the scalar tier expanding the same loop nest) + generates the output-store
  glue (the ``Write`` at the grid cell — not stored). Warp-flash is just the inner ``Contraction``
  gaining a warp ``TilePlan`` (routed through ``_factor``) — no new path, the nesting carries it.

``is_flash_score_producer`` lets ``010_recognize`` defer the general lift of a scaled-QK
score producer until its softmax-then-P@V consumer has fused (the fusion reads the
producer's Q/K as plain ``Load``\\ s, so it must stay a ``LoopOp`` until then).

(Online softmax — flash's softmax-stats half without the P@V — lives in ``_softmax``.)

Scope is the **clean** scaled-QK producer (Q/K recoverable as plain ``Load``\\ s). A
fused score producer whose Q/K are computed SSA — RoPE'd QK — is NOT recognized as
flash (it falls back to its un-fused tiers); a producer-splicing builder for that case
was removed rather than kept half-converted to the op tree.

The fragment fuses scaled-dot-product attention into ONE kernel that tiles the KV
(reduce) axis and never materializes the ``[S_q, S_k]`` score matrix. It runs one
independent streaming softmax per output element ``(…, m, d)`` — a correct, if
redundant, scalar form; the tensor-core P@V tier is future work::

    for *batch, m (query rows), d (value dim):       # free / grid
      Init (m_i = -inf, l_i = 0, O_i = 0)            # running (max, denom, out)
      for kv in 0..S_k:                              # streaming reduce
        Init sacc = 0
        for dd in 0..head_dim: sacc += Q[…,m,dd]·K[…,kv,dd]   # score reduce
        s = sacc · scale
        Carrier((m_i,l_i,O_i), (s, V[…,kv,d]))      # the LSE rescale (flash_combine)
      out[…,m,d] = O_i / l_i

Scope: static OR dynamic (symbolic ``seq_len`` on Q/K/V dim -2 — one cached kernel
carrying ``int seq_len`` serves every runtime size, the symbol landing on BOTH the
masked-row M and the symbolic reduce), causal or non-causal (causal masks the
score per element, ``kv ≤ m`` else −inf — tile-skip is a tensor-core-tier
follow-up), an optional broadcast additive mask (the HF ``(1,1,S,S)`` float bias),
and GQA (``q_heads == group · kv_heads``; the K/V head axis read at ``head //
group`` directly, no materialized broadcast). The tensor-core P@V tier is future
work. Read from the ``DEPLODOCK_FLASH=1`` env pin; the two-level ``OptionFork``
offer + ``AnalyticPrior`` cold-start are a follow-up.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Carrier, Load, Loop, Select, SelectBranch, Write
from deplodock.compiler.ir.tile import Contraction, Placement, Reduction, TileOp, TilePlan
from deplodock.compiler.ir.tile.ops import Map
from deplodock.compiler.pipeline.passes.lowering.tile._carrier import denom, exp_family_twist, expect

if TYPE_CHECKING:
    from deplodock.compiler.graph import Node
    from deplodock.compiler.ir.stmt.base import Stmt


def flash_combine(m: str, ll: str, o: str, s: str, v: str) -> Carrier:
    """The online-softmax (log-sum-exp) :class:`Carrier` for one streaming KV step: state
    ``(m, l, O)`` folds this key's ``(score, value)`` partial::

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        l = l·alpha + p;     O = O·alpha + p·v;         m = m_new   (last)

    Built as a name-free exp-family **spec** (``pivot``, ``denom``, ``expect``) — the streaming
    ``merge`` and the cross-partition ``combine_states`` are *generated* from the spec (and the
    stable max-rescale recovered by the stabilizer), not authored here. See
    ``ir/stmt/carrier.py``. Rides on the ``kv`` reduce ``Loop`` (``loop.carrier``); the φ
    projection ``O/l`` is a stmt after that loop in the ``Map`` body, added in ``_flash_op``."""
    return exp_family_twist(s, [denom(), expect(v)], (m, ll, o))


def _static(d) -> int | None:
    """The static extent of a ``Dim``, or ``None`` when symbolic."""
    return d.as_static() if d.is_static else None


def gqa_group(q_shape: tuple, k_shape: tuple) -> int | None:
    """The grouped-query head ratio ``q_heads // kv_heads`` (1 when equal-head),
    or ``None`` when the head axis isn't statically divisible. The head axis is the
    last batch dim (``shape[-3]``); rank < 3 has no head (group 1)."""
    qh = _static(q_shape[-3]) if len(q_shape) >= 3 else 1
    kh = _static(k_shape[-3]) if len(k_shape) >= 3 else 1
    if qh is None or kh is None or kh == 0 or qh % kh != 0:
        return None
    return qh // kh


def flash_shape_eligible(q_shape: tuple, k_shape: tuple, v_shape: tuple, *, group: int, mask_shape: tuple | None) -> bool:
    """True iff the flash nest can serve this SDPA — static batch/head (only the
    seq axis may be symbolic), an optional broadcastable additive mask, and GQA
    where ``q_heads == group · kv_heads``. The K/V head axis is read at
    ``head // group`` directly in the nest (no materialized broadcast). The
    recognizer and this predicate MUST agree, so both call it."""
    if len(q_shape) < 2 or len(k_shape) < 2 or len(v_shape) < 2:
        return False
    q_batch = [_static(d) for d in q_shape[:-2]]
    k_batch = [_static(d) for d in k_shape[:-2]]
    v_batch = [_static(d) for d in v_shape[:-2]]
    if any(b is None for b in (*q_batch, *k_batch, *v_batch)):
        return False  # symbolic batch / head — only the seq axis may be dynamic
    if len(q_batch) != len(k_batch) or len(q_batch) != len(v_batch):
        return False
    if q_batch:
        # Leading (non-head) batch dims must match exactly; the head axis (last
        # batch dim) is q = group · kv.
        if q_batch[:-1] != k_batch[:-1] or q_batch[:-1] != v_batch[:-1]:
            return False
        if k_batch[-1] != v_batch[-1] or q_batch[-1] != group * k_batch[-1]:
            return False
    elif group != 1:
        return False  # no head axis but a non-trivial group makes no sense
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    if head_dim is None or d_v is None:
        return False  # symbolic head_dim / value-dim
    if _static(k_shape[-1]) != head_dim:
        return False
    if v_shape[-2] != k_shape[-2]:  # V seq must match K seq
        return False
    if mask_shape is not None:
        # Per-(m, kv) additive bias: leading dims must be static 1 (indexed to 0),
        # the trailing two address the query / key seq.
        if len(mask_shape) < 2:
            return False
        if any(_static(d) != 1 for d in mask_shape[:-2]):
            return False
        if mask_shape[-2] != q_shape[-2] or mask_shape[-1] != k_shape[-2]:
            return False
    return True


def build_flash_frag(
    q_id: str,
    k_id: str,
    v_id: str,
    q_shape: tuple,
    k_shape: tuple,
    v_shape: tuple,
    out: Tensor,
    *,
    causal: bool,
    group: int = 1,
    mask: tuple[str, tuple] | None = None,
) -> Graph:
    """Build the fragment graph holding the fused flash ``TileOp`` (+ its scale /
    -inf constants). The caller guarantees :func:`flash_shape_eligible`.

    The compute is the op tree itself — a ``Map`` whose body is the ``(m,l,O)`` LSE
    ``TWISTED`` reduce ``Loop`` then the ``O/l`` projection, carried unlowered on the ``TileOp`` with an empty
    schedule; the free ``(batch…, m, d)`` axes are the ``Schedule``'s ``free`` (no
    free-axis loop nest). ``020_schedule`` maps them onto the grid; ``materialize`` lowers
    the node and generates the output-store glue (the ``Write`` at the grid cell) — it
    isn't stored here.

    ``group`` is the GQA head ratio (K/V indexed at ``head // group``); ``mask`` is
    an optional ``(buffer_id, shape)`` additive bias loaded per ``(m, kv)``."""
    batch = [_static(d) for d in q_shape[:-2]]
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    s_q_dim, s_k_dim = q_shape[-2], k_shape[-2]  # Dim instances — static int or symbolic seq_len
    scale = 1.0 / math.sqrt(head_dim)
    mask_buf, mask_shape = mask if mask is not None else (None, None)

    flash_op = _flash_op(
        q_id, k_id, v_id, batch, s_q_dim, s_k_dim, head_dim, causal=causal, group=group, mask_buf=mask_buf, mask_shape=mask_shape
    )
    grid = (
        *(Axis(name=f"b{i}", extent=Dim(b)) for i, b in enumerate(batch)),
        Axis(name="m", extent=s_q_dim),
        Axis(name="d", extent=Dim(d_v)),
    )
    # The free axes are the schedule's, carried on the ``TileOp`` with an UNMAPPED grid —
    # like every other recognizer; ``020_schedule`` maps ``free`` onto the grid. Flash's outermost
    # reduce is the ``TWISTED`` kv loop (a nested ``CONTRACTION`` score loop inside it); its
    # cooperative-KV / warp tier is future work, so it keeps the scalar (serial) reduce partition.
    tile = TileOp(op=flash_op, place=Placement(free=grid))

    frag = Graph()
    for nid, shp in ((q_id, q_shape), (k_id, k_shape), (v_id, v_shape)):
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(nid, shp, out.dtype), node_id=nid)
    inputs = [q_id, k_id, v_id, "_flash_scale"]
    frag.add_node(
        op=ConstantOp(name="_flash_scale", value=scale), inputs=[], output=Tensor("_flash_scale", (1,), out.dtype), node_id="_flash_scale"
    )
    if mask_buf is not None:
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(mask_buf, mask_shape, out.dtype), node_id=mask_buf)
        inputs.append(mask_buf)
    if causal:
        # -inf bias for masked (key-after-query) positions: exp(-inf)=0, so a
        # masked score contributes nothing to the streaming softmax / output.
        frag.add_node(
            op=ConstantOp(name="_flash_ninf", value=-1e30), inputs=[], output=Tensor("_flash_ninf", (1,), out.dtype), node_id="_flash_ninf"
        )
        inputs.append("_flash_ninf")
    frag.add_node(op=tile, inputs=inputs, output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


def _batch_vars(n: int) -> tuple[Var, ...]:
    return tuple(Var(f"b{i}") for i in range(n))


def _flash_op(
    q_buf: str,
    k_buf: str,
    v_buf: str,
    batch: list[int],
    s_q: Dim,
    s_k: Dim,
    head_dim: int,
    *,
    causal: bool = False,
    group: int = 1,
    mask_buf: str | None = None,
    mask_shape: tuple | None = None,
) -> Map:
    """The per-output-element ``(…, m, d)`` compute as the structural-node tree: flash is
    ``Map(body=[O/l projection], source=Reduction(role=TWISTED, axis=kv, source=Contraction(QK)))`` —
    the ``(m,l,O)`` LSE streaming reduce over ``kv`` whose per-step **partial** folds a NESTED
    ``Σ_dd Q·K`` :class:`Contraction` node (then scaled, optionally masked, the value read + the
    carrier's dissolved merge), projected ``O/l`` after the loop. The ``Reduction ⊃ Contraction``
    composition: :meth:`Reduction.loop` splices the contraction's synthesized ``Σ Q·K`` loop ahead of
    the partial, so the scalar tier expands the same loop-in-body nest as before — and warp-flash is
    just the inner ``Contraction`` gaining a warp ``TilePlan`` (routed through ``_factor``), no new path.
    The free ``(batch…, m, d)`` axes are the ``TileOp``'s grid, not loops here; the output store is glue
    generated at materialize.

    GQA: the K/V head axis (last batch dim) is read at ``head // group``, the same
    ``//group`` the upstream ``IndexMapOp`` encodes, moved into the load index so the
    kv_heads-many K/V are read without materializing the q_heads expansion. An additive
    ``mask_buf`` (broadcast leading dims) is added to the score; causal masking is a
    ``Select`` stmt (``kv ≤ m`` else −inf) in the score ``Map`` — the index predicate
    lives in the op tree, never in the carrier. Both make ``exp(s − m_new) = 0``, so
    masked keys contribute nothing."""
    bvars = _batch_vars(len(batch))
    head_axis = len(batch) - 1  # last batch dim is the head (when there is one)
    kv_bvars = tuple(BinaryExpr("/", bv, Literal(group, "int")) if (group > 1 and i == head_axis) else bv for i, bv in enumerate(bvars))
    q_idx = (*bvars, Var("m"), Var("dd"))
    k_idx = (*kv_bvars, Var("kv"), Var("dd"))
    v_idx = (*kv_bvars, Var("kv"), Var("d"))

    # s = Σ_dd Q·K — the inner contraction as a high-level :class:`Contraction` structural node, the
    # ``source`` of the streaming kv :class:`Reduction`. Per-cell scalar (``TilePlan()``) today — the
    # redundant one-dot-per-output-element score; a warp ``TilePlan`` here is warp-flash (the same
    # ``_factor`` a standalone matmul takes). Its output axes are the score matrix ``[m, kv]`` (untiled
    # today; the m/kv geometry is only read by the warp expander). The scale / mask reads ``sacc``.
    score_contraction = Contraction(
        axes=(Axis(name="m", extent=s_q), Axis(name="kv", extent=s_k)),
        k_axis=Axis(name="dd", extent=Dim(head_dim)),
        a_load=Load(name="q_e", input=q_buf, index=q_idx),
        b_load=Load(name="k_e", input=k_buf, index=k_idx),
        acc="sacc",
        tile=TilePlan(),
    )
    score_post = [
        Load(name="scale_c", input="_flash_scale", index=()),
        Assign(name="s", op="multiply", args=("sacc", "scale_c")),
    ]
    if causal:
        # Causal mask: keep the score where key ≤ query (kv ≤ m), else −inf.
        score_post += [
            Load(name="ninf_c", input="_flash_ninf", index=()),
            Select(
                name="s_masked",
                branches=(
                    SelectBranch(value="s", select=BinaryExpr("<=", Var("kv"), Var("m"))),
                    SelectBranch(value="ninf_c", select=Literal(1, "int")),
                ),
            ),
        ]
        score_name = "s_masked"
    elif mask_buf is not None:
        # Additive bias: leading dims broadcast (indexed to 0), trailing two are the
        # query row m and the streaming key kv.
        mask_idx = (*(Literal(0, "int") for _ in mask_shape[:-2]), Var("m"), Var("kv"))
        score_post += [Load(name="mask_e", input=mask_buf, index=mask_idx), Assign(name="s_masked", op="add", args=("s", "mask_e"))]
        score_name = "s_masked"
    else:
        score_name = "s"
    # The (m,l,O) streaming fold over kv, as a ``TWISTED`` :class:`Reduction` node: its per-step
    # ``partial`` is the scale / mask binding ``score_name``, the value ``Load``, then the carrier's
    # dissolved streaming ``merge`` (``flash_combine`` supplies state + twist); the nested ``Σ Q·K``
    # contraction rides on ``source`` (``Reduction.loop`` splices its loop ahead of the partial).
    carrier = flash_combine("m_i", "l_i", "O_i", score_name, "v_e")
    partial = [*score_post, Load(name="v_e", input=v_buf, index=v_idx), *carrier.dissolve()]
    reduction = Reduction(
        carrier=carrier,
        axis=Axis(name="kv", extent=s_k),
        partial=Body(tuple(partial)),
        role=AxisRole.TWISTED,
        source=score_contraction,
    )
    # φ projection: normalize the streamed (unnormalized) output by the LSE denominator —
    # O_i / l_i after the kv loop, the ``Map`` body over the reduction ``source``.
    proj = Assign(name="O_i__proj", op="divide", args=("O_i", "l_i"))
    return Map(body=Body((proj,)), source=reduction)


# --------------------------------------------------------------------------- #
# Recognition — match a softmax-then-P@V kernel (+ its clean scaled-QK producer)
# and emit the fused flash fragment. Called from ``lowering/tile/010_recognize``.
# --------------------------------------------------------------------------- #


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


def try_flash(graph: Graph, root: Node) -> Graph | None:
    """Recognize SDPA on ``root`` and return the fused flash ``Graph`` fragment (a
    ``TileOp`` holding the flash ``Map`` (a ``TWISTED`` kv loop) + its scale / -inf constants), or ``None``
    if ``root`` is not a recognizable / eligible attention kernel."""
    found = _recognize(graph, root)
    if found is None:
        return None
    x_buf, v_id, mask_kind, mask_buf = found
    operands = (x_buf, v_id, *((mask_buf,) if mask_buf is not None else ()))
    if any(nid not in graph.nodes for nid in operands):
        return None

    # A clean scaled-QK producer (Q/K recoverable as plain Loads). A fused score
    # producer (RoPE / GQA index inline) whose Q/K aren't plain loads is not handled —
    # flash isn't recognized, and the softmax-then-P@V falls back to its un-fused tiers.
    qk = _extract_qk(graph.nodes[x_buf])
    if qk is None:
        return None
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


def is_flash_score_producer(graph: Graph, root: Node) -> bool:
    """True iff ``root`` is the **score producer** (the scaled-QK matmul) of a flash kernel
    that :func:`try_flash` would fuse — i.e. some consumer of ``root`` is an eligible
    softmax-then-P@V kernel whose fusion **consumes** ``root``. The general lift in
    ``010_recognize`` defers such a node (leaving it a ``LoopOp``) so the consumer's fusion
    can still read its Q/K as plain loads. The consumed score buffer is the one node absent
    from the fused fragment — Q/K/V inputs survive as the fragment's operands."""
    for consumer in graph.nodes.values():
        if root.id not in consumer.inputs:
            continue
        frag = try_flash(graph, consumer)
        if frag is not None and root.id not in frag.nodes:
            return True
    return False
