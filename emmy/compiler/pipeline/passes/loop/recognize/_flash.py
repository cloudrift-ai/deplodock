"""Flash-attention shared helper — the ``FLASH`` knob, eligibility predicate, and
the streaming online-softmax ``LoopOp`` nest builder.

Flash recognition is a **Loop-IR** pass (``loop/fusion/025_recognize_flash``) that
runs AFTER the generic fuser: a non-causal SDPA fuses to two ``LoopOp``s — the
scaled scores and the softmax-then-P@V kernel — and the pass pattern-matches that
consolidated softmax-attention kernel and rewrites it into one fused streaming
kernel, with NO modification to the decomposition stage. This module is the sole
owner of the ``FLASH`` knob, the ``flash_shape_eligible`` predicate, and the nest
builder ``build_flash_frag``.

The nest fuses scaled-dot-product attention into ONE kernel that tiles the KV
(reduce) axis and never materializes the ``[S_q, S_k]`` score matrix. It runs one
independent streaming softmax per output element ``(…, m, d)`` — a correct, if
redundant, scalar form; the tensor-core P@V tier is future work
(``plans/online-softmax-flash-attention.md``)::

    for *batch, m (query rows), d (value dim):       # free / grid
      Init (m_i = -inf, l_i = 0, O_i = 0)            # running (max, denom, out)
      for kv in 0..S_k:                              # streaming reduce
        Init sacc = 0
        for dd in 0..head_dim: sacc += Q[…,m,dd]·K[…,kv,dd]   # score reduce
        s = sacc · scale
        Monoid((m_i,l_i,O_i), (s, V[…,kv,d]))       # the LSE rescale (flash_combine)
      out[…,m,d] = O_i / l_i

Scope: static OR dynamic (symbolic ``seq_len`` on Q/K/V dim -2 — one cached kernel
carrying ``int seq_len`` serves every runtime size, the symbol landing on BOTH the
masked-row M and the symbolic reduce), causal or non-causal (causal masks the
score per element, ``kv ≤ m`` else −inf — tile-skip is a tensor-core-tier
follow-up), an optional broadcast additive mask (the HF ``(1,1,S,S)`` float bias),
and GQA (``q_heads == group · kv_heads``; the K/V head axis read at ``head //
group`` directly, no materialized broadcast). The tensor-core P@V tier is future
work. Read from the ``EMMY_FLASH=1`` env pin; the two-level ``OptionFork``
offer + ``AnalyticPrior`` cold-start are a follow-up.
"""

from __future__ import annotations

import math

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.loop.ir import LoopOp
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Init, Load, Loop, Monoid, Select, SelectBranch, Write
from emmy.compiler.ir.stmt.passes import rewrite as rewrite_stmt
from emmy.compiler.pipeline.knob import Knob, KnobType


def flash_combine(m: str, ll: str, o: str, s: str, v: str) -> Monoid:
    """Build the online-softmax (log-sum-exp) :class:`Monoid` for one streaming
    KV step: state ``(m, l, O)`` folds this key's ``(score, value)`` partial.

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        l = l·alpha + p;     O = O·alpha + p·v;         m = m_new   (last)

    The LSE monoid is asymmetric (partial ``(s, v)`` has different arity than
    state ``(m, l, O)``), so the **state-merges-state** form can't be derived from
    ``merge`` — ``combine_states`` is authored too: it merges this carrier's
    ``(m, l, O)`` with a second partition's ``(m_o, l_o, O_o)`` (named by
    ``state_b``), the form the cross-partition combine (cooperative-tree /
    split-KV / split-K cross-CTA reduce) folds::

        m_new = max(m, m_o);  a = exp(m − m_new);  b = exp(m_o − m_new)
        l = l·a + l_o·b;      O = O·a + O_o·b;     m = m_new   (last)

    Temps are namespaced by the state's first name so they stay unique per kernel
    (they're internal to the carrier — invisible to the body's SSA renamer)."""

    def t(suf: str) -> str:
        return f"{m}__{suf}"

    merge = (
        Assign(t("mx"), "maximum", (m, s)),  # m_new = max(m, s)
        Assign(t("dm"), "subtract", (m, t("mx"))),  # m − m_new
        Assign(t("al"), "exp", (t("dm"),)),  # alpha = exp(m − m_new)  (reads OLD m)
        Assign(t("ds"), "subtract", (s, t("mx"))),  # s − m_new
        Assign(t("p"), "exp", (t("ds"),)),  # p = exp(s − m_new)
        Assign(t("lm"), "multiply", (ll, t("al"))),  # l·alpha
        Assign(ll, "add", (t("lm"), t("p"))),  # l = l·alpha + p          [state]
        Assign(t("om"), "multiply", (o, t("al"))),  # O·alpha
        Assign(t("pv"), "multiply", (t("p"), v)),  # p·v
        Assign(o, "add", (t("om"), t("pv"))),  # O = O·alpha + p·v        [state]
        Assign(m, "copy", (t("mx"),)),  # m = m_new                       [state, last]
    )
    identity = (Literal(-1e30), Literal(0.0), Literal(0.0))  # (−inf, 0, 0)
    state_b = (f"{m}__o", f"{ll}__o", f"{o}__o")  # the second partition's (m, l, O)
    mb, lb, ob = state_b
    combine_states = (
        Assign(t("cmx"), "maximum", (m, mb)),  # m_new = max(m, m_o)
        Assign(t("cda"), "subtract", (m, t("cmx"))),  # m − m_new
        Assign(t("ca"), "exp", (t("cda"),)),  # a = exp(m − m_new)   (reads OLD m)
        Assign(t("cdb"), "subtract", (mb, t("cmx"))),  # m_o − m_new
        Assign(t("cb"), "exp", (t("cdb"),)),  # b = exp(m_o − m_new)
        Assign(t("cla"), "multiply", (ll, t("ca"))),  # l·a
        Assign(t("clb"), "multiply", (lb, t("cb"))),  # l_o·b
        Assign(ll, "add", (t("cla"), t("clb"))),  # l = l·a + l_o·b           [state]
        Assign(t("coa"), "multiply", (o, t("ca"))),  # O·a
        Assign(t("cob"), "multiply", (ob, t("cb"))),  # O_o·b
        Assign(o, "add", (t("coa"), t("cob"))),  # O = O·a + O_o·b            [state]
        Assign(m, "copy", (t("cmx"),)),  # m = m_new                          [state, last]
    )
    return Monoid(
        state=(m, ll, o),
        partial=(s, v),
        merge=merge,
        identity=identity,
        commutative=True,
        axes=("kv",),
        state_b=state_b,
        combine_states=combine_states,
    )


def online_softmax_combine(m: str, d: str, s: str, *, axis: str = "kv") -> Monoid:
    """The standalone **online-softmax** :class:`Monoid` — flash's softmax-stats half without the P@V
    accumulator. State ``(m, d)`` (running row max / exp-sum denominator) folds this element's score
    partial ``s`` in ONE streaming pass (vs the classic two: a row-max reduce then a
    ``Σ exp(x − max)`` reduce)::

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        d = d·alpha + p;     m = m_new   (last)

    A non-twisted carrier (no value partial) — :meth:`ScalarCombiner.combine` realizes it at the
    scalar tier; the downstream normalize pass reads the final ``m`` and ``1/d``. Temps are namespaced
    by ``m`` so they stay unique per kernel."""

    def t(suf: str) -> str:
        return f"{m}__{suf}"

    merge = (
        Assign(t("mx"), "maximum", (m, s)),  # m_new = max(m, s)
        Assign(t("dm"), "subtract", (m, t("mx"))),  # m − m_new
        Assign(t("al"), "exp", (t("dm"),)),  # alpha = exp(m − m_new)  (reads OLD m)
        Assign(t("ds"), "subtract", (s, t("mx"))),  # s − m_new
        Assign(t("p"), "exp", (t("ds"),)),  # p = exp(s − m_new)
        Assign(t("dl"), "multiply", (d, t("al"))),  # d·alpha
        Assign(d, "add", (t("dl"), t("p"))),  # d = d·alpha + p          [state]
        Assign(m, "copy", (t("mx"),)),  # m = m_new                       [state, last]
    )
    # The LSE monoid is asymmetric (partial arity ≠ state arity), so the state-merges-state form the
    # cross-partition combine (cooperative-tree / split-KV) needs is authored explicitly (not derivable
    # from ``merge``): merge this partition's (m, d) with a second's (m_o, d_o).
    mb, db = f"{m}__o", f"{d}__o"
    combine_states = (
        Assign(t("cmx"), "maximum", (m, mb)),  # m_new = max(m, m_o)
        Assign(t("cda"), "subtract", (m, t("cmx"))),  # m − m_new
        Assign(t("ca"), "exp", (t("cda"),)),  # a = exp(m − m_new)   (reads OLD m)
        Assign(t("cdb"), "subtract", (mb, t("cmx"))),  # m_o − m_new
        Assign(t("cb"), "exp", (t("cdb"),)),  # b = exp(m_o − m_new)
        Assign(t("cda2"), "multiply", (d, t("ca"))),  # d·a
        Assign(t("cdb2"), "multiply", (db, t("cb"))),  # d_o·b
        Assign(d, "add", (t("cda2"), t("cdb2"))),  # d = d·a + d_o·b           [state]
        Assign(m, "copy", (t("cmx"),)),  # m = m_new                          [state, last]
    )
    return Monoid(
        state=(m, d),
        partial=(s,),
        merge=merge,
        identity=(Literal(-1e30), Literal(0.0)),  # (−inf, 0)
        commutative=True,
        axes=(axis,),
        state_b=(mb, db),
        combine_states=combine_states,
    )


# Structural fork: deploy the fused streaming nest, or fall through to the
# score-materializing 010_sdpa path. A structural-decision BOOL like the cut fork's
# CUT mask; auto-registered by knob.registry()'s module walk.
FLASH = Knob("FLASH", KnobType.BOOL, hints=(True, False), help="Fuse SDPA into the streaming online-softmax flash nest")
# Standalone-softmax online fusion (the row-max + exp-sum two-pass reduce → one streaming
# online-softmax monoid pass). Off by default; auto-registered by knob.registry()'s module walk.
ONLINE_SOFTMAX = Knob("ONLINE_SOFTMAX", KnobType.BOOL, hints=(False, True), help="Fuse a 2-pass softmax into one online-softmax pass")


def flash_enabled() -> bool:
    """The explicit ``FLASH`` knob. The move composer ALSO implies flash, but only
    for symbolic-seq SDPA (static SDPA decomposes — the composer covers the static
    QK^T / softmax / P@V kernels, and the loop reference backend can run those but
    can't render the streaming flash carrier). That conditional implication lives
    at the recognizer (`_composer_wants_flash`), which has the shape."""
    raw = FLASH.raw()
    return raw is not None and FLASH.parse(raw)


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
    """Build the fragment graph holding the fused flash ``LoopOp`` (+ its scale /
    -inf constants). The caller guarantees :func:`flash_shape_eligible`.

    ``group`` is the GQA head ratio (K/V indexed at ``head // group``); ``mask`` is
    an optional ``(buffer_id, shape)`` additive bias loaded per ``(m, kv)``."""
    batch = [_static(d) for d in q_shape[:-2]]
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    s_q_dim, s_k_dim = q_shape[-2], k_shape[-2]  # Dim instances — static int or symbolic seq_len
    scale = 1.0 / math.sqrt(head_dim)
    mask_buf, mask_shape = mask if mask is not None else (None, None)

    body = _flash_loop_body(
        q_id,
        k_id,
        v_id,
        "_flash_scale",
        batch,
        s_q_dim,
        s_k_dim,
        head_dim,
        d_v,
        out.name,
        causal=causal,
        group=group,
        mask_buf=mask_buf,
        mask_shape=mask_shape,
    )
    nest = _wrap_free_axes(body, batch, s_q_dim, d_v)

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
    frag.add_node(op=LoopOp(body=nest), inputs=inputs, output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag


def _batch_vars(n: int) -> tuple[Var, ...]:
    return tuple(Var(f"b{i}") for i in range(n))


def _flash_loop_body(
    q_buf: str,
    k_buf: str,
    v_buf: str,
    scale_buf: str,
    batch: list[int],
    s_q: Dim,
    s_k: Dim,
    head_dim: int,
    d_v: int,
    out_buf: str,
    *,
    causal: bool = False,
    group: int = 1,
    mask_buf: str | None = None,
    mask_shape: tuple | None = None,
) -> tuple:
    """The per-output-element ``(…, m, d)`` body: streaming KV reduce + finalize.

    The score ``s = Σ_dd Q·K`` is an inner reduce nested in the KV streaming
    reduce, so ``Init(sacc)`` is pre-placed at the KV-body scope (reset per step):
    the streaming ``Monoid`` loop is a per-iteration boundary the default
    Init-placement would otherwise cross.

    GQA: the K/V head axis (last batch dim) is read at ``head // group``, the same
    ``//group`` the upstream ``IndexMapOp`` encodes, moved into the load index so
    the kv_heads-many K/V are read without materializing the q_heads expansion.
    An additive ``mask_buf`` (broadcast leading dims) is loaded per ``(m, kv)`` and
    summed into the score before the streaming combine — ``-inf`` entries make
    ``exp(s − m_new) = 0``, so masked keys contribute nothing."""
    bvars = _batch_vars(len(batch))
    head_axis = len(batch) - 1  # last batch dim is the head (when there is one)
    kv_bvars = tuple(BinaryExpr("/", bv, Literal(group, "int")) if (group > 1 and i == head_axis) else bv for i, bv in enumerate(bvars))
    q_idx = (*bvars, Var("m"), Var("dd"))
    k_idx = (*kv_bvars, Var("kv"), Var("dd"))
    v_idx = (*kv_bvars, Var("kv"), Var("d"))
    out_idx = (*bvars, Var("m"), Var("d"))

    score_reduce = Loop(
        axis=Axis(name="dd", extent=Dim(head_dim)),
        body=(
            Load(name="q_e", input=q_buf, index=q_idx),
            Load(name="k_e", input=k_buf, index=k_idx),
            Assign(name="qk", op="multiply", args=("q_e", "k_e")),
            Accum(name="sacc", value="qk", op=ElementwiseImpl("add")),
        ),
    )
    score: tuple = (Assign(name="s", op="multiply", args=("sacc", "scale_c")),)
    score_name = "s"
    prologue: tuple = ()
    if causal:
        # Causal mask: keep the score where key ≤ query (kv ≤ m), else −inf.
        prologue = (Load(name="ninf_c", input="_flash_ninf", index=()),)
        score += (
            Select(
                name="s_masked",
                branches=(
                    SelectBranch(value="s", select=BinaryExpr("<=", Var("kv"), Var("m"))),
                    SelectBranch(value="ninf_c", select=Literal(1, "int")),
                ),
            ),
        )
        score_name = "s_masked"
    elif mask_buf is not None:
        # Additive bias: leading dims broadcast (indexed to 0), trailing two are
        # the query row m and the streaming key kv.
        mask_idx = (*(Literal(0, "int") for _ in mask_shape[:-2]), Var("m"), Var("kv"))
        score += (
            Load(name="mask_e", input=mask_buf, index=mask_idx),
            Assign(name="s_masked", op="add", args=(score_name, "mask_e")),
        )
        score_name = "s_masked"
    kv_body = (
        Init(name="sacc", op=ElementwiseImpl("add"), dtype="f32"),
        score_reduce,
        *score,
        Load(name="v_e", input=v_buf, index=v_idx),
        flash_combine("m_i", "l_i", "O_i", score_name, "v_e"),
    )
    return (
        Load(name="scale_c", input=scale_buf, index=()),
        *prologue,
        Init(name="m_i", op=ElementwiseImpl("maximum"), dtype="f32"),
        Init(name="l_i", op=ElementwiseImpl("add"), dtype="f32"),
        Init(name="O_i", op=ElementwiseImpl("add"), dtype="f32"),
        Loop(axis=Axis(name="kv", extent=s_k), body=kv_body),
        Assign(name="res", op="divide", args=("O_i", "l_i")),
        Write(output=out_buf, index=out_idx, value="res"),
    )


def _wrap_free_axes(body: tuple, batch: list[int], s_q: Dim, d_v: int) -> tuple:
    """Wrap the per-element body in the free grid loops: batch…, query m, value d."""
    nest: tuple = (Loop(axis=Axis(name="m", extent=s_q), body=(Loop(axis=Axis(name="d", extent=Dim(d_v)), body=body),)),)
    for i in reversed(range(len(batch))):
        nest = (Loop(axis=Axis(name=f"b{i}", extent=Dim(batch[i])), body=nest),)
    return nest


# --------------------------------------------------------------------------- #
# Recovered-body flash — for fused score producers the synthetic builder can't
# reconstruct (the QK reduce carries RoPE + GQA index + scale + mask inline, so
# Q/K are computed SSA values, not plain Loads). Instead of rebuilding the score
# from recovered Q/K buffers, inline the producer's score COMPUTATION wholesale
# and recover the V-load / output-write index arithmetic from the consumer — so
# RoPE, the GQA ``head // group`` index, the per-element mask, and the o_proj
# reshape all ride along verbatim. The only thing flash adds is the streaming
# softmax + P@V (the score never round-trips through gmem).
# --------------------------------------------------------------------------- #


def _var_name(e: Expr) -> str | None:
    return e.name if isinstance(e, Var) else None


def _only_write(op: LoopOp) -> Write | None:
    ws = [s for s in op.body.iter() if isinstance(s, Write)]
    return ws[0] if len(ws) == 1 else None


def _loop_containing(stmts: tuple, target: object) -> Loop | None:
    """The innermost ``Loop`` whose immediate body holds ``target`` (by identity)."""
    for s in stmts:
        if isinstance(s, Loop):
            if any(c is target for c in s.body):
                return s
            inner = _loop_containing(s.body, target)
            if inner is not None:
                return inner
    return None


def _collect_ssa(stmts: tuple) -> set[str]:
    """Every SSA name a (possibly nested) stmt sequence defines."""
    names: set[str] = set()
    for s in stmts:
        if isinstance(s, Loop):
            names |= _collect_ssa(s.body)
        else:
            names |= set(s.defines())
    return names


def _all_loads(stmts: tuple) -> list[Load]:
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Loop):
            out += _all_loads(s.body)
        elif isinstance(s, Load):
            out.append(s)
    return out


def _ext(op: LoopOp, name: str):
    for lp in op.body.iter_of_type(Loop):
        if lp.axis.name == name:
            return lp.axis.extent
    return None


def build_flash_recovered(graph: Graph, producer: LoopOp, consumer: LoopOp, x_buf: str, v_id: str, out: Tensor) -> Graph | None:
    """Rewrite a (score producer, softmax-P@V consumer) pair into one streaming
    flash ``LoopOp`` by inlining the producer's recovered score body and recovering
    the consumer's V-load / output-write indices. Returns ``None`` (→ fall through)
    when the pair doesn't match the recoverable shape.

    The producer must write the score per ``(…, head, m, kv)`` with a single inner
    reduce (over ``dd``); the consumer's P@V reduce supplies the V access pattern
    (``head // group`` + any o_proj reshape) and the output index."""
    # --- producer: the per-(head, m, kv) score computation ---
    pw = _only_write(producer)
    if pw is None or pw.output != x_buf or len(pw.index) < 3:
        return None
    p_head, p_m, p_kv = _var_name(pw.index[-3]), _var_name(pw.index[-2]), _var_name(pw.index[-1])
    if None in (p_head, p_m, p_kv):
        return None
    score_loop = _loop_containing(producer.body, pw)
    if score_loop is None or score_loop.axis.name != p_kv:
        return None
    score_stmts = tuple(s for s in score_loop.body if s is not pw)
    reduce_loops = [s for s in score_stmts if isinstance(s, Loop)]
    if len(reduce_loops) != 1:
        return None  # exactly one inner score reduce (the dd dot product)
    p_dd = reduce_loops[0].axis.name
    accums = [s for s in reduce_loops[0].body if isinstance(s, Accum)]
    if not accums:
        return None
    prologue: list = []
    for s in producer.body:
        if isinstance(s, Loop):
            break
        prologue.append(s)

    # --- consumer: V-load index, output index, free extents ---
    cw = _only_write(consumer)
    if cw is None:
        return None
    pv_loop = None
    for lp in consumer.body.iter_of_type(Loop):
        if any(isinstance(s, Accum) and s.name == cw.value for s in lp.body):
            pv_loop = lp
    if pv_loop is None:
        return None
    c_kv = pv_loop.axis.name
    v_loads = [s for s in pv_loop.body if isinstance(s, Load) and s.input == v_id]
    if len(v_loads) != 1:
        return None
    v_index = v_loads[0].index
    d_loop = _loop_containing(consumer.body, cw)
    if d_loop is None:
        return None
    c_d = d_loop.axis.name
    x_loads = [s for s in consumer.body.iter() if isinstance(s, Load) and s.input == x_buf]
    if not x_loads or len(x_loads[0].index) < 3:
        return None
    c_head, c_m = _var_name(x_loads[0].index[-3]), _var_name(x_loads[0].index[-2])
    if c_head is None or c_m is None:
        return None

    head_ext, m_ext = _ext(consumer, c_head), _ext(consumer, c_m)
    d_ext, kv_ext = d_loop.axis.extent, pv_loop.axis.extent  # the dd reduce keeps the producer's own extent
    if any(e is None for e in (head_ext, m_ext)):
        return None

    # --- remap the producer score body into flash's (h, m, kv, dd) namespace ---
    prod_sigma = Sigma({p_head: Var("h"), p_m: Var("m"), p_kv: Var("kv"), p_dd: Var("dd")})

    def prod_axis(ax: Axis) -> Axis:
        return Axis(name="dd", extent=ax.extent) if ax.name == p_dd else ax

    ssa = _collect_ssa(tuple(prologue)) | _collect_ssa(score_stmts)

    def prod_rename(n: str) -> str:
        return f"sc_{n}" if n in ssa else n

    new_prologue = [rewrite_stmt(s, prod_rename, prod_sigma, prod_axis) for s in prologue]
    new_score = [rewrite_stmt(s, prod_rename, prod_sigma, prod_axis) for s in score_stmts]
    score_name = prod_rename(pw.value)
    inits = [Init(name=prod_rename(a.name), op=a.op, dtype="f32") for a in accums]

    # --- recover the consumer's V-load and output indices ---
    v_sigma = Sigma({c_head: Var("h"), c_kv: Var("kv"), c_d: Var("d")})
    out_sigma = Sigma({c_head: Var("h"), c_m: Var("m"), c_d: Var("d")})
    new_v_index = tuple(v_sigma.apply(e) for e in v_index)
    new_out_index = tuple(out_sigma.apply(e) for e in cw.index)

    kv_body = (
        *inits,
        *new_score,
        Load(name="rv_e", input=v_id, index=new_v_index),
        flash_combine("m_i", "l_i", "O_i", score_name, "rv_e"),
    )
    elem = (
        Init(name="m_i", op=ElementwiseImpl("maximum"), dtype="f32"),
        Init(name="l_i", op=ElementwiseImpl("add"), dtype="f32"),
        Init(name="O_i", op=ElementwiseImpl("add"), dtype="f32"),
        Loop(axis=Axis(name="kv", extent=kv_ext), body=kv_body),
        Assign(name="res", op="divide", args=("O_i", "l_i")),
        Write(output=out.name, index=new_out_index, value="res"),
    )
    nest = (
        *new_prologue,
        Loop(
            axis=Axis(name="h", extent=head_ext),
            body=(Loop(axis=Axis(name="m", extent=m_ext), body=(Loop(axis=Axis(name="d", extent=d_ext), body=elem),)),),
        ),
    )

    frag = Graph()
    bufs = {ld.input for ld in (*_all_loads(tuple(new_prologue)), *_all_loads(tuple(new_score)))} | {v_id}
    for bid in sorted(bufs):
        t = graph.nodes[bid].output
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(bid, t.shape, t.dtype), node_id=bid)
    frag.add_node(op=LoopOp(body=nest), inputs=sorted(bufs), output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag
