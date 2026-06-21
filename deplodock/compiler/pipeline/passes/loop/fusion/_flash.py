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
work. Read from the ``DEPLODOCK_FLASH=1`` env pin; the two-level ``OptionFork``
offer + ``AnalyticPrior`` cold-start are a follow-up.
"""

from __future__ import annotations

import math

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Init, Load, Loop, Monoid, Select, SelectBranch, Write
from deplodock.compiler.pipeline.knob import Knob, KnobType


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


# Structural fork: deploy the fused streaming nest, or fall through to the
# score-materializing 010_sdpa path. Declared like 005_split_demoted's SPLIT_CONE;
# auto-registered by knob.registry()'s module walk.
FLASH = Knob("FLASH", KnobType.BOOL, hints=(True, False), help="Fuse SDPA into the streaming online-softmax flash nest")


def flash_enabled() -> bool:
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
