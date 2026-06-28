"""Flash-attention shared helper — the carrier builders, eligibility predicate,
and the graph-fragment builder.

Recognition itself lives in ``lowering/tile/010_recognize`` (which calls these
builders); this module owns the *construction* side: the ``flash_combine`` /
``online_softmax_combine`` carriers, the ``flash_shape_eligible`` / ``gqa_group``
predicates, and the fragment builder ``build_flash_frag``. It doesn't hand-assemble a
kernel body — it builds the high-level op tree (``ir/tile/ops``): flash is the
``(m,l,O)`` LSE ``Reduce`` over kv whose score partial ``Map`` holds the ``Σ Q·K``
contraction, the carrier ``finalize``\\ ing ``O/l``. ``build_flash_frag`` returns that
``Reduce`` UNLOWERED, on a ``TileOp`` — the free ``(batch…, m, d)`` axes are its
``grid_axes``; ``020_schedule`` passes the ``TileOp`` through, and ``materialize`` lowers
it + generates the output-store glue (the ``Write`` at the grid cell — not stored).

Scope is the **clean** scaled-QK producer (Q/K recoverable as plain ``Load``\\ s). A
fused score producer whose Q/K are computed SSA — RoPE'd QK — is NOT recognized as
flash (it falls back to its un-fused tiers); a producer-splicing builder for that case
was removed rather than kept half-converted to the op tree.

The fragment fuses scaled-dot-product attention into ONE kernel that tiles the KV
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
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Monoid, Select, SelectBranch, Twist
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.ir.tile.ops import Map, Reduce, TensorRef, lower


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
        twist=Twist(merge=merge, combine_states=combine_states, state_b=state_b),
        identity=identity,
        commutative=True,
        axes=("kv",),
        # φ projection: the streamed output O is unnormalized — divide by the
        # log-sum-exp denominator l once, after the kv loop. ``lower`` emits it post-loop.
        finalize=(Assign(f"{o}__proj", "divide", (o, ll)),),
    )


def online_softmax_combine(m: str, d: str, s: str, *, axis: str = "kv") -> Monoid:
    """The standalone **online-softmax** :class:`Monoid` — flash's softmax-stats half without the P@V
    accumulator. State ``(m, d)`` (running row max / exp-sum denominator) folds this element's score
    partial ``s`` in ONE streaming pass (vs the classic two: a row-max reduce then a
    ``Σ exp(x − max)`` reduce)::

        m_new = max(m, s);   alpha = exp(m − m_new);   p = exp(s − m_new)
        d = d·alpha + p;     m = m_new   (last)

    A non-twisted carrier (no value partial) — ``Monoid.render`` realizes its merge at the scalar
    tier; the downstream normalize pass reads the final ``m`` and ``1/d``. Temps are namespaced
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
        twist=Twist(merge=merge, combine_states=combine_states, state_b=(mb, db)),
        identity=(Literal(-1e30), Literal(0.0)),  # (−inf, 0)
        commutative=True,
        axes=(axis,),
    )


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

    The compute is the op tree itself — a ``Reduce`` (the ``(m,l,O)`` LSE fold whose
    carrier ``finalize``\\ s ``O/l``) carried unlowered on the ``TileOp``; the free
    ``(batch…, m, d)`` axes become its ``grid_axes`` (no free-axis loop nest).
    ``020_schedule`` passes the ``TileOp`` through; ``materialize`` lowers it and
    generates the output-store glue (the ``Write`` at the grid cell) — it isn't stored
    here.

    ``group`` is the GQA head ratio (K/V indexed at ``head // group``); ``mask`` is
    an optional ``(buffer_id, shape)`` additive bias loaded per ``(m, kv)``."""
    batch = [_static(d) for d in q_shape[:-2]]
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    s_q_dim, s_k_dim = q_shape[-2], k_shape[-2]  # Dim instances — static int or symbolic seq_len
    scale = 1.0 / math.sqrt(head_dim)
    mask_buf, mask_shape = mask if mask is not None else (None, None)

    flash_reduce = _flash_op(
        q_id, k_id, v_id, batch, s_k_dim, head_dim, causal=causal, group=group, mask_buf=mask_buf, mask_shape=mask_shape
    )
    grid = (
        *(Axis(name=f"b{i}", extent=Dim(b)) for i, b in enumerate(batch)),
        Axis(name="m", extent=s_q_dim),
        Axis(name="d", extent=Dim(d_v)),
    )
    tile = TileOp(op=flash_reduce, grid_axes=grid)

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
    s_k: Dim,
    head_dim: int,
    *,
    causal: bool = False,
    group: int = 1,
    mask_buf: str | None = None,
    mask_shape: tuple | None = None,
) -> Reduce:
    """The per-output-element ``(…, m, d)`` compute as the op tree itself: flash is the
    ``(m,l,O)`` LSE :class:`Reduce` over ``kv`` whose score partial ``Map`` holds a
    NESTED contraction ``Σ_dd Q·K`` (scaled, optionally masked); the carrier
    ``finalize``\\ s ``O/l``. Returns the unlowered ``Reduce`` (carried on the
    ``TileOp``). The free ``(batch…, m, d)`` axes are the ``TileOp``'s grid, not loops
    here; the output store is glue generated at materialize.

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

    add = ElementwiseImpl("add")
    score_reduce = Reduce(  # s = Σ_dd Q·K — the inner contraction (SEMIRING), reset per kv step
        out="sacc",
        axis=Axis(name="dd", extent=Dim(head_dim)),
        carrier=Accum(name="sacc", value="qk", op=add).as_monoid(),
        partials=(
            Map(
                [
                    Load(name="q_e", input=q_buf, index=q_idx),
                    Load(name="k_e", input=k_buf, index=k_idx),
                    Assign(name="qk", op="multiply", args=("q_e", "k_e")),
                ]
            ),
        ),
        init_ops=(add,),
    )
    # The score partial is one Map: the dd contraction, the scale, and the mask — its
    # last stmt binds ``score_name`` (the carrier's score partial).
    score_stmts = [
        *lower(score_reduce),
        Load(name="scale_c", input="_flash_scale", index=()),
        Assign(name="s", op="multiply", args=("sacc", "scale_c")),
    ]
    if causal:
        # Causal mask: keep the score where key ≤ query (kv ≤ m), else −inf.
        score_stmts += [
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
        score_stmts += [Load(name="mask_e", input=mask_buf, index=mask_idx), Assign(name="s_masked", op="add", args=("s", "mask_e"))]
        score_name = "s_masked"
    else:
        score_name = "s"
    flash_reduce = Reduce(  # the (m,l,O) streaming fold over kv; carrier.finalize emits O_i/l_i post-loop
        out="O_i__proj",
        axis=Axis(name="kv", extent=s_k),
        carrier=flash_combine("m_i", "l_i", "O_i", score_name, "v_e"),
        partials=(Map(score_stmts), TensorRef(v_buf, v_idx)),
        init_ops=(ElementwiseImpl("maximum"), add, add),
    )
    return flash_reduce
