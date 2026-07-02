"""The fragment realizer — a TWISTED carrier realized at WARP-FRAGMENT residence.

The one ``_bind`` pipeline calls :func:`realize_warp_twist` when the reduce arm's tree carries a
warp tile (:func:`warp_source` — the stream's head :class:`Contraction` stamped with an mma
:class:`TilePlan` by ``_schedule`` (inside ``010_recognize``)): the streaming reduce keeps its per-step values in mma
C-fragments instead of scalars, so every piece of the scalar lowering is *realized* at the new
residence — the fragment row of the placement-keyed fold (a within-warp ``FragmentRowReduce``
``__shfl`` move where the scalar tier folds in-thread):

- the head contraction (Q@K) emits ``ldmatrix`` + ``mma.sync`` into score C-fragments off its node
  geometry (:func:`_frag_contraction` — operands / ``b_trans`` / guards from the node, the atom
  counts from its stamped ``TilePlan``);
- the score prologue (the scale ``Assign``, the causal ``Select``) is realized stmt-by-stmt
  (:func:`_realize_prologue`): a pointwise ``Assign`` → :class:`FragmentApply`, a
  coordinate-predicated ``Select`` → :class:`FragmentMask` (the keep-predicate negated), a
  loop-invariant scalar ``Load`` hoisted above the stream;
- the twisted carrier's streaming merge is regenerated FROM ITS CHANNEL SPEC (``twist.family ==
  "exp"``, ``channels = (pivot, …)``): the pivot's per-block fold is a ``FragmentRowReduce`` of its
  ``fold`` op (rowmax) + the running-stat update / rescale; a ``denom`` channel (no lift)
  row-reduces the exp-weight fragments (rowsum) into ``l = l·α + Σp``; an ``expect`` channel
  (``lift = multiply``) IS the P@V :class:`Contraction` node — its ⊗ lowers to ``mma.sync`` with
  the register-resident probability fed through the ``flash_pv_smem`` C→A handoff (``RegStore`` →
  ``ldmatrix``), the one genuinely new data-move of this tier;
- the projection tail (``O / l``) realizes as an in-place ``FragmentApply`` + the ``RegStore``
  output close.

Nothing here keys on a kernel *identity* — the walk reads node structure, channel algebra, and the
stamped schedule; an unrealizable tree is rejected at schedule time (``_schedule._twisted_warp_
option`` never stamps the warp tile), so a raise here is an invariant breach, not a fallback.
Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel.ir import (
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    ROW,
    UNIFORM,
    FragmentApply,
    FragmentMask,
    FragmentRowReduce,
    LdmatrixLoad,
    MmaSyncPtx,
    Reassign,
    RegFragment,
    RegStore,
    Smem,
    Sync,
)
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, Body, Init, Load, Select, Stmt, StridedLoop
from emmy.compiler.ir.tile.ir import Contraction, Map, Reduction

_ADD = ElementwiseImpl("add")
_SUB = ElementwiseImpl("subtract")
_MUL = ElementwiseImpl("multiply")
_EXP = ElementwiseImpl("exp")

#: The 2 C-fragment rows per lane the m16n8 stats distribute over (the ``0`` / ``1`` suffixes).
_COMPS = ("0", "1")

#: The C→A handoff slab — the probability C-fragments staged to smem and ``ldmatrix``-read back as
#: the expect contraction's A operand (pinned by the e2e recovery contract).
_PV_SMEM = "flash_pv_smem"

#: Negation of a comparison — a ``Select`` KEEPS where its predicate holds; a ``FragmentMask``
#: FILLS where its predicate holds, so the keep-predicate flips.
_NEGATE = {"<=": ">", "<": ">=", ">=": "<", ">": "<=", "==": "!="}


def warp_source(op) -> Contraction | None:
    """The warp-tiled stream-head :class:`Contraction` of a ``TWISTED`` :class:`Reduction` tree
    (``op`` bare or under a projecting :class:`Map`), or ``None`` — the structural schedule read
    the one binder keys the fragment realization on (like ``plan.coop`` keys the lane tiling)."""
    red = op.source if isinstance(op, Map) else op
    if not isinstance(red, Reduction) or len(red.partial) == 0:
        return None
    head = red.partial[0]
    if isinstance(head, Contraction) and head.tile.is_warp:
        return head
    return None


def _ext(axis: Axis) -> Expr:
    e = axis.extent
    return Literal(e.as_static(), "int") if e.is_static else e.expr


def _idx(load: Load, sub: dict[str, Expr]) -> tuple:
    """The operand gmem index at the fragment tile origin — σ on the load's own index exprs, so
    batch / GQA (``head // group``) terms ride through untouched."""
    return tuple(Sigma(sub).apply(e) for e in load.index)


def _stats(name: str, op: ElementwiseImpl, args: tuple[str, ...]) -> list[Stmt]:
    """One per-row (``0`` / ``1``) scalar-stat update — a fresh ``Assign`` per fragment row."""
    return [Assign(name=f"{name}{c}", op=op, args=tuple(f"{a}{c}" for a in args)) for c in _COMPS]


def _rebind(name: str, value: str) -> list[Stmt]:
    """Advance a running per-row stat in place (the enclosing ``Init`` is not shadowed)."""
    return [Reassign(name=f"{name}{c}", value=f"{value}{c}") for c in _COMPS]


def _row_pair(name: str) -> tuple[str, str]:
    return (f"{name}0", f"{name}1")


def _reads(s: Stmt) -> set[str]:
    """The SSA names a plain score-prologue / merge stmt reads (explicit per kind — the stop
    condition of the prologue walk)."""
    if isinstance(s, Assign):
        return set(s.args)
    if isinstance(s, Select):
        return {br.value for br in s.branches}
    return set()


def _frag_contraction(
    c: Contraction, acc_frags: tuple[str, ...], a_frag: str, *, m_sub: Expr, n_sub, k_sub: Expr, mask: dict
) -> list[Stmt]:
    """One warp-tiled :class:`Contraction` step as fragment codegen — the ``read → ⊗ → fold`` spine
    at fragment residence, geometry off the node (``b_trans`` / operand indices / ``ldm``) and its
    stamped :class:`TilePlan` (``regs`` / ``bk``). ``a_frag`` is the A-operand fragment: filled per
    K-step from the node's ``a_operand`` gmem ``Load`` when it is one, else already resident (the
    caller ran the C→A handoff). ``m_sub`` / ``n_sub(t)`` / ``k_sub`` give each axis's tile-origin
    expr; ``mask`` maps an axis name to its ``(coord, extent)`` overhang guard — a masked A row
    clamp-reads (``gmem_guard``), a masked B column clamp-reads (transposed-B ``gmem_guard``) or
    zero-fills its overhanging K rows (canonical-B ``k_zero``)."""
    atom = c.tile.atom
    shape = atom.shape
    nt = c.tile.regs[1]  # output n-atoms per step (warp order (FM, FN))
    m_name, n_name, k_name = c.m_axis.name, c.n_axis.name, c.k_axis.name
    b_trans = c.b_trans
    a_load = c.a_operand if isinstance(c.a_operand, Load) else None
    ldm_b = c.k_axis.extent.as_static() if b_trans else c.n_axis.extent.as_static()

    out: list[Stmt] = []
    for t in range(nt):
        out.append(RegFragment(name=f"_{c.acc}_b{t}", role="b", shape=shape, dtype=atom.operand_dtype("b")))
    for step in range(c.tile.bk):
        off = Literal(step * shape[2], "int")
        k0 = k_sub if step == 0 else (off if isinstance(k_sub, Literal) and k_sub.value == 0 else BinaryExpr("+", k_sub, off))
        if a_load is not None:
            out.append(
                LdmatrixLoad(
                    frag=a_frag,
                    src_buffer=a_load.input,
                    src_index=_idx(a_load, {m_name: m_sub, k_name: k0}),
                    role="a",
                    ldm=c.k_axis.extent.as_static(),
                    staged=False,
                    gmem_guard=mask.get(m_name),
                )
            )
        for t in range(nt):
            col = n_sub(t)
            n_guard = mask.get(n_name)
            out.append(
                LdmatrixLoad(
                    frag=f"_{c.acc}_b{t}",
                    src_buffer=c.b_load.input,
                    src_index=_idx(c.b_load, {n_name: col, k_name: k0}),
                    role="b",
                    ldm=ldm_b,
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=(col, n_guard[1]) if (n_guard is not None and b_trans) else None,
                    k_zero=mask.get(k_name) if not b_trans else None,
                )
            )
            out.append(MmaSyncPtx(c_frag=acc_frags[t], a_frag=a_frag, b_frag=f"_{c.acc}_b{t}", shape=shape, ab_dtype=atom.ab_dtype))
    return out


def _realize_prologue(stmts, qk: Contraction, frags: tuple[str, ...], col_bases, row_base: Expr, state) -> tuple[list, list]:
    """Realize the score prologue (the plain stmts between the head contraction and the carrier's
    streaming merge) at fragment residence, stmt kind by stmt kind. Returns ``(hoisted, stream)`` —
    the loop-invariant scalar ``Load``\\ s (hoisted above the stream when a realized stmt reads
    them) and the in-stream fragment stmts. Stops at the first stmt touching a carrier state name —
    from there the merge is REGENERATED from the channel spec, not walked."""
    hoisted: list[Load] = []
    stream: list[Stmt] = []
    score = qk.acc
    m_name, n_name = qk.m_axis.name, qk.n_axis.name
    for s in stmts:
        if isinstance(s, Contraction):
            continue  # the expect contraction — regenerated from its channel
        if state & (set(s.defines()) | _reads(s)):
            break  # the dissolved merge — regenerated from the twist channels
        if isinstance(s, Load) and len(s.index) == 0:
            hoisted.append(s)  # a scalar constant (the 1/√d scale, the −inf fill) — loop-invariant
            continue
        if isinstance(s, Assign) and score in s.args:
            others = tuple(a for a in s.args if a != score)
            for f in frags:
                stream.append(FragmentApply(out=f, op=s.op, args=(f, *others), kinds=(FRAG, *(UNIFORM,) * len(others)), in_place=True))
            score = s.name
            continue
        if isinstance(s, Select) and s.branches[0].value == score:
            keep = s.branches[0].select  # a coordinate predicate over the (m, kv) axis vars
            sub = {m_name: Var(FRAG_ROW), n_name: Var(FRAG_COL)}
            mask_when = BinaryExpr(_NEGATE[keep.op], keep.left.substitute(sub), keep.right.substitute(sub))
            for t, f in enumerate(frags):
                stream.append(FragmentMask(frag=f, mask_when=mask_when, col_base=col_bases[t], row_base=row_base))
            score = s.name
            continue
        raise NotImplementedError(f"fragment realizer: unrealizable score-prologue stmt {type(s).__name__}")
    used = {a for st in stream for a in (st.deps() if hasattr(st, "deps") else ())}
    return [ld for ld in hoisted if set(ld.names) & used], stream


def _pv_streamed(pv: Contraction, kv_axis: Axis) -> Contraction:
    """The expect contraction with its singleton intra-block axis swapped for the STREAM axis — the
    scalar tree contracts one key per step (``k_axis = pj``, extent 1); the fragment tier contracts
    the whole block, whose keys ride the stream axis in the value ``Load``'s index (``tile.bk = 1``:
    the block is one atom-K step)."""
    return replace(pv, k_axis=kv_axis)


def realize_warp_twist(op, ctx, tail: tuple) -> tuple[list[Stmt], list[Stmt], list[Stmt]]:
    """Realize a warp-tiled ``TWISTED`` reduce tree at fragment residence — the ``(state, fold,
    close)`` triple the one ``_bind`` pipeline seals (state = the handoff slab + running stats +
    output fragments + hoisted scalars; fold = the streaming :class:`StridedLoop`; close = the
    realized projection + the fragment output store). See the module docstring for the walk."""
    red: Reduction = op.source if isinstance(op, Map) else op
    partial = list(red.partial)
    qk: Contraction = partial[0]
    pv: Contraction = next(s for s in partial[1:] if isinstance(s, Contraction))
    atom = qk.tile.atom
    shape = atom.shape
    atom_n = shape[1]
    nt = qk.tile.regs[1]  # score n-atoms per streaming block
    nd = pv.tile.regs[1]  # output n-atoms (d_v / atom_n)
    bn = nt * atom_n  # the streaming KV block width
    d_v = pv.n_axis.extent.as_static()

    # The ambient warp cell: the last grid axis is the shrunk query-block axis (one warp per
    # ``atom_m`` query rows); the stream rides the reduce axis in ``bn``-key blocks.
    grid = tuple(ctx.grid)
    row_base = BinaryExpr("*", Var(grid[-1].name), Literal(shape[0], "int"))
    kv_axis = red.axis
    kv0 = Axis(name=f"{kv_axis.name}0", extent=kv_axis.extent)
    kv0_var = Var(kv0.name)
    symbolic_k = not kv_axis.extent.is_static
    symbolic_q = not qk.m_axis.extent.is_static
    seq = _ext(kv_axis)
    col_bases = tuple(BinaryExpr("+", kv0_var, Literal(t * atom_n, "int")) for t in range(nt))

    # The channel spec (pivot first, one carried name per channel) — the merge is REGENERATED from
    # it at fragment residence. The expect channel's ⊗ is the pv node; the denom folds the weights.
    channels = red.carrier.twist.channels
    names = red.carrier.state.names
    pivot_name = names[0]
    expect_name = next(n for n, ch in zip(names, channels, strict=True) if ch.lift is not None)
    denom_name = next(n for n, ch in zip(names[1:], channels[1:], strict=True) if ch.lift is None)

    # ---- state: the C→A slab + running stats + output accumulators ---------------------------- #
    state: list[Stmt] = [Smem(name=_PV_SMEM, extents=(shape[0], bn), dtype=_cuda(atom.operand_dtype("a")))]
    for c in _COMPS:
        state.append(Init(name=f"{pivot_name}{c}", identity=-1e30, dtype=F32))
        state.append(Init(name=f"{denom_name}{c}", identity=0.0, dtype=F32))
    ofrags = tuple(f"{expect_name}_f{j}" for j in range(nd))
    for f in ofrags:
        state.append(RegFragment(name=f, role="c", shape=shape, dtype=F32))

    # ---- the streaming step -------------------------------------------------------------------- #
    stream: list[Stmt] = []
    sfrags = tuple(f"{qk.acc}_f{t}" for t in range(nt))
    for f in sfrags:
        stream.append(RegFragment(name=f, role="c", shape=shape, dtype=F32))
    stream.append(RegFragment(name=f"_{qk.acc}_a", role="a", shape=shape, dtype=atom.operand_dtype("a")))
    qk_mask: dict[str, tuple] = {}
    if symbolic_q:
        qk_mask[qk.m_axis.name] = (row_base, _ext(qk.m_axis))
    if symbolic_k:
        qk_mask[qk.n_axis.name] = (kv0_var, seq)
    stream += _frag_contraction(
        qk, sfrags, f"_{qk.acc}_a", m_sub=row_base, n_sub=lambda t: col_bases[t], k_sub=Literal(0, "int"), mask=qk_mask
    )

    hoisted, pro = _realize_prologue(partial[1:], qk, sfrags, col_bases, row_base, set(names))
    state += hoisted
    stream += pro
    if symbolic_k:
        # The blocked stream may overrun a symbolic extent — clamp the overhanging keys to the
        # pivot's fold identity so they contribute nothing (the gmem reads were already clamped).
        for t, f in enumerate(sfrags):
            stream.append(FragmentMask(frag=f, mask_when=BinaryExpr(">=", Var(FRAG_COL), seq), col_base=col_bases[t]))

    # ---- the merge, regenerated from the channel spec at fragment residence ------------------- #
    # pivot: the per-block fold (rowmax) then the running update mn = fold(m, rowmax(S)) + rescale α.
    stream.append(FragmentRowReduce(top="_rmx0", bot="_rmx1", frags=sfrags, op=channels[0].fold, group=4))
    stream += _stats("_mn", channels[0].fold, (pivot_name, "_rmx"))
    stream += _stats("_al__d", _SUB, (pivot_name, "_mn"))  # α = exp(m − mn)
    stream += _stats("_al", _EXP, ("_al__d",))
    pfrags = tuple(f"_p_f{t}" for t in range(nt))  # the softmax weights P = exp(S − mn)
    for sf, pf in zip(sfrags, pfrags, strict=True):
        stream.append(FragmentApply(out=pf, op=_SUB, args=(sf, _row_pair("_mn")), kinds=(FRAG, ROW)))
        stream.append(FragmentApply(out=pf, op=_EXP, args=(pf,), kinds=(FRAG,), in_place=True))
    # denom (no lift): the per-block fold is the exp-weight rowsum; l = l·α + Σp.
    stream.append(FragmentRowReduce(top="_rsm0", bot="_rsm1", frags=pfrags, op=channels[1].fold, group=4))
    stream += _stats(f"{denom_name}__s", _MUL, (denom_name, "_al"))
    stream += _stats(f"{denom_name}__n", _ADD, (f"{denom_name}__s", "_rsm"))
    stream += _rebind(denom_name, f"{denom_name}__n")
    # expect (lift = ⊗): rescale the accumulator, then the lift IS the P@V contraction — the
    # register-resident P fed through the flash_pv_smem C→A handoff as its A operand.
    for f in ofrags:
        stream.append(FragmentApply(out=f, op=_MUL, args=(f, _row_pair("_al")), kinds=(FRAG, ROW), in_place=True))
    for t, pf in enumerate(pfrags):
        stream.append(
            RegStore(dst_buffer=_PV_SMEM, dst_index=(Literal(0, "int"), Literal(t * atom_n, "int")), frag=pf, shape=shape, ldm=bn)
        )
    stream.append(Sync())
    stream.append(RegFragment(name="_pa", role="a", shape=shape, dtype=atom.operand_dtype("a")))
    stream.append(LdmatrixLoad(frag="_pa", src_buffer=_PV_SMEM, src_index=(Literal(0, "int"),), role="a", ldm=bn, staged=True))
    pv_mask = {kv_axis.name: (kv0_var, seq)} if symbolic_k else {}
    stream += _frag_contraction(
        _pv_streamed(pv, kv_axis), ofrags, "_pa", m_sub=row_base, n_sub=lambda j: Literal(j * atom_n, "int"), k_sub=kv0_var, mask=pv_mask
    )
    stream += _rebind(pivot_name, "_mn")  # advance the running pivot: m = mn

    static_small = kv_axis.extent.is_static and kv_axis.extent.as_static() <= 128
    fold = [StridedLoop(axis=kv0, start=Literal(0, "int"), step=Literal(bn, "int"), body=Body(tuple(stream)), unroll=static_small)]

    # ---- close: the projection tail realized on the output fragments + the store -------------- #
    close: list[Stmt] = []
    for s in tail:
        if isinstance(s, Assign) and expect_name in s.args:
            others = tuple(_row_pair(a) if a in names else a for a in s.args if a != expect_name)
            kinds = tuple(ROW if a in names else UNIFORM for a in s.args if a != expect_name)
            for f in ofrags:
                close.append(FragmentApply(out=f, op=s.op, args=(f, *others), kinds=(FRAG, *kinds), in_place=True))
            continue
        raise NotImplementedError(f"fragment realizer: unrealizable projection stmt {type(s).__name__}")
    batch_idx = tuple(qk.a_operand.index[:-2])  # (batch…, head) — passthrough grid vars
    m_guard = (row_base, _ext(qk.m_axis)) if symbolic_q else None
    for j, f in enumerate(ofrags):
        close.append(
            RegStore(
                dst_buffer=ctx.output,
                dst_index=(*batch_idx, row_base, Literal(j * atom_n, "int")),
                frag=f,
                shape=shape,
                ldm=d_v,
                m_guard=m_guard,
            )
        )
    return state, fold, close


def _cuda(dtype) -> str:
    from emmy.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

    return cuda_name(dtype)


__all__ = ["realize_warp_twist", "warp_source"]
