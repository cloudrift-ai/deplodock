"""The tensor-core flash (warp-chain) emitter — the fragment-resident FA-2 materializer.

``factorize`` (``_factor``) dispatches here when ``tile.op`` is the **mma-flash tree**
``Map(source=Reduction(role=TWISTED, source=Contraction(QK, mma), partial=[…, Contraction(PV, mma)]))``
— the two-``Contraction`` flash tree whose contractions carry a tensor-core :class:`TilePlan`
(the ``DEPLODOCK_CHAIN`` build in ``lowering/tile/_flash.py``). One warp owns 16 query rows and the
full head; it streams the KV in 16-key blocks, keeping the running softmax stats + the ``O``
accumulator **in registers** (never materializing the score matrix):

    QK: S = Q·Kᵀ  → score C-fragments (gmem-direct mma, transposed-B)
    softmax twist over the score fragments (scale · [causal/boundary mask] · rowmax · exp · rowsum),
        the running (m, l) stats + the O·α rescale distributed 2-rows/lane
    C→A handoff: the probability C-fragments → ``flash_pv_smem`` → ``ldmatrix`` → the PV A operand
    PV: O += P·V  → the output accumulator C-fragments (gmem-direct mma, canonical-B)
    epilogue: O / l → RegStore

This reuses the **shared** kernel-IR fragment / mma / store nodes (``LdmatrixLoad`` / ``MmaSyncPtx``
/ ``RegFragment`` / ``FragmentApply`` / ``FragmentRowReduce`` / ``FragmentMask`` / ``RegStore``) — the
same primitives the matmul warp tier renders — so there is no divergent tensor-core codegen; only the
fixed single-warp *orchestration* (the streaming softmax + the register-resident handoff) lives here,
the one genuinely-new primitive the flat matmul tier has no place for.

The softmax algebra realized is the log-sum-exp (flash) twist the ``Reduction``'s ``Carrier`` carries;
the ``scale`` / causal-``Select`` / additive-mask facts are read structurally off the reduce
``partial`` (the same op-tree the scalar flash lowers), never a flag. Geometry (head dim, value dim,
GQA index, symbolic seq) falls out of the operand ``Load`` indices + axes on the two contractions.

The atom is m16n8k16 (f16 / bf16 — the operand dtype only swaps the ``mma.sync`` PTX field); the
fragment C-layout is :data:`~deplodock.compiler.ir.kernel.ir.M16N8` (2 rows / lane, the ``0`` / ``1``
component suffixes). Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

import math

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import (
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
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Assign, Body, Init, Load, Select, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Contraction, Map, Reduction

_MAX = ElementwiseImpl("maximum")
_ADD = ElementwiseImpl("add")
_SUB = ElementwiseImpl("subtract")
_MUL = ElementwiseImpl("multiply")
_DIV = ElementwiseImpl("divide")
_EXP = ElementwiseImpl("exp")

_BN = 16  # the KV block width — one score n-atom pair (2·atom_n) per streaming step

#: The 2 C-fragment rows per lane the m16n8 softmax stats distribute over (``m_i0`` / ``m_i1``).
_COMPS = ("0", "1")


def is_mma_flash(op) -> bool:
    """True iff ``op`` is the mma-flash tree — a ``TWISTED`` :class:`Reduction` (bare or under a
    projecting :class:`Map`) whose ``source`` :class:`Contraction` carries a tensor-core
    :class:`TilePlan`. The ``DEPLODOCK_CHAIN`` build stamps the warp atom; the scalar flash keeps the
    per-cell ``TilePlan()`` and is **not** matched here (it lowers through the cooperative / scalar tier)."""
    red = op.source if isinstance(op, Map) else op
    if not isinstance(red, Reduction) or not isinstance(red.source, Contraction):
        return False
    return red.source.tile.is_warp


def _ext(axis: Axis) -> Expr:
    """The axis extent as an ``Expr`` — a static ``Literal`` or the symbolic ``Dim`` expr."""
    e = axis.extent
    return Literal(e.as_static(), "int") if e.is_static else e.expr


def _pv_contraction(red: Reduction) -> Contraction:
    """The P@V :class:`Contraction` spliced into the reduce ``partial`` (the second contraction of
    the flash tree — its B operand is the value ``Load``, its A the register-resident probability)."""
    return next(s for s in red.partial if isinstance(s, Contraction))


def _is_causal(red: Reduction) -> bool:
    """Causal masking is the score ``Select`` (``kv ≤ m`` else −inf) in the reduce ``partial`` —
    read structurally, never a flag."""
    return any(isinstance(s, Select) for s in red.partial)


def _scalar_chain(name: str, op: ElementwiseImpl, args: tuple[str, ...], *, reassign: bool) -> list[Stmt]:
    """Emit one per-row (``0`` / ``1``) scalar update. ``reassign`` rebinds an already-declared
    carried stat (``m`` / ``l`` — via a fresh temp then :class:`Reassign`, so the enclosing ``Init``
    is not shadowed); else a fresh :class:`Assign` per row component."""
    out: list[Stmt] = []
    for c in _COMPS:
        rc = tuple(f"{a}{c}" for a in args)
        if reassign:
            out.append(Assign(name=f"{name}{c}__t", op=op, args=rc))
            out.append(Reassign(name=f"{name}{c}", value=f"{name}{c}__t"))
        else:
            out.append(Assign(name=f"{name}{c}", op=op, args=rc))
    return out


def factorize_flash(tile, root) -> Tile:
    """Materialize the mma-flash tree into its single-warp streaming ``Tile`` (see the module
    docstring). ``tile.place.grid`` is ``(batch…, qb)`` (one warp per 16-query block); this appends
    the 32-lane axis + ``block_threads = 32``."""
    op: Map = tile.op
    red: Reduction = op.source
    qk: Contraction = red.source
    pv: Contraction = _pv_contraction(red)
    atom = qk.tile.atom
    ab = atom.ab_dtype  # "f16" / "bf16"
    shape = atom.shape  # (16, 8, 16)
    atom_n = atom.atom_n

    q_load, k_load, v_load = qk.a_operand, qk.b_load, pv.b_load
    m_name, kv_name, dd_name = qk.m_axis.name, qk.n_axis.name, qk.k_axis.name
    d_name = pv.n_axis.name
    head_dim = qk.k_axis.extent.as_static()  # QK reduce (static per flash_shape_eligible)
    d_v = pv.n_axis.extent.as_static()  # value dim
    s_k = qk.n_axis  # KV (may be symbolic)
    s_q = qk.m_axis  # query seq (may be symbolic)

    nt = _BN // atom_n  # score n-atoms per block (2)
    nd = d_v // atom_n  # PV output n-atoms / O accumulators
    nk = head_dim // shape[2]  # QK reduce steps (D_head / atom_k)
    scale = 1.0 / math.sqrt(head_dim)
    causal = _is_causal(red)

    grid = tuple(tile.place.grid)  # (batch…, qb)
    qb = grid[-1]
    qb_var = Var(qb.name)
    row_base = BinaryExpr("*", qb_var, Literal(16, "int"))  # this warp's query-row origin
    symbolic = not s_k.extent.is_static
    seq = _ext(s_k)
    q_symbolic = not s_q.extent.is_static

    def _idx(load: Load, sub: dict[str, Expr]) -> tuple:
        return tuple(Sigma(sub).apply(e) for e in load.index)

    scale_lit = f"{scale!r}f"  # 1/√d as a compile-time f32 literal (the head dim is a shape fact)
    body: list[Stmt] = []
    # Running softmax stats (2 rows / lane) + the O accumulators, seeded above the KV stream.
    for c in _COMPS:
        body.append(Init(name=f"m_i{c}", identity=-1e30, dtype=F32))
        body.append(Init(name=f"l_i{c}", identity=0.0, dtype=F32))
    for j in range(nd):
        body.append(RegFragment(name=f"Of{j}", role="c", shape=shape, dtype=F32))

    kv0 = Axis(name=f"{kv_name}0", extent=s_k.extent)
    kv0_var = Var(kv0.name)
    stream: list[Stmt] = []

    # -- QK: score C-fragments (gmem-direct mma, transposed-B) --------------------------------- #
    a_dt, b_dt = atom.operand_dtype("a"), atom.operand_dtype("b")
    sfrags = tuple(f"Sf{t}" for t in range(nt))
    for t in range(nt):
        stream.append(RegFragment(name=sfrags[t], role="c", shape=shape, dtype=F32))
    stream.append(RegFragment(name="_qa", role="a", shape=shape, dtype=a_dt))
    for t in range(nt):
        stream.append(RegFragment(name=f"_kb{t}", role="b", shape=shape, dtype=b_dt))
    for step in range(nk):
        dd0 = Literal(step * shape[2], "int")
        qguard = (row_base, _ext(s_q)) if q_symbolic else None
        stream.append(
            LdmatrixLoad(
                frag="_qa",
                src_buffer=q_load.input,
                src_index=_idx(q_load, {m_name: row_base, dd_name: dd0}),
                role="a",
                ldm=head_dim,
                staged=False,
                gmem_guard=qguard,
            )
        )
        for t in range(nt):
            col_base = BinaryExpr("+", kv0_var, Literal(t * atom_n, "int"))
            # The kv keys are the score's N axis (the contraction is dd) — an out-of-range key past a
            # symbolic ``seq`` is an N-clamp (avoids the OOB gmem read); its score cell is then set to
            # −inf by the boundary ``FragmentMask`` below, so the clamped (duplicate) value is discarded.
            kguard = (col_base, seq) if symbolic else None
            stream.append(
                LdmatrixLoad(
                    frag=f"_kb{t}",
                    src_buffer=k_load.input,
                    src_index=_idx(k_load, {kv_name: col_base, dd_name: dd0}),
                    role="b",
                    ldm=head_dim,
                    staged=False,
                    b_trans=True,
                    gmem_guard=kguard,
                )
            )
            stream.append(MmaSyncPtx(c_frag=sfrags[t], a_frag="_qa", b_frag=f"_kb{t}", shape=shape, ab_dtype=ab))
    # scale, then the coordinate masks (causal upper-triangle, symbolic-seq boundary).
    kv_col_bases = tuple(BinaryExpr("+", kv0_var, Literal(t * atom_n, "int")) for t in range(nt))
    for t in range(nt):
        stream.append(FragmentApply(out=sfrags[t], op=_MUL, args=(sfrags[t], scale_lit), kinds=(FRAG, UNIFORM), in_place=True))
    if causal:
        for t in range(nt):
            stream.append(
                FragmentMask(
                    frag=sfrags[t], mask_when=BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW)), col_base=kv_col_bases[t], row_base=row_base
                )
            )
    if symbolic:
        for t in range(nt):
            stream.append(FragmentMask(frag=sfrags[t], mask_when=BinaryExpr(">=", Var(FRAG_COL), seq), col_base=kv_col_bases[t]))

    # -- softmax twist over the fragments (rowmax · α · exp · rowsum · l · O·α) ------------------ #
    stream.append(FragmentRowReduce(top="rmx0", bot="rmx1", frags=sfrags, op=_MAX, group=4))
    stream += _scalar_chain("mn", _MAX, ("m_i", "rmx"), reassign=False)  # mn = max(m_i, rowmax)
    stream += _scalar_chain("al__d", _SUB, ("m_i", "mn"), reassign=False)  # α = exp(m_i − mn)
    stream += _scalar_chain("al", _EXP, ("al__d",), reassign=False)
    pfrags = tuple(f"Pf{t}" for t in range(nt))
    for t in range(nt):
        stream.append(FragmentApply(out=pfrags[t], op=_SUB, args=(sfrags[t], ("mn0", "mn1")), kinds=(FRAG, ROW)))
        stream.append(FragmentApply(out=pfrags[t], op=_EXP, args=(pfrags[t],), kinds=(FRAG,), in_place=True))
    stream.append(FragmentRowReduce(top="rsm0", bot="rsm1", frags=pfrags, op=_ADD, group=4))
    stream += _scalar_chain("l_i__s", _MUL, ("l_i", "al"), reassign=False)  # l = l·α + rowsum
    stream += _scalar_chain("l_i__n", _ADD, ("l_i__s", "rsm"), reassign=False)
    for c in _COMPS:
        stream.append(Reassign(name=f"l_i{c}", value=f"l_i__n{c}"))
    for j in range(nd):
        stream.append(FragmentApply(out=f"Of{j}", op=_MUL, args=(f"Of{j}", ("al0", "al1")), kinds=(FRAG, ROW), in_place=True))

    # -- C→A handoff: P C-fragments → smem → ldmatrix A → PV mma -------------------------------- #
    for t in range(nt):
        stream.append(
            RegStore(
                dst_buffer="flash_pv_smem", dst_index=(Literal(0, "int"), Literal(t * atom_n, "int")), frag=pfrags[t], shape=shape, ldm=_BN
            )
        )
    stream.append(Sync())
    stream.append(RegFragment(name="_pa", role="a", shape=shape, dtype=atom.operand_dtype("a")))
    stream.append(LdmatrixLoad(frag="_pa", src_buffer="flash_pv_smem", src_index=(Literal(0, "int"),), role="a", ldm=_BN, staged=True))
    kz_pv = (kv0_var, seq) if symbolic else None
    for j in range(nd):
        stream.append(RegFragment(name=f"_vb{j}", role="b", shape=shape, dtype=b_dt))
        stream.append(
            LdmatrixLoad(
                frag=f"_vb{j}",
                src_buffer=v_load.input,
                src_index=_idx(v_load, {kv_name: kv0_var, d_name: Literal(j * atom_n, "int")}),
                role="b",
                ldm=d_v,
                staged=False,
                k_zero=kz_pv,
            )
        )
        stream.append(MmaSyncPtx(c_frag=f"Of{j}", a_frag="_pa", b_frag=f"_vb{j}", shape=shape, ab_dtype=ab))
    for c in _COMPS:
        stream.append(Reassign(name=f"m_i{c}", value=f"mn{c}"))  # advance the running max

    static_small = s_k.extent.is_static and s_k.extent.as_static() <= 128
    body.append(StridedLoop(axis=kv0, start=Literal(0, "int"), step=Literal(_BN, "int"), body=Body(tuple(stream)), unroll=static_small))

    # -- epilogue: O / l → store ---------------------------------------------------------------- #
    for j in range(nd):
        body.append(FragmentApply(out=f"Of{j}", op=_DIV, args=(f"Of{j}", ("l_i0", "l_i1")), kinds=(FRAG, ROW), in_place=True))
    batch_idx = tuple(q_load.index[:-2])  # (batch…, head) — passthrough grid vars
    store_mguard = (row_base, _ext(s_q)) if q_symbolic else None
    for j in range(nd):
        body.append(
            RegStore(
                dst_buffer=root.output.name,
                dst_index=(*batch_idx, row_base, Literal(j * atom_n, "int")),
                frag=f"Of{j}",
                shape=shape,
                ldm=d_v,
                m_guard=store_mguard,
            )
        )

    smem = Smem(name="flash_pv_smem", extents=(16, _BN), dtype=_cuda(atom.operand_dtype("a")))
    lane = Axis(name="_lane", extent=32)
    return Tile(axes=(*grid, lane), body=Body((smem, *body)), block_threads=32)


def _cuda(dtype) -> str:
    from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

    return cuda_name(dtype)


__all__ = ["factorize_flash", "is_mma_flash"]
