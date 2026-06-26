"""Warp-tier streaming-flash realizer — `realize_flash` turns the logical flash TileGraph into a
`TileOp` through the **generic carry assembler** (`_assemble.assemble_carry`).

The fused tensor-core flash is the maximal carrier (an MMA contraction AND an online-updating
monoid), so it materializes through the same tower path as matmul / reduce — only the carrier
phases differ. This module owns the flash-specific *authoring* (the produce / merge / handoff /
consume phases + the `(B,H,S,D)` addressing, GQA head-mapping, causal + symbolic-`seq_len`
masks, the C→A smem handoff); the tower nesting, the MMA-cell lowering (`kernel/005`), and the
fragment-softmax realization (`_frag_softmax` over the shared `carrier_algebra` classifier) are
all the generic shared machinery.

`split/005_warp_chain` is the eligibility/offer shim: it detects an in-scope flash `LoopOp`,
reads the carrier + attaches the logical FA-2 TileGraph, and emits a `TileGraphOp(tilegraph=…, flash=carrier)`; the
assembly pass (`assembly/010_assemble`) calls :func:`realize_flash` on it. No separate assembler
entry, no enumeration bypass at the Graph level — flash rides the standard TileGraphOp →
assemble pass like every other kernel.

Scope (:func:`warp_chain_eligible`): fp16/bf16, causal or non-causal, `D%16==0`; static
`S%16==0` OR symbolic `seq_len`; equal-head OR GQA. The validated reference kernel
(`tests/compiler/e2e/test_flash_tensorcore_reference.py`) is the oracle.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import BF16, F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import FragmentScale, LdmatrixLoad, RegFragment, RegStore, Smem, Sync
from deplodock.compiler.ir.stmt import Body, Load, Mma
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, AtomTile, SerialTile, TileGraphOp, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_carry
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._frag_softmax import (
    FragGeom,
    realize_boundary_mask,
    realize_fragment_softmax,
    realize_score_mask,
)
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import CarryScope, Role


def _static(d) -> int | None:
    """The static extent of a shape entry, or ``None`` for a symbolic dim."""
    if isinstance(d, int):
        return d
    f = getattr(d, "as_static", None)
    return f() if (f is not None and getattr(d, "is_static", True)) else None


def warp_chain_eligible(*, B: int, H: int, S: int, D: int, group: int, causal: bool, mask: bool, symbolic: bool) -> bool:
    """The fused-TC-flash scope: 16-bit (fp16 / bf16), causal **or** non-causal (masked at
    the score fragment), ``D % 16 == 0``. A **symbolic** ``seq_len`` is accepted — the KV-stream
    / query-tile extents ceil-div the runtime ``seq_len``, the partial final tile is masked at the
    score fragment + clamped loads + guarded store — and drops the ``S % 16 == 0`` requirement. A
    STATIC ``S`` still requires ``S % 16 == 0``. **GQA** (``group > 1``) is accepted — K/V at
    ``head // group``. Additive-mask falls back to the scalar chain."""
    if mask or group < 1 or H % group != 0 or D % 16 != 0 or not (16 <= D <= 256):
        return False
    if symbolic:
        return True
    return S % 16 == 0 and S >= 16


def _add(*terms):
    """Sum a list of int / Expr terms into one Expr (dropping literal zeros)."""
    out = None
    for t in terms:
        e = Literal(t, "int") if isinstance(t, int) else t
        if isinstance(e, Literal) and e.value == 0:
            continue
        out = e if out is None else BinaryExpr("+", out, e)
    return out if out is not None else Literal(0, "int")


def _mul(a, b: int):
    return _add() if b == 0 else (a if b == 1 else BinaryExpr("*", a if not isinstance(a, int) else Literal(a, "int"), Literal(b, "int")))


def realize_flash(op: TileGraphOp) -> TileOp:
    """The fused TC flash as a ``TileOp`` (``GridTile > WarpTile > [init] SerialTile [epilogue]``),
    assembled through the generic carry assembler (``assemble_carry``) and lowered by the standard
    kernel passes. Driven by the **logical flash TileGraph** the offer shim ``split/005_warp_chain``
    attached: the geometry (q/k/v/out buffers + ``(B,H,S,D)`` + GQA + causal + symbolic ``seq``) is
    derived from the op's logical gmem ``buffers``, and the streaming online-softmax twisted
    ``Monoid`` carrier rides ``op.flash``. The QK^T / P@V mma codegen is ``kernel/005``'s shared
    AtomTile lowering; the softmax phases are ``realize_fragment_softmax(carrier)``."""
    block = op.tilegraph.blocks[0]
    buffers = op.buffers
    out = block.writes[0].buffer
    rank4 = [n for n, b in buffers.items() if len(b.shape) == 4 and n != out]
    q, k, v = rank4[0], rank4[1], rank4[2]
    causal = any("ninf" in n for n in buffers)
    qshape = buffers[q].shape
    B, H, D = _static(qshape[0]), _static(qshape[1]), _static(qshape[3])
    S = _static(qshape[2])
    seq_var = None if S is not None else next(iter(qshape[2].expr.free_vars()))
    group = H // _static(buffers[k].shape[1])  # GQA: q-heads / kv-heads
    carrier = op.flash  # the twisted online-softmax Monoid (split read it off dag.chain.carrier)
    atom = ATOM_REGISTRY["mma_m16n8k16_bf16" if buffers[q].dtype == BF16 else "mma_m16n8k16_f16"]
    qk_bt, pv_bt = True, False  # QK^T transposed-B; P@V canonical-B (v1 m16n8k16)
    atom_m, atom_n, atom_k = atom.shape
    assert (atom_m, atom_n, atom_k) == (16, 8, 16), "v1 warp-chain fragment layout assumes m16n8k16"

    atom_shape = atom.shape
    ab_dt = atom.operand_dtype("a")  # the 16-bit operand dtype (F16 / BF16)
    scale = f"{1.0 / math.sqrt(D)!r}f"
    kt = D // atom_k  # QK^T K-tiles (reduce over D)
    nd = D // atom_n  # P@V N-tiles (output over D)

    symbolic = seq_var is not None
    s_dim = Dim(seq_var).ceil_div(16) if symbolic else Dim(S // 16)
    seq = Var(seq_var) if symbolic else Literal(S, "int")

    bh, qb, kv = Var("bh"), Var("qb"), Var("kv")
    row_stride = _mul(seq, D) if symbolic else Literal(S * D, "int")
    base_q = BinaryExpr("*", bh, row_stride)
    if group > 1:
        h_kv = H // group
        bh_kv = _add(
            _mul(BinaryExpr("//", bh, Literal(H, "int")), h_kv),
            BinaryExpr("//", BinaryExpr("%", bh, Literal(H, "int")), Literal(group, "int")),
        )
        base_kv = BinaryExpr("*", bh_kv, row_stride)
    else:
        base_kv = base_q
    qrow = _add(base_q, _mul(qb, 16 * D))  # query-row base (q-head)
    kvrow = _add(base_kv, _mul(kv, 16 * D))  # kv-tile base (kv-head under GQA)

    def ld(frag, buf, src_index, role, *, b_trans=False, staged=False, ldm=D):
        return LdmatrixLoad(frag=frag, src_buffer=buf, src_index=(src_index,), role=role, ldm=ldm, staged=staged, b_trans=b_trans)

    geom = FragGeom(
        atom_m=atom_m,
        atom_n=atom_n,
        score_frags=tuple(f"Sf{nt}_frag" for nt in range(2)),
        prob_frags=tuple(f"Pf{nt}" for nt in range(2)),
        accum_frags=tuple(f"Of{n}" for n in range(nd)),
    )
    fs = realize_fragment_softmax(carrier, geom=geom)

    # carried state (the realizer's m/l Inits) + the O accumulators + the C→A smem slab.
    init: list = list(fs.init)
    init += [RegFragment(name=f"Of{n}", role="c", shape=atom_shape, dtype=F32) for n in range(nd)]
    init.append(Smem(name="flash_pv_smem", extents=(16, 16), dtype=cuda_name(ab_dt), align=16))

    # produce — QK^T: 2 N-atoms, reduce over kt K-tiles → the INLINE score fragments, scaled.
    def _qk_guards(nt: int) -> dict:
        if not symbolic:
            return {}
        return {"m_guard": (_mul(qb, 16), seq), "n_guard": (_add(_mul(kv, 16), nt * 8), seq)}

    produce: list = []
    for nt in range(2):
        qk_mma = Mma(c=f"Sf{nt}", a=f"qv{nt}", b=f"kc{nt}", atom=atom, b_trans=qk_bt, **_qk_guards(nt))
        if kt > 1:
            ko = Var(f"ko{nt}")
            rbody = (
                Load(name=f"qv{nt}", input=q, index=(_add(qrow, _mul(ko, 16)),)),
                Load(name=f"kc{nt}", input=k, index=(_add(kvrow, nt * 8 * D, _mul(ko, 16)),)),
                qk_mma,
            )
            cellbody: tuple = (SerialTile(axis=Axis(f"ko{nt}", Dim(kt)), body=Body(rbody), kind="plain"),)
        else:
            cellbody = (
                Load(name=f"qv{nt}", input=q, index=(qrow,)),
                Load(name=f"kc{nt}", input=k, index=(_add(kvrow, nt * 8 * D),)),
                qk_mma,
            )
        produce.append(AtomTile(axes=(Axis("qm", Dim(atom_m)), Axis("qn", Dim(atom_n))), body=Body(cellbody), atom=atom))
    produce += [FragmentScale(frag=f"Sf{nt}_frag", top=scale, bot=scale) for nt in range(2)]
    kv_col_bases = tuple(_add(_mul(kv, 16), nt * 8) for nt in range(2))
    if causal:
        produce += realize_score_mask(geom, q_row_base=_mul(qb, 16), kv_col_bases=kv_col_bases)
    if symbolic:
        produce += realize_boundary_mask(geom, kv_col_bases=kv_col_bases, bound=seq)

    # merge / rescale — the carrier's new state + the O *= alpha twist (fragment realizer).
    merge: list = list(fs.merge)
    rescale: list = list(fs.rescale)

    # handoff — the C→A edge: write P to the smem slab, ldmatrix it back as A.
    handoff: list = [
        RegStore(
            dst_buffer="flash_pv_smem", dst_index=(Literal(0, "int"), Literal(nt * 8, "int")), frag=f"Pf{nt}", shape=atom_shape, ldm=16
        )
        for nt in range(2)
    ]
    handoff.append(Sync())
    handoff.append(RegFragment(name="pa", role="a", shape=atom_shape, dtype=ab_dt))
    handoff.append(ld("pa", "flash_pv_smem", Literal(0, "int"), "a", staged=True, ldm=16))

    # consume — P@V: A = P (live ``pa``), B = V (gmem-direct), accumulate into O over the KV tile.
    pv_kzero = {"k_zero": (_mul(kv, 16), seq)} if symbolic else {}
    consume: list = []
    for n in range(nd):
        cell = (
            Load(name=f"vv{n}", input=v, index=(_add(kvrow, n * 8),)),
            Mma(c=f"Of{n}", a="pa", b=f"vv{n}", atom=atom, b_trans=pv_bt, **pv_kzero),
        )
        consume.append(AtomTile(axes=(Axis("am", Dim(atom_m)), Axis("an", Dim(atom_n))), body=Body(cell), atom=atom))
    consume.append(Sync())

    update: list = list(fs.update)

    # epilogue — O /= l (normalize) then store, guarding the partial query tile's rows.
    store_mguard = (_mul(qb, 16), seq) if symbolic else None
    epilogue: list = []
    for n in range(nd):
        epilogue.append(fs.epilogue[n])
        epilogue.append(
            RegStore(dst_buffer=out, dst_index=(_add(qrow, n * 8),), frag=f"Of{n}", shape=atom_shape, ldm=D, m_guard=store_mguard)
        )

    carry = CarryScope(
        axis=Axis("kv", s_dim),
        init=tuple(init),
        produce=tuple(produce),
        merge=tuple(merge),
        rescale=tuple(rescale),
        handoff=tuple(handoff),
        consume=tuple(consume),
        update=tuple(update),
        epilogue=tuple(epilogue),
    )
    parallel_layers = [
        (Axis("w", Dim(1)), Role.WARP),
        (Axis("qb", s_dim), Role.BLOCK),
        (Axis("bh", Dim(B * H)), Role.BLOCK),
    ]
    body = assemble_carry(carry, parallel_layers=parallel_layers)
    name = op.name if op.name.startswith("k_") else f"k_{op.name}"
    return TileOp(body=body, name=name, knobs={})
