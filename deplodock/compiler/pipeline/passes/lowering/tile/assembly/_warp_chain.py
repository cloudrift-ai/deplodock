"""The warp-chain assembler — build the fused tensor-core flash as a **TileOp**
(``plans/tensor-core-streaming-flash-mma.md`` Phase 2.3 + Phase 3, step 2 of the
"Generalized ``_tower``" migration).

Produces a :class:`TileOp` (``GridTile > WarpTile > [init] SerialTile [epilogue]``) that
flows through the **generic kernel passes** (``kernel/005``…``100`` → ``cuda``), NOT a
KernelOp that bypasses them — so the flash shares the matmul's own lowering chain. The
QK^T / P@V mma codegen goes through the **shared** ``kernel/005`` AtomTile lowering — the
``produce`` phase is a transposed-B **fragment-output** ``AtomTile`` (the score C-fragment
stays live) and ``consume`` is a **fragment-A** ``AtomTile`` (the C→A handoff), exactly the
two cell paths ``005`` grew; no hand-emitted ``MmaSyncPtx`` remains. ``005`` names the live
QK^T C-fragment ``Sf{nt}_frag``, which the softmax below reads. What is still authored here
is the **carrier**: the fragment online-softmax (``FragmentRowReduce`` / ``FragmentExp`` /
``FragmentScale``) + the carried ``m``/``l`` recurrence (``Init`` + ``Reassign``) + the
C→A smem handoff — all ``rewrite``-registered so they survive the SSA-rewriting passes. The
validated reference kernel (``tests/compiler/e2e/test_flash_tensorcore_reference.py``) is
the oracle.

Geometry: one warp per 16 query rows; Q/K/V/O are addressed as flat ``(B·H·S·D)``
buffers (gmem-direct mma operands — no smem staging of Q/K/V), the score ``P`` rides an
smem slab for the C→A handoff. v1 scope: fp16, non-causal, equal-head, ``D%16==0``,
``S%16==0`` (:func:`warp_chain_eligible`).
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    FragmentExp,
    FragmentRowReduce,
    FragmentScale,
    LdmatrixLoad,
    Reassign,
    RegFragment,
    RegStore,
    Smem,
    Sync,
)
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Assign, Body, Init, Load, Mma
from deplodock.compiler.ir.tile.ir import AtomTile, SerialTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import CarryScope, wrap_carry_tower


def warp_chain_eligible(*, B: int, H: int, S: int, D: int, group: int, causal: bool, mask: bool, symbolic: bool) -> bool:
    """The v1 fused-TC-flash scope: static 16-bit (fp16 / bf16 — Phase 4), non-causal,
    equal-head, both extents a multiple of 16. Everything else falls back to the scalar
    chain. (Dtype is gated upstream in ``_flash_params``; this checks only geometry.)"""
    return not symbolic and not causal and not mask and group == 1 and D % 16 == 0 and S % 16 == 0 and 16 <= D <= 256 and S >= 16


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


def build_warp_chain_tileop(
    loop_op: LoopOp, *, q: str, k: str, v: str, out: str, B: int, H: int, S: int, D: int, qk: Mma, pv: Mma
) -> TileOp:
    """The fused TC flash as a **TileOp** (``GridTile > WarpTile > [init] SerialTile [epilogue]``)
    for ``(B,H,S,D)`` fp16, non-causal — routed through the generic kernel passes
    (``kernel/005``…``100`` → ``cuda``), NOT a KernelOp bypassing them.

    ``qk`` / ``pv`` are the two cells' atomized ``Mma``s (built by the ``split`` pass via
    the ``atomize`` move — the layering keeps the ``enumeration`` ``atomize`` import out of
    ``assembly``): the operand ``b_trans`` (transposed-B Q@K^T vs canonical-B P@V) + the
    atom (cell shape + dtype) are read off them, not hard-coded (the v1 path assumes
    ``m16n8k16``)."""
    atom_m, atom_n, atom_k = qk.atom.shape
    assert (atom_m, atom_n, atom_k) == (16, 8, 16), "v1 warp-chain fragment layout assumes m16n8k16"

    atom_shape = qk.atom.shape
    ab_dt = qk.atom.operand_dtype("a")  # the 16-bit operand dtype (F16 / BF16) — Phase 4
    scale = f"{1.0 / math.sqrt(D)!r}f"
    kt = D // atom_k  # QK^T K-tiles (reduce over D)
    nd = D // atom_n  # P@V N-tiles (output over D)
    add = ElementwiseImpl("add")
    mx = ElementwiseImpl("maximum")

    bh, qb, kv = Var("bh"), Var("qb"), Var("kv")
    base = _mul(bh, S * D)  # (b,h) offset into the flat buffer
    qrow = _add(base, _mul(qb, 16 * D))  # query-row base
    kvrow = _add(base, _mul(kv, 16 * D))  # kv-tile base

    def ld(frag, buf, src_index, role, *, b_trans=False, staged=False, ldm=D):
        return LdmatrixLoad(frag=frag, src_buffer=buf, src_index=(src_index,), role=role, ldm=ldm, staged=staged, b_trans=b_trans)

    # The flash kv-stream is a MONOID(SEMIRING) ``CarryScope`` over ``kv`` (the streaming
    # accumulator ``O`` + the online-softmax stats ``m``/``l``): the two SEMIRING cells
    # (QK^T ``produce``, P@V ``consume``) bracket the carrier's ``merge``/``rescale``/
    # ``update``. ``wrap_carry_tower`` does the GridTile>WarpTile>[init]SerialTile[epilogue]
    # nesting — the streaming accumulator the generic single-cell assembly can't model.

    # carried state + the Q A-fragments (loaded once, reused across the KV stream).
    init: list = [
        Init(name="m0", op=mx, dtype=F32),
        Init(name="m1", op=mx, dtype=F32),
        Init(name="l0", op=add, dtype=F32),
        Init(name="l1", op=add, dtype=F32),
    ]
    init += [RegFragment(name=f"Of{n}", role="c", shape=atom_shape, dtype=F32) for n in range(nd)]
    init.append(Smem(name="flash_pv_smem", extents=(16, 16), dtype=cuda_name(ab_dt), align=16))

    # produce — QK^T: 2 N-atoms (kv cols 0-7 / 8-15), reduce over kt K-tiles → the INLINE
    # score fragments ``Sf{nt}``, scaled by 1/sqrt(D). Each N-atom is a **fragment-output
    # AtomTile** (transposed-B Q@K^T, no ``Write`` → the C-fragment stays live) that
    # ``kernel/005`` lowers — the mma codegen is the matmul's own pass, not hand-emitted.
    # ``005`` names the live C-fragment ``Sf{nt}_frag`` (the softmax below reads that).
    produce: list = []
    for nt in range(2):
        if kt > 1:
            ko = Var(f"ko{nt}")
            rbody = (
                Load(name=f"qv{nt}", input=q, index=(_add(qrow, _mul(ko, 16)),)),
                Load(name=f"kc{nt}", input=k, index=(_add(kvrow, nt * 8 * D, _mul(ko, 16)),)),
                Mma(c=f"Sf{nt}", a=f"qv{nt}", b=f"kc{nt}", atom=qk.atom, b_trans=qk.b_trans),
            )
            cellbody: tuple = (SerialTile(axis=Axis(f"ko{nt}", Dim(kt)), body=Body(rbody), kind="plain"),)
        else:
            cellbody = (
                Load(name=f"qv{nt}", input=q, index=(qrow,)),
                Load(name=f"kc{nt}", input=k, index=(_add(kvrow, nt * 8 * D),)),
                Mma(c=f"Sf{nt}", a=f"qv{nt}", b=f"kc{nt}", atom=qk.atom, b_trans=qk.b_trans),
            )
        produce.append(AtomTile(axes=(Axis("qm", Dim(atom_m)), Axis("qn", Dim(atom_n))), body=Body(cellbody), atom=qk.atom))
    produce += [FragmentScale(frag=f"Sf{nt}_frag", top=scale, bot=scale) for nt in range(2)]

    # merge — the carrier's new state from this tile: rowmax → m_new / alpha; P = exp(S - m_new); rowsum → l.
    merge: list = [
        FragmentRowReduce(top="r0", bot="r1", frags=("Sf0_frag", "Sf1_frag"), op=mx),
        Assign(name="mn0", op=mx, args=("m0", "r0")),
        Assign(name="mn1", op=mx, args=("m1", "r1")),
        Assign(name="dm0", op=ElementwiseImpl("subtract"), args=("m0", "mn0")),
        Assign(name="dm1", op=ElementwiseImpl("subtract"), args=("m1", "mn1")),
        Assign(name="alpha0", op=ElementwiseImpl("exp"), args=("dm0",)),
        Assign(name="alpha1", op=ElementwiseImpl("exp"), args=("dm1",)),
    ]
    merge += [FragmentExp(out=f"Pf{nt}", src=f"Sf{nt}_frag", top_sub="mn0", bot_sub="mn1") for nt in range(2)]
    merge += [
        FragmentRowReduce(top="s0", bot="s1", frags=("Pf0", "Pf1"), op=add),
        Assign(name="lm0", op=ElementwiseImpl("multiply"), args=("l0", "alpha0")),
        Assign(name="lm1", op=ElementwiseImpl("multiply"), args=("l1", "alpha1")),
        Assign(name="ln0", op=add, args=("lm0", "s0")),
        Assign(name="ln1", op=add, args=("lm1", "s1")),
        Reassign(name="l0", value="ln0"),
        Reassign(name="l1", value="ln1"),
    ]

    # rescale — the twist: O *= alpha before the P@V accumulation.
    rescale: list = [FragmentScale(frag=f"Of{n}", top="alpha0", bot="alpha1") for n in range(nd)]

    # handoff — the C→A edge: write the P C-fragment to the smem slab, ldmatrix it back as A.
    handoff: list = [
        RegStore(
            dst_buffer="flash_pv_smem", dst_index=(Literal(0, "int"), Literal(nt * 8, "int")), frag=f"Pf{nt}", shape=atom_shape, ldm=16
        )
        for nt in range(2)
    ]
    handoff.append(Sync())  # the warp must finish writing ps before ldmatrix reads it back
    handoff.append(RegFragment(name="pa", role="a", shape=atom_shape, dtype=ab_dt))
    handoff.append(ld("pa", "flash_pv_smem", Literal(0, "int"), "a", staged=True, ldm=16))  # ps row stride = BN = 16, not D

    # consume — P@V: A = P (the live ``pa`` fragment), B = V (gmem-direct), accumulate into O
    # over the KV tile. Each N-atom is a **fragment-A AtomTile** that ``kernel/005`` lowers
    # (one gmem B Load + the live A — the C→A handoff cell): the mma codegen is the matmul's
    # own pass, not hand-emitted here.
    consume: list = []
    for n in range(nd):
        cell = (
            Load(name=f"vv{n}", input=v, index=(_add(kvrow, n * 8),)),
            Mma(c=f"Of{n}", a="pa", b=f"vv{n}", atom=pv.atom, b_trans=pv.b_trans),
        )
        consume.append(AtomTile(axes=(Axis("am", Dim(atom_m)), Axis("an", Dim(atom_n))), body=Body(cell), atom=pv.atom))
    consume.append(Sync())  # finish reading ps (the ldmatrix) before the next KV tile overwrites it

    # update — commit the carrier max.
    update: list = [Reassign(name="m0", value="mn0"), Reassign(name="m1", value="mn1")]

    # epilogue — O /= l, store.
    epilogue: list = []
    for n in range(nd):
        epilogue.append(FragmentScale(frag=f"Of{n}", top="(1.0f/l0)", bot="(1.0f/l1)"))
        epilogue.append(RegStore(dst_buffer=out, dst_index=(_add(qrow, n * 8),), frag=f"Of{n}", shape=atom_shape, ldm=D))

    carry = CarryScope(
        axis=Axis("kv", Dim(S // 16)),
        init=tuple(init),
        produce=tuple(produce),
        merge=tuple(merge),
        rescale=tuple(rescale),
        handoff=tuple(handoff),
        consume=tuple(consume),
        update=tuple(update),
        epilogue=tuple(epilogue),
    )
    body = wrap_carry_tower(carry, warp_axes=(Axis("w", Dim(1)),), grid_axes=(Axis("bh", Dim(B * H)), Axis("qb", Dim(S // 16))))
    name = loop_op.name if loop_op.name.startswith("k_") else f"k_{loop_op.name}"
    return TileOp(body=body, name=name, knobs={})


def assemble_warp_chain(loop_op: LoopOp, *, B: int, H: int, S: int, D: int, qk: Mma, pv: Mma) -> Graph:
    """Build a single-``TileOp`` ``Graph`` fragment for the fused TC flash — the q/k/v
    inputs + one kernel node. Spliced by the engine; the generic kernel passes lower it.
    ``qk`` / ``pv`` are the cells' atomized ``Mma``s (the ``split`` pass ran the ``atomize``
    move)."""
    rank4 = [n for n, t in loop_op.inputs.items() if len(t.shape) == 4]
    q_id, k_id, v_id = rank4[0], rank4[1], rank4[2]
    out_t = next(iter(loop_op.outputs.values()))
    kop = build_warp_chain_tileop(loop_op, q=q_id, k=k_id, v=v_id, out=out_t.name, B=B, H=H, S=S, D=D, qk=qk, pv=pv)

    g = Graph()
    for nid in (q_id, k_id, v_id):
        t = loop_op.inputs[nid]
        g.add_node(op=InputOp(), inputs=[], output=Tensor(nid, tuple(t.shape), t.dtype), node_id=nid)
    g.add_node(op=kop, inputs=[q_id, k_id, v_id], output=Tensor(out_t.name, tuple(out_t.shape), out_t.dtype), node_id=out_t.name)
    g.outputs = [out_t.name]
    return g
