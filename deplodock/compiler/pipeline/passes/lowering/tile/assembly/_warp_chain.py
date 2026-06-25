"""The warp-chain assembler — build the fused tensor-core flash as **kernel-IR**
(``plans/tensor-core-streaming-flash-mma.md`` Phase 2.3 + Phase 3).

Produces a :class:`KernelOp` (NOT a source string) that the standard
``cuda/010_lower_kernelop`` renders via ``render_kernelop``: the QK^T / P@V mma + the
A/V ldmatrix loads fall out of the **same** kernel-IR ops as the warp-tier matmul
(``MmaSyncPtx`` / ``LdmatrixLoad`` / ``RegStore``), and the fragment online-softmax is
the ``FragmentRowReduce`` / ``FragmentExp`` / ``FragmentScale`` ops + the carried
``m``/``l`` recurrence (``Init`` + ``Reassign``). The validated reference kernel
(``tests/compiler/e2e/test_flash_tensorcore_reference.py``) is the oracle.

Geometry: one warp per 16 query rows; Q/K/V/O are addressed as flat ``(B·H·S·D)``
buffers (gmem-direct mma operands — no smem staging of Q/K/V), the score ``P`` rides an
smem slab for the C→A handoff. v1 scope: fp16, non-causal, equal-head, ``D%16==0``,
``S%16==0`` (:func:`warp_chain_eligible`).
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    FragmentExp,
    FragmentRowReduce,
    FragmentScale,
    KernelOp,
    LdmatrixLoad,
    MmaSyncPtx,
    Reassign,
    RegFragment,
    RegStore,
    Smem,
    Sync,
)
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Mma
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, GridTile, SerialTile, WarpTile
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell


def _classify_cell(a_buf, a_idx, b_buf, b_idx, k_name, out_index) -> Mma:
    """Run the algebraic ``atomize`` move on one ``[Load, Load, mul, Accum]`` cell — the
    operand A/B assignment, the ``b_trans`` (transposed-B Q@K^T), and the atom all fall
    out of ``classify_matmul_operands`` (the same decision the warp-tier matmul makes),
    instead of being hard-coded. The warp-chain build reads the layout off the returned
    ``Mma`` (Phase 2: ``atomize`` composes over the two cells)."""
    atom = ATOM_REGISTRY["mma_m16n8k16_f16"]
    cell = (
        Load(name="ca", input=a_buf, index=a_idx),
        Load(name="cb", input=b_buf, index=b_idx),
        Assign(name="cm", op=ElementwiseImpl("multiply"), args=("ca", "cb")),
        Accum(name="cc", value="cm"),
    )
    return next(s for s in atomize_cell(cell, atom=atom, k_name=k_name, write=None, out_index=out_index) if isinstance(s, Mma))


def warp_chain_eligible(*, B: int, H: int, S: int, D: int, group: int, causal: bool, mask: bool, symbolic: bool) -> bool:
    """The v1 fused-TC-flash scope: static fp16, non-causal, equal-head, both extents a
    multiple of 16. Everything else falls back to the scalar chain."""
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


def build_warp_chain_kernelop(loop_op: LoopOp, *, q: str, k: str, v: str, out: str, B: int, H: int, S: int, D: int) -> KernelOp:
    """The fused TC flash kernel-IR for ``(B,H,S,D)`` fp16, non-causal."""
    # The two cells' operand layout falls out of the ``atomize`` move: QK^T is a
    # transposed-B Q@K^T (``b_trans`` derived via the score's M/N coords), P@V is a
    # canonical-B P@V (A = P from the smem slab, B = V). The atom (cell shape + dtype) is
    # read off the resulting ``Mma``, not hard-coded — the v1 path assumes ``m16n8k16``.
    qk = _classify_cell(q, (Var("m"), Var("dd")), k, (Var("kv"), Var("dd")), "dd", (Var("m"), Var("kv")))
    pv = _classify_cell("flash_pv_smem", (Var("m"), Var("kv")), v, (Var("kv"), Var("d")), "kv", None)
    atom_m, atom_n, atom_k = qk.atom.shape
    ab_dt = qk.atom.operand_dtype("a").name
    assert (atom_m, atom_n, atom_k) == (16, 8, 16), "v1 warp-chain fragment layout assumes m16n8k16"

    atom_shape = qk.atom.shape
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

    body: list = []
    # --- carried state + the Q A-fragments (loaded once, reused across the KV stream) ---
    body.append(Init(name="m0", op=mx, dtype=F32))
    body.append(Init(name="m1", op=mx, dtype=F32))
    body.append(Init(name="l0", op=add, dtype=F32))
    body.append(Init(name="l1", op=add, dtype=F32))
    for n in range(nd):
        body.append(RegFragment(name=f"Of{n}", role="c", shape=atom_shape, dtype=F32))
    for t in range(kt):
        body.append(RegFragment(name=f"qa{t}", role="a", shape=atom_shape, dtype=F16))
        body.append(ld(f"qa{t}", q, _add(qrow, t * 16), "a"))
    body.append(Smem(name="flash_pv_smem", extents=(16, 16), dtype=cuda_name(F16), align=16))

    # --- the KV stream body ---
    kvb: list = []
    # QK^T: 2 N-atoms (kv cols 0-7 / 8-15), accumulate over kt K-tiles.
    for nt in range(2):
        kvb.append(RegFragment(name=f"Sf{nt}", role="c", shape=atom_shape, dtype=F32))
        for t in range(kt):
            kvb.append(RegFragment(name=f"kb{nt}_{t}", role="b", shape=atom_shape, dtype=F16))
            kvb.append(ld(f"kb{nt}_{t}", k, _add(kvrow, _mul(Literal(1, "int"), nt * 8 * D), t * 16), "b", b_trans=qk.b_trans))
            kvb.append(MmaSyncPtx(c_frag=f"Sf{nt}", a_frag=f"qa{t}", b_frag=f"kb{nt}_{t}", shape=atom_shape, ab_dtype=ab_dt))
        kvb.append(FragmentScale(frag=f"Sf{nt}", top=scale, bot=scale))
    # fragment softmax: rowmax -> m_new / alpha; P = exp(S - m_new); rowsum -> l.
    kvb.append(FragmentRowReduce(top="r0", bot="r1", frags=("Sf0", "Sf1"), op=mx))
    kvb.append(Assign(name="mn0", op=mx, args=("m0", "r0")))
    kvb.append(Assign(name="mn1", op=mx, args=("m1", "r1")))
    kvb.append(Assign(name="dm0", op=ElementwiseImpl("subtract"), args=("m0", "mn0")))
    kvb.append(Assign(name="dm1", op=ElementwiseImpl("subtract"), args=("m1", "mn1")))
    kvb.append(Assign(name="a0", op=ElementwiseImpl("exp"), args=("dm0",)))
    kvb.append(Assign(name="a1", op=ElementwiseImpl("exp"), args=("dm1",)))
    for nt in range(2):
        kvb.append(FragmentExp(out=f"Pf{nt}", src=f"Sf{nt}", top_sub="mn0", bot_sub="mn1"))
    kvb.append(FragmentRowReduce(top="s0", bot="s1", frags=("Pf0", "Pf1"), op=add))
    kvb.append(Assign(name="lm0", op=ElementwiseImpl("multiply"), args=("l0", "a0")))
    kvb.append(Assign(name="lm1", op=ElementwiseImpl("multiply"), args=("l1", "a1")))
    kvb.append(Assign(name="ln0", op=add, args=("lm0", "s0")))
    kvb.append(Assign(name="ln1", op=add, args=("lm1", "s1")))
    kvb.append(Reassign(name="l0", value="ln0"))
    kvb.append(Reassign(name="l1", value="ln1"))
    for n in range(nd):
        kvb.append(FragmentScale(frag=f"Of{n}", top="a0", bot="a1"))  # rescale O *= alpha
    # C->A handoff: write P C-fragment to the smem slab, ldmatrix it back as the A operand.
    for nt in range(2):
        kvb.append(
            RegStore(
                dst_buffer="flash_pv_smem", dst_index=(Literal(0, "int"), Literal(nt * 8, "int")), frag=f"Pf{nt}", shape=atom_shape, ldm=16
            )
        )
    kvb.append(Sync())  # the warp must finish writing ps before ldmatrix reads it back
    kvb.append(RegFragment(name="pa", role="a", shape=atom_shape, dtype=F16))
    kvb.append(ld("pa", "flash_pv_smem", Literal(0, "int"), "a", staged=True, ldm=16))  # ps row stride = BN = 16, not D
    # P@V: A = P (smem), B = V (gmem-direct), accumulate into O over the KV tile.
    for n in range(nd):
        kvb.append(RegFragment(name=f"vb{n}", role="b", shape=atom_shape, dtype=F16))
        kvb.append(ld(f"vb{n}", v, _add(kvrow, n * 8), "b", b_trans=pv.b_trans))
        kvb.append(MmaSyncPtx(c_frag=f"Of{n}", a_frag="pa", b_frag=f"vb{n}", shape=atom_shape, ab_dtype=ab_dt))
    kvb.append(Sync())  # finish reading ps (the ldmatrix) before the next KV tile overwrites it
    kvb.append(Reassign(name="m0", value="mn0"))
    kvb.append(Reassign(name="m1", value="mn1"))
    body.append(SerialTile(axis=Axis("kv", Dim(S // 16)), body=Body(tuple(kvb)), kind="serial_outer"))

    # --- epilogue: O /= l, store. ---
    for n in range(nd):
        body.append(FragmentScale(frag=f"Of{n}", top="(1.0f/l0)", bot="(1.0f/l1)"))
        body.append(RegStore(dst_buffer=out, dst_index=(_add(qrow, n * 8),), frag=f"Of{n}", shape=atom_shape, ldm=D))

    warp = WarpTile(axes=(Axis("w", Dim(1)),), body=Body(tuple(body)))
    grid = GridTile(axes=(Axis("bh", Dim(B * H)), Axis("qb", Dim(S // 16))), body=Body((warp,)))
    return KernelOp(name=loop_op.name if loop_op.name.startswith("k_") else f"k_{loop_op.name}", body=Body((grid,)))


def assemble_warp_chain(loop_op: LoopOp, *, B: int, H: int, S: int, D: int) -> Graph:
    """Build a single-``KernelOp`` ``Graph`` fragment for the fused TC flash — the q/k/v
    inputs + one kernel node. Spliced by the engine; the standard cuda lowering renders it."""
    rank4 = [n for n, t in loop_op.inputs.items() if len(t.shape) == 4]
    q_id, k_id, v_id = rank4[0], rank4[1], rank4[2]
    out_t = next(iter(loop_op.outputs.values()))
    kop = build_warp_chain_kernelop(loop_op, q=q_id, k=k_id, v=v_id, out=out_t.name, B=B, H=H, S=S, D=D)

    g = Graph()
    for nid in (q_id, k_id, v_id):
        t = loop_op.inputs[nid]
        g.add_node(op=InputOp(), inputs=[], output=Tensor(nid, tuple(t.shape), F16), node_id=nid)
    g.add_node(op=kop, inputs=[q_id, k_id, v_id], output=Tensor(out_t.name, tuple(out_t.shape), out_t.dtype), node_id=out_t.name)
    g.outputs = [out_t.name]
    return g
