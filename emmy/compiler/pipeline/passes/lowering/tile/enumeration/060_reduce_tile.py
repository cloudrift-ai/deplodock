"""Reduce-tile pass (fork) — the ``TileSerial`` move on a contraction axis.

For a ``SEMIRING`` seed, offer the carrier-licensed ``(bk, fk, splitk)`` K-tilings
(``_moves.reduce_offers`` → ``legal_decomps``) and fork — each option **applies the
reduce-decomposition body move** (``_build.reduce_decomp``: re-bracket K into the
``K_o`` / ``K_i`` tower in ``Block.compute``) to the stored algorithm and pins its
reduce-knob group. The first of the F3-b incremental body moves; the free-axis split
follows at ``100_register_tile``. A ``MAP`` nest has no contraction, so this passes
through (``RuleSkipped``).
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.dtype import F16
from emmy.compiler.graph import Node
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.stmt import Load, Loop, Write
from emmy.compiler.ir.tile.ir import TileGraphOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import mma_atom
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._build import reduce_decomp
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import reduce_knobs, reduce_offers

PATTERN = [Pattern("root", TileGraphOp)]


def _is_fp16_matmul(op: TileGraphOp) -> bool:
    """True iff every K-indexed operand ``Load`` resolves to an ``F16`` gmem buffer
    AND the reduce has no fused prologue / epilogue. Gates the fp16 half2
    accumulation window (the ``FK`` knob's scalar-matmul reinterpretation): fp32 /
    bf16 / mixed matmuls and prologue-fused reduces (SDPA P@V) keep the scalar
    fp32-accumulate path. Loop-IR Loads carry no dtype yet (stamped at
    ``kernel/030_stamp_types``), so the operand dtype comes off the logical gmem
    ``Buffer`` (``op.buffers``)."""
    if any(not isinstance(s, (Loop, Write)) for s in op.dag.inner_body):
        return False  # a fused scale / residual epilogue would interleave with the windowed flush
    k_loop = op.dag.k_node.loop
    k_name = k_loop.axis.name
    k_loads = [ld for ld in k_loop.body.iter_of_type(Load) if k_name in {v for e in ld.index for v in e.free_vars()}]
    if not k_loads:
        return False
    return all((b := op.buffers.get(ld.input)) is not None and b.dtype == F16 for ld in k_loads)


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.SEMIRING or mma_atom(op.knobs) is not None:
        raise RuleSkipped("reduce tile not applicable / warp tier")  # MAP / warp: no reduce axis to key on
    rk = fam.reduce_key(op.dag.k_node.loop.axis.name)
    if rk in op.knobs:
        raise RuleSkipped("reduce tile already pinned")
    offers = reduce_offers(op.dag)
    if not offers:
        raise RuleSkipped("no legal reduce tiling")
    fp16 = _is_fp16_matmul(op)
    out = []
    for bk, fk, sk in offers:
        knobs = {**op.knobs, **reduce_knobs(op.dag, (bk, fk, sk))}
        # fp16 matmul + an even FK window (FK == the stage chunk bk): reinterpret FK
        # as the half2 accumulation window — keep
        # the FK=1 fp32 K factorization (no K_f register fold) and stamp ``FKWIN`` so
        # kernel/015_pack_fk_window packs the even bk inner loop into __hfma2. The
        # register FK fold and the half2 window are mutually exclusive realizations
        # of the FK knob; for fp16 the window wins. fp32 / bf16 keep the fold.
        if fp16 and fk > 1 and fk == bk and bk % 2 == 0:
            knobs = {**knobs, rk: fam.enc_reduce(serial=bk, fold=1, cta=sk), "FKWIN": fk}
        tg = reduce_decomp(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
