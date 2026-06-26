"""Pre-build pass: the **eligibility / offer SHIM** for the fused tensor-core flash.

Runs FIRST (before ``010_split_demoted``), on the un-tiled flash ``LoopOp``: when the nest
is a streaming ``MONOID(SEMIRING)`` chain (``dag.chain``) in the fused-TC scope (fp16/bf16,
causal or non-causal, equal-head or GQA, ``D%16==0``; static ``S%16==0`` OR symbolic
``seq_len``), it replaces the LoopOp with a ``TileGraphOp`` carrying the **logical FA-2
TileGraph** (``seed_graph`` — the un-tiled streaming-flash algorithm over the q/k/v/o buffers)
plus the twisted online-softmax carrier ``Monoid`` on ``flash``. It holds NO assembly logic —
the assembly pass (``assembly/010_assemble``) realizes it via ``assembly/_flash.realize_flash``
(geometry derived from the logical ``buffers``) through the **generic carry assembler**
(``assemble_carry``, the same tower path matmul / reduce use), so flash rides the standard
TileGraphOp → assemble pipeline like every other kernel.

**Routing (Phase 3 of ``plans/smem-tiled-symbolic-flash.md``):** a **symbolic** ``seq_len``
flash fires the warp chain BY DEFAULT (the tensor-core / smem-shared-K/V flash is the
deployed symbolic default — the ~100× win over the scalar streaming nest at seq=512). A
**static** flash stays a ``DEPLODOCK_CHAIN=1`` opt-in (greedy keeps the scalar nest there
until the static search-fork integration). Out of scope ⇒ ``RuleSkipped``, so the flash
falls through to ``chain_build`` (scalar) / the materialized path — the correct fallback for
the symbolic shapes the warp chain declines (fp32, odd ``D``, additive mask).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from deplodock.compiler.dtype import BF16, F16
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import Buffer, Space, TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._flash import warp_chain_eligible
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

PATTERN = [Pattern("root", LoopOp)]


def _chain_pinned() -> bool:
    return fam.pin_inline_chain()


def _static(d) -> int | None:
    """The static extent of a shape entry, or ``None`` for a symbolic dim."""
    if isinstance(d, int):
        return d
    as_static = getattr(d, "as_static", None)
    if as_static is not None and getattr(d, "is_static", True):
        return as_static()
    return None


def _flash_params(op: LoopOp):
    """``(B, H, S, D, group, dtype, causal, seq_var)`` for an eligible flash LoopOp, or
    ``None`` (out of scope: GQA / additive mask / non-16-bit). The 3 rank-4 inputs in
    declared order are Q/K/V; a 4th rank-4 input is an additive mask (out of scope), a
    ``*ninf*`` input is the **causal** bias (the ``kv ≤ m`` Select — Phase 5, masked at
    the score fragment). The 16-bit operand dtype (``F16`` / ``BF16``) selects the mma
    atom kind (Phase 4). A **symbolic** ``S`` (Q/K/V dim -2) is in scope (Phase 1): ``seq_var``
    is its runtime symbol name (``None`` for a static ``S``); B/H/D must stay static."""
    rank4 = [(n, t) for n, t in op.inputs.items() if len(t.shape) == 4]
    if len(rank4) != 3:  # an additive mask adds a 4th rank-4 input — out of scope
        return None
    causal = any("ninf" in n for n in op.inputs)  # the causal -inf bias (kv ≤ m Select)
    (qn, q), (_kn, k), (_vn, _v) = rank4
    if q.dtype not in (F16, BF16):
        return None
    B, H, S, D = (_static(q.shape[0]), _static(q.shape[1]), _static(q.shape[2]), _static(q.shape[3]))
    # The seq axis (dim -2) may be symbolic; B/H/D must be static (the fragment geometry).
    seq_var = None
    if S is None:
        fvs = q.shape[2].expr.free_vars()
        if len(fvs) != 1:  # a single runtime symbol drives the ceil-div extents
            return None
        seq_var = next(iter(fvs))
    if any(d is None for d in (B, H, D)):
        return None
    kdim1 = _static(k.shape[1])  # the kv-head count (static — GQA group ratio)
    if kdim1 is None or kdim1 == 0:
        return None
    group = H // kdim1
    return B, H, S, D, group, q.dtype, causal, seq_var


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    op: LoopOp = root.op
    dag = iter_dag(op)
    if not dag.streaming or dag.chain is None:
        raise RuleSkipped("not a streaming-flash chain")
    params = _flash_params(op)
    if params is None:
        raise RuleSkipped("flash shape out of the v1 warp-chain scope")
    B, H, S, D, group, _dt, causal, seq_var = params
    symbolic = seq_var is not None
    # Phase 3 — the warp chain is the **symbolic** default (the tensor-core / smem-shared K/V
    # flash; the scalar streaming `chain_build` stays the fallback for the non-eligible
    # symbolic shapes `070_coop_reduce` still handles: odd D, additive mask). A **static**
    # flash stays a `DEPLODOCK_CHAIN=1` opt-in (greedy keeps the scalar nest there — the
    # static search-fork integration is later work).
    if not symbolic and not _chain_pinned():
        raise RuleSkipped("static warp-chain flash is CHAIN-pinned only")
    if not warp_chain_eligible(B=B, H=H, S=S, D=D, group=group, causal=causal, mask=False, symbolic=symbolic):
        raise RuleSkipped("flash shape out of the v1 warp-chain scope")
    # This pass is the eligibility/offer SHIM: emit a ``TileGraphOp`` carrying the **logical
    # flash TileGraph** (``seed_graph`` — the un-restructured FA-2 algorithm over q/k/v/o, the
    # twisted online-softmax ``Monoid`` carrier intact at ``block.carrier``) with the kv stream
    # axis marked ``Schedule.carry`` — the representation that tells ``assembly/010_assemble`` to
    # realize this carrier at the FRAGMENT tier (the warp chain). Geometry is derived from the
    # logical ``buffers``; no flat spec, no ``flash`` marker, no separate assembler entry. ``dag``
    # is left unset so the enumeration forks skip the op (it flows straight to assembly).
    buffers: dict[str, Buffer] = {}
    for name, t in op.inputs.items():
        buffers[name] = Buffer(name=name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM)
    for t in op.outputs.values():
        buffers[t.name] = Buffer(name=t.name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM)
    tg = seed_graph(dag, kernel_name=op.name, buffers=buffers)
    tg = replace(tg, schedule=replace(tg.schedule, carry=frozenset({dag.chain.hinge_name})))
    return TileGraphOp(name=op.name, tilegraph=tg, buffers=buffers)
