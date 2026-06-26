"""Pre-build pass: emit the fused tensor-core flash for an eligible streaming nest.

Runs FIRST (before
``010_split_demoted``), on the un-tiled flash ``LoopOp``: when ``DEPLODOCK_CHAIN=1`` and
the nest is a streaming ``MONOID(SEMIRING)`` chain (``dag.chain``) in the v1 fused-TC
scope (fp16, non-causal, equal-head, ``D%16==0``, ``S%16==0``), it replaces the LoopOp
with a single ``TileOp`` running the warp-chain kernel (``_warp_chain.assemble_warp_chain``
— the validated FA-2 kernel) and the engine splices it. Out of scope ⇒ ``RuleSkipped``,
so the flash falls through to ``chain_build`` (scalar) / the materialized path — the
deployed default is unchanged (this only fires under the explicit pin).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.dtype import BF16, F16
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Mma
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._warp_chain import assemble_warp_chain, warp_chain_eligible
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell
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


def rewrite(ctx: Context, root: Node, match) -> Graph:  # noqa: ARG001
    op: LoopOp = root.op
    if not _chain_pinned():
        raise RuleSkipped("warp-chain flash is CHAIN-pinned only")
    dag = iter_dag(op)
    if not dag.streaming or dag.chain is None:
        raise RuleSkipped("not a streaming-flash chain")
    params = _flash_params(op)
    if params is None:
        raise RuleSkipped("flash shape out of the v1 warp-chain scope")
    B, H, S, D, group, dt, causal, seq_var = params
    if not warp_chain_eligible(B=B, H=H, S=S, D=D, group=group, causal=causal, mask=False, symbolic=seq_var is not None):
        raise RuleSkipped("flash shape out of the v1 warp-chain scope")
    # Run the algebraic ``atomize`` move on the two cells HERE (the ``split`` phase may
    # import ``enumeration``; the ``assembly`` may not) — the operand layout (``b_trans``
    # transposed-B Q@K^T vs canonical-B P@V) + the atom fall out of the move, not hard-coded.
    kind = "mma_m16n8k16_bf16" if dt == BF16 else "mma_m16n8k16_f16"
    qk = _classify_cell("Q", (Var("m"), Var("dd")), "K", (Var("kv"), Var("dd")), "dd", (Var("m"), Var("kv")), kind=kind)
    pv = _classify_cell("P", (Var("m"), Var("kv")), "V", (Var("kv"), Var("d")), "kv", None, kind=kind)
    # The streaming online-softmax Monoid (state (m,l,O), partial (score,value)) — the
    # algebraic source the fragment realizer regenerates the softmax phases from. Reading
    # it here keeps the ``enumeration`` import in ``split``, not ``assembly``.
    return assemble_warp_chain(op, B=B, H=H, S=S, D=D, qk=qk, pv=pv, causal=causal, carrier=dag.chain.carrier, seq_var=seq_var)


def _classify_cell(a_buf, a_idx, b_buf, b_idx, k_name, out_index, *, kind="mma_m16n8k16_f16"):
    """Atomize one ``[Load, Load, mul, Accum]`` cell → its ``Mma`` (the A/B assignment,
    ``b_trans``, and atom from ``classify_matmul_operands`` — the same decision the
    warp-tier matmul makes). Phase 2: ``atomize`` composes over the two flash cells.
    ``kind`` selects the mma atom (``mma_m16n8k16_{f16,bf16}``, Phase 4)."""
    atom = ATOM_REGISTRY[kind]
    cell = (
        Load(name="ca", input=a_buf, index=a_idx),
        Load(name="cb", input=b_buf, index=b_idx),
        Assign(name="cm", op=ElementwiseImpl("multiply"), args=("ca", "cb")),
        Accum(name="cc", value="cm"),
    )
    return next(s for s in atomize_cell(cell, atom=atom, k_name=k_name, write=None, out_index=out_index) if isinstance(s, Mma))
