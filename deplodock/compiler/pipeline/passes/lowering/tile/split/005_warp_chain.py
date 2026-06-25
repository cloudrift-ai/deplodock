"""Pre-build pass: emit the fused tensor-core flash for an eligible streaming nest.

``plans/tensor-core-streaming-flash-mma.md`` Phase 2.3 + Phase 3. Runs FIRST (before
``010_split_demoted``), on the un-tiled flash ``LoopOp``: when ``DEPLODOCK_CHAIN=1`` and
the nest is a streaming ``MONOID(SEMIRING)`` chain (``dag.chain``) in the v1 fused-TC
scope (fp16, non-causal, equal-head, ``D%16==0``, ``S%16==0``), it replaces the LoopOp
with a single ``CudaOp`` running the warp-chain kernel (``_warp_chain.assemble_warp_chain``
— the validated FA-2 kernel) and the engine splices it. Out of scope ⇒ ``RuleSkipped``,
so the flash falls through to ``chain_build`` (scalar) / the materialized path — the
deployed default is unchanged (this only fires under the explicit pin).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._warp_chain import assemble_warp_chain, warp_chain_eligible
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import CHAIN

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

PATTERN = [Pattern("root", LoopOp)]


def _chain_pinned() -> bool:
    raw = CHAIN.raw()
    return raw is not None and CHAIN.parse(raw)


def _static(d) -> int | None:
    """The static extent of a shape entry, or ``None`` for a symbolic dim."""
    if isinstance(d, int):
        return d
    as_static = getattr(d, "as_static", None)
    if as_static is not None and getattr(d, "is_static", True):
        return as_static()
    return None


def _flash_params(op: LoopOp):
    """``(B, H, S, D, group)`` for an eligible flash LoopOp, or ``None`` (out of scope:
    GQA / mask / causal / symbolic / non-fp16). The 3 rank-4 inputs in declared order are
    Q/K/V; a 4th rank-4 input is an additive mask, a ``*ninf*`` input is the causal bias."""
    rank4 = [(n, t) for n, t in op.inputs.items() if len(t.shape) == 4]
    if len(rank4) != 3:  # a mask adds a 4th rank-4 input
        return None
    if any("ninf" in n for n in op.inputs):  # causal bias
        return None
    (qn, q), (_kn, k), (_vn, _v) = rank4
    if q.dtype != F16:
        return None
    dims = [_static(d) for d in q.shape]
    kdims = [_static(d) for d in k.shape]
    if any(d is None for d in dims) or kdims[1] is None or kdims[1] == 0:
        return None
    B, H, S, D = dims
    group = H // kdims[1]
    return B, H, S, D, group


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
    B, H, S, D, group = params
    if not warp_chain_eligible(B=B, H=H, S=S, D=D, group=group, causal=False, mask=False, symbolic=False):
        raise RuleSkipped("flash shape out of the v1 warp-chain scope")
    return assemble_warp_chain(op, B=B, H=H, S=S, D=D)
