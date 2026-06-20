"""Lift an intact ``SdpaOp`` into the fused flash-attention ``LoopOp`` nest.

The ``SdpaOp`` reaches this lifting stage only when ``frontend/decomposition/
010_sdpa`` deferred it (FLASH on + flash-eligible) — both sites consult the same
:func:`~._flash.flash_enabled` / :func:`~._flash.flash_shape_eligible`, so an
ineligible SDPA was already decomposed and never arrives here. The nest itself
(the streaming online-softmax recurrence on the ``FlashCombine`` carrier) is built
by :func:`~._flash.build_flash_frag`. See ``_flash`` for scope + the recurrence.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.frontend.ir import SdpaOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.loop.lifting._flash import build_flash_frag, flash_enabled, flash_shape_eligible

PATTERN = [Pattern("root", SdpaOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    graph = match.graph
    if not flash_enabled():
        # FLASH off — 010_sdpa should have decomposed this. Reaching lifting with
        # an intact SdpaOp and FLASH off means no rule lowered it; skip and let
        # the pipeline surface the un-lowered node rather than silently fuse.
        raise RuleSkipped("FLASH off — SdpaOp lowering belongs to 010_sdpa")
    q_id, k_id, v_id = root.inputs[0], root.inputs[1], root.inputs[2]
    has_mask = len(root.inputs) > 3
    q_shape = graph.nodes[q_id].output.shape
    k_shape = graph.nodes[k_id].output.shape
    v_shape = graph.nodes[v_id].output.shape
    if not flash_shape_eligible(q_shape, k_shape, v_shape, has_mask=has_mask):
        raise RuleSkipped("flash: SDPA not eligible (mask / GQA / symbolic non-seq) — should have been decomposed by 010_sdpa")
    return build_flash_frag(q_id, k_id, v_id, q_shape, k_shape, v_shape, root.output, causal=bool(root.op.is_causal))
