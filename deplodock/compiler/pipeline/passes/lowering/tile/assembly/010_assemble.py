"""Assembly pass — the one deterministic ``TileGraph`` → ``TileOp`` step.

The second half of the block-DAG tile phase (``plans/tile-ir-block-dag.md``):
``enumeration/010_enumerate`` produces a ``TileGraphOp`` (a chosen ``Schedule``'s
``TileGraph``); this pass materializes it into the ``TileOp`` tower the kernel
passes lower. ``assemble`` is deterministic — every scheduling decision already
lives on the ``TileGraph`` / ``Schedule``, so there is no search here.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileGraphOp, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileOp:  # noqa: ARG001
    """Assemble the enumeration pass's built ``TileGraphOp`` into its ``TileOp``
    tower. The variant knobs are already merged onto ``op.knobs`` (so ``base_knobs``
    is empty here); ``op.leading`` is the per-CTA prologue ``assemble`` prepends. A
    still-in-flight seed (``tilegraph is None``) is left for ``040_lower`` to build."""
    op: TileGraphOp = root.op
    if op.tilegraph is None:
        raise RuleSkipped("TileGraphOp not yet built (still an enumeration seed)")
    return assemble_block(op.tilegraph, knobs=op.knobs, base_knobs={}, kernel_name=op.name, leading=op.leading)
