"""Assembly pass — the one deterministic ``TileGraph`` → ``TileOp`` step.

The second half of the block-DAG tile phase (``plans/tile-ir-block-dag.md``):
``enumeration/`` refines a stored, knob-invariant ``TileGraph`` in place — the body
moves (``reduce_decomp`` / ``free_tile``) σ-split the algorithm and the ``stage`` move
annotates ``Schedule.staged`` — and hands the fully-tiled ``TileGraphOp`` here. This
pass materializes it into the ``TileOp`` tower the kernel passes lower: ``assemble``
does the register/thread replication (``_wrap_tower``) + slab synthesis from the
``Schedule``. It is **deterministic** — every scheduling decision already lives on the
``TileGraph`` / ``Schedule``, so there is no search and no build here.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileGraphOp, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileOp:  # noqa: ARG001
    """Assemble the fully-tiled ``TileGraphOp`` into its ``TileOp`` tower. The variant
    knobs are already merged onto ``op.knobs`` (so ``base_knobs`` is empty); ``op.leading``
    is the per-CTA prologue ``assemble`` prepends. A still-logical seed (the free-axis tile
    not yet applied — its block ``domain`` still empty) is left for the enumeration passes
    to finish. The "fully tiled" test reads the materialization side's own state (a
    populated ``Block.domain``), never a search-side knob — assembly stays independent of
    the enumeration vocabulary."""
    op: TileGraphOp = root.op
    if op.tilegraph is None or not op.tilegraph.blocks[0].domain:
        raise RuleSkipped("TileGraphOp not yet fully tiled (still a logical seed)")
    return assemble_block(op.tilegraph, knobs=op.knobs, base_knobs={}, kernel_name=op.name, leading=op.leading)
