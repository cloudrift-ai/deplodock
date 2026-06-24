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
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.tile.ir import TileGraphOp, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._fused import assemble_fused, fused_producer_blocks, is_fused_graph

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileOp | Graph:  # noqa: ARG001
    """Assemble the fully-tiled ``TileGraphOp`` into its ``TileOp`` tower. The variant
    knobs are already merged onto ``op.knobs`` (so ``base_knobs`` is empty); ``op.leading``
    is the per-CTA prologue ``assemble`` prepends. A still-logical seed (the free-axis tile
    not yet applied — its block ``domain`` still empty) is left for the enumeration passes
    to finish. The "fully tiled" test reads the materialization side's own state (a
    populated ``Block.domain``), never a search-side knob — assembly stays independent of
    the enumeration vocabulary.

    A multi-block ``TileGraph`` (the edge-placement ``GMEM`` cut, R7) assembles to a
    ``Graph`` of ``TileOp`` kernels the engine splices, the same shape the structural
    forks (``140_atomic_free_splitk``) already return. A **same-launch-group** multi-block
    ``TileGraph`` (the ``SMEM`` fused edge) assembles to **one** fused ``TileOp`` via
    ``assemble_fused`` — the producer rides the consumer's slab, so only the consumer must
    be tiled (the logical producer becomes the slab's ``compute`` phase)."""
    op: TileGraphOp = root.op
    if op.tilegraph is None:
        raise RuleSkipped("TileGraphOp not yet fully tiled (still a logical seed)")
    tg = op.tilegraph
    if is_fused_graph(tg):
        # The SMEM fused edge: only the consumer needs tiling; the producer blocks
        # stay logical (they fold into the consumer's slab compute phase).
        producers = fused_producer_blocks(tg)
        if any(not b.domain for b in tg.blocks if b.name not in producers):
            raise RuleSkipped("fused consumer not yet tiled")
        return assemble_fused(tg, knobs=op.knobs, base_knobs={}, kernel_name=op.name, leading=op.leading)
    if any(not b.domain for b in tg.blocks):
        raise RuleSkipped("TileGraphOp not yet fully tiled (still a logical seed)")
    return assemble_block(tg, knobs=op.knobs, base_knobs={}, kernel_name=op.name, leading=op.leading)
