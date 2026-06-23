"""Build pass (deterministic) — seed the enumeration: ``LoopOp → TileGraphOp``.

The first half of the per-family enumeration split (``plans/tile-ir-block-dag.md``,
RF/F2). Derives the iteration DAG (``iter_dag``) and classifies the regime
(``_tree.classify``), then emits a **seed** ``TileGraphOp`` (``tilegraph is None``)
carrying the ``dag`` + regime the downstream tile passes' offer fns read. No fork
— one deterministic rewrite. A regime the moves can't build (coop / flash) raises
here, so the ``LoopOp`` never enters the tile-pass chain.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag

PATTERN = [Pattern("root", LoopOp)]

# Regimes the move set currently builds. MONOID (coop) / TWISTED_MONOID (flash) are
# recognised by classify but not yet built — they raise (quarantined).
_BUILDABLE = (AlgebraKind.MAP, AlgebraKind.SEMIRING)


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    """Seed a ``TileGraphOp`` from the fused ``LoopOp`` — the dag + regime the tile
    passes fork on. The carry-forward ``LoopOp`` knobs ride ``op.knobs`` (the engine
    merges them forward)."""
    loop_op: LoopOp = root.op
    dag = iter_dag(loop_op)
    regime = classify(dag)
    if regime is None or regime.algebra not in _BUILDABLE:
        raise RuleSkipped(f"move composer cannot lower kernel {loop_op.name!r}")
    return TileGraphOp(
        name=loop_op.name,
        dag=dag,
        algebra=regime.algebra,
        target_names=regime.target_names,
        leading=tuple(dag.leading),
        seed_key=loop_op.body.structural_key(),
    )
