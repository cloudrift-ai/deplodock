"""Build pass (deterministic) — seed the enumeration: ``LoopOp → TileGraphOp``.

The first half of the per-family enumeration split (``plans/tile-ir-block-dag.md``,
RF/F3-b). Derives the iteration DAG (``iter_dag``) and classifies the regime
(``_tree.classify``), then emits a seed ``TileGraphOp`` carrying the **logical
(un-tiled) algorithm** (``_build.seed_graph`` — one ``Block`` whose ``compute`` is
the DAG's inner body, empty ``domain`` / ``Schedule``) plus the ``dag`` + regime the
downstream tile passes' offer fns read. The algorithm is then refined **in place** by
the tile passes' incremental body moves (F3-b: ``010_reduce_tile`` re-brackets K,
``030_register_tile`` σ-splits the free axes); nothing is built all-at-once. No fork
— one deterministic rewrite. A regime the moves can't build (coop / flash) raises
here, so the ``LoopOp`` never enters the tile-pass chain.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import Buffer, Space, TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag

PATTERN = [Pattern("root", LoopOp)]

# Regimes the move set builds: MAP / SEMIRING / MONOID + TWISTED_MONOID (flash, R6 —
# the streaming online-softmax nest, built by enumeration/017_streaming).
_BUILDABLE = (AlgebraKind.MAP, AlgebraKind.SEMIRING, AlgebraKind.MONOID, AlgebraKind.TWISTED_MONOID)


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    """Seed a ``TileGraphOp`` from the fused ``LoopOp`` — the logical algorithm + the
    dag + regime the tile passes fork on. The carry-forward ``LoopOp`` knobs ride
    ``op.knobs`` (the engine merges them forward)."""
    loop_op: LoopOp = root.op
    dag = iter_dag(loop_op)
    regime = classify(dag)
    if regime is None or regime.algebra not in _BUILDABLE:
        raise RuleSkipped(f"move composer cannot lower kernel {loop_op.name!r}")
    # Logical gmem Buffers (inputs + outputs) — the ``stage`` move reads operand
    # dtypes off these to size smem slabs; ``assemble`` stamps ``Source.dtype``.
    buffers: dict[str, Buffer] = {}
    for name, t in loop_op.inputs.items():
        buffers[name] = Buffer(name=name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM)
    for t in loop_op.outputs.values():
        buffers[t.name] = Buffer(name=t.name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM)
    return TileGraphOp(
        name=loop_op.name,
        tilegraph=seed_graph(dag, kernel_name=loop_op.name, buffers=buffers),
        dag=dag,
        algebra=regime.algebra,
        target_names=regime.target_names,
        leading=tuple(dag.leading),
        seed_key=loop_op.body.structural_key(),
        buffers=buffers,
    )
