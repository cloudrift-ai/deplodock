"""Build pass (deterministic) — seed the enumeration: ``LoopOp → TileGraphOp``.

The first half of the per-family enumeration split. Derives the iteration DAG (``iter_dag``) and classifies the regime
(``_tree.classify``), then emits a seed ``TileGraphOp`` carrying the **logical
(un-tiled) algorithm** (``_build.seed_graph`` — one ``Block`` whose ``compute`` is
the DAG's inner body, empty ``domain`` / ``Schedule``) plus the ``dag`` + regime the
downstream tile passes' offer fns read. The algorithm is then refined **in place** by
the tile passes' incremental body moves (F3-b: ``060_reduce_tile`` re-brackets K,
``100_register_tile`` σ-splits the free axes); nothing is built all-at-once. No fork
— one deterministic rewrite. A regime the moves can't build (coop / flash) raises
here, so the ``LoopOp`` never enters the tile-pass chain.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.tile.ir import Buffer, Space, TileGraphOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._build import seed_graph
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._validate import validate_pins

PATTERN = [Pattern("root", LoopOp)]

# Regimes the move set builds: MAP / SEMIRING / MONOID. Flash (R6 — the streaming
# online-softmax nest, built by enumeration/070_coop_reduce) is the MONOID algebra on the
# streaming schedule, derived structurally on demand (``dag.reduction.nested``), not a distinct kind.
_BUILDABLE = (AlgebraKind.MAP, AlgebraKind.SEMIRING, AlgebraKind.MONOID)


def _is_union_pinned(match) -> bool:
    """True when the graph is a **multi-op kernel set** — a split-produced
    ``xn`` producer + gemm consumer (``split/010_split_demoted``), or a whole-model
    graph. A global ``EMMY_<KNOB>`` pin is then a UNION pin: each op takes its
    tier's subset (e.g. ``MMA`` lands on the SEMIRING consumer, the MONOID softmax
    producer ignores it), so a per-op tier-foreign pin is NOT a user contradiction.
    A single-kernel graph keeps the strict ``validate_pins`` check — matching the
    tune search's own ``validate_pins=False`` exemption for union-pinned graphs."""
    graph = getattr(match, "graph", None)
    if graph is None:
        return False
    compute = sum(1 for n in graph.nodes.values() if not isinstance(n.op, (InputOp, ConstantOp)))
    return compute > 1


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:
    """Seed a ``TileGraphOp`` from the fused ``LoopOp`` — the logical algorithm + the
    dag + regime the tile passes fork on. The carry-forward ``LoopOp`` knobs ride
    ``op.knobs`` (the engine merges them forward)."""
    loop_op: LoopOp = root.op
    dag = iter_dag(loop_op)
    regime = classify(dag)
    if regime is None or regime.algebra not in _BUILDABLE:
        raise RuleSkipped(f"move composer cannot lower kernel {loop_op.name!r}")
    # Strict per-op knob-pin validation: a force-pinned env knob foreign to the tier
    # this op resolves to is a hard error, not a silent drop (``_validate``). Greedy
    # compile/run only — the tune search (``ctx.validate_pins=False``) explores
    # tier-foreign forks and union-pinned multi-op graphs (see ``Run.drive``).
    if ctx.validate_pins and not _is_union_pinned(match):
        validate_pins(regime.algebra)
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
