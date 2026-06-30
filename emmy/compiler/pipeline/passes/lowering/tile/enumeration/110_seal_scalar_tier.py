"""Seal-scalar-tier pass (deterministic) — stamp the scalar-tier OFF sentinels.

This pass is **knob-only** — it does not touch
the stored algorithm (the body moves run in ``060_reduce_tile`` / ``100_register_tile``).
It seals the scalar ``SEMIRING`` reduce's warp-tier atom knob to its OFF sentinel
(``MMA=0``) so every reduce variant carries the warp-tier marker the perf DB / learned
prior key on. The warp counts are now the par factor of ``SPLIT@<axis>`` (no separate
``WM``/``WN`` sentinel), and the cooperative ``REDUCE@<axis>.coop`` rides the reduce value.

It runs at the **post-register** level (gate: every free-axis tile pinned), exactly
where the old monolithic ``040_lower`` stamped these sentinels — so the per-level
greedy ranking sees the same partial knob set at every fork, keeping the deployed pick
byte-identical. It is deterministic (one rewrite, no ``Fork``); a pointwise (``MAP``)
nest carries no warp/reduce knobs and is left untouched.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.tile.ir import TileGraphOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    op: TileGraphOp = root.op
    cell = fam.atom_key(fam.MATMUL_CELL)
    if op.algebra is not AlgebraKind.SEMIRING or op.dag is None or cell in op.knobs:
        raise RuleSkipped("not a scalar SEMIRING reduce / already sealed / warp tier / no dag")
    nkey = fam.split_key(op.dag.inner_n.axis.name)
    if nkey not in op.knobs or not fam.split_complete(op.knobs[nkey]):
        raise RuleSkipped("scalar SEMIRING reduce not fully tiled yet")
    return replace(op, knobs={**op.knobs, cell: fam.SCALAR})
