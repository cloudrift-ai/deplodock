"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, etc.) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes (``001_tileify``, ``007_stage_inputs``,
``008_register_tile``, …) read the tags and skip their own equivalent
decisions, doing only the leftover rewrites (lift to ``Tile.axes``,
build stages, replicate stmts).

**M2 scope** — infrastructure only. Adds the planner pass slot, the
``canonicalize_free_axis_order`` role-terminator fix (so planner-tagged
loops survive normalization), and 008's tag-driven path (exercised via
synthetic tests). No matmul emission yet — the matmul register-tile
slice was prototyped here and reverted: pre-tileify REGISTER splits
collide with ``007_stage_inputs``'s cache-axis selection (Stages get
duplicated with name collisions because N_i / M_i aren't cache axes).
Resolving that needs the full launch-geometry + staging story moved
together, so it lives in M4 alongside the matmul K / SPLITK migration.

Subsequent milestones populate the planner: M3 = non-matmul reduce
chunking; M4 = matmul K + launch geometry + SPLITK + register tile;
M5 = cooperative-reduce + pipeline.

Gated by ``DEPLODOCK_PLANNER`` env var so each milestone can test the
new path against the legacy default (=0) for structural equivalence.
"""

from __future__ import annotations

import os

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]

_ENABLE_ENV = "DEPLODOCK_PLANNER"


def rewrite(root: Node) -> Graph | None:
    if not os.environ.get(_ENABLE_ENV):
        raise RuleSkipped(f"{_ENABLE_ENV} not set")
    raise RuleSkipped("planner emits no role tags yet — M3/M4/M5 populate")
