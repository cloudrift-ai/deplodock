"""Coordination for planner-emitted ``TileOp``s.

Runs after ``000_partition_planner`` (which now constructs ``TileOp``
directly with typed tile flavors) and before ``002_stage_inputs``. Walks
the body for two coordination triggers:

- ``GridTile.splitk_axes`` non-empty: rewrites the epilogue ``Write`` to
  atomic-add so cross-CTA split-K reductions accumulate cleanly.
- ``ThreadTile.cooperative_axes`` non-empty: emits ``Combine`` siblings
  after each reduce subtree and wraps scalar ``Write``s in
  ``Cond(coop_axis == 0)`` so only one thread of each cooperative group
  writes the final value.

When neither trigger fires, raises ``RuleSkipped`` so the engine moves
on. Pattern is ``Pattern("root", TileOp)``; the LoopOp-fallback path in
``001_launch_geometry`` handles the rare kernel shapes the planner
skips.

This file is named ``001b_`` so the pipeline loader (which orders rules
alphabetically) slots it between ``001_launch_geometry`` and
``002_stage_inputs``. The actual coordination logic lives in
:func:`deplodock.compiler.pipeline.passes.lowering.tile._launch_geometry.rewrite._coordinate`
via a tiny re-import so we don't duplicate the implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern

# Load 001_launch_geometry as a sibling module (the file name starts with a
# digit so it isn't a valid Python identifier; load it explicitly).
_spec = importlib.util.spec_from_file_location("_001_launch_geometry", Path(__file__).parent / "001_launch_geometry.py")
_lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lg)


PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    """Dispatch to ``001_launch_geometry._coordinate`` (shared helper)."""
    return _lg._coordinate(root.op)
