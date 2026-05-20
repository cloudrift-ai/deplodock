"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
every relevant split (output partition, K chunking, register tile,
serial-outer / pipeline / cooperative-stride marking) and communicates
those decisions to downstream materialization passes via ``Role`` tags
on body ``Loop`` / ``StridedLoop`` stmts (see :class:`Role`).

The downstream passes (``001_tileify``, ``007_stage_inputs``,
``008_register_tile``, ``010_double_buffer``, ``015_pipeline_k_outer``,
etc.) then materialize those tags into actual launch geometry, smem
stages, replicated stmts, pipelined schedules, and so on — without
having to re-derive what to do from the IR shape.

**M1 scope** — this file is currently a stub that always skips. The
mechanism (role field + tileify REGISTER-stop guard) is in place; the
matmul story moves into the planner in M3, when ``002_chunk_matmul_k``
/ ``003_split_matmul_k`` / ``004_launch_geometry`` / ``008_register_tile``
lose their split logic together. ``006_chunk_reduce`` moves in M2.

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
        raise RuleSkipped("DEPLODOCK_PLANNER not set; planner is a no-op in M1")
    raise RuleSkipped("planner emits no role tags yet — M2/M3 will populate")
