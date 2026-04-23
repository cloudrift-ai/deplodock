"""Compiler pipeline entry point.

``run_pipeline(graph, passes, dump=None)`` runs the named passes from
``deplodock/compiler/passes/`` in order. After each pass, it dispatches
to ``dump.on_pass`` so post-pass artifacts land in the right place
without the caller hard-coding which dump methods belong to which pass.

Standard pass sequences:

- ``["decomposition", "optimization", "fusion"]`` — the base compile: produces
  ``Graph[LoopOp + InputOp + ConstantOp]``. Used by the loop backend and
  the ``deplodock compile`` / ``deplodock trace`` CLIs.
- ``+ ["lowering/kernel", "lowering/cuda"]`` — CUDA backend's full chain
  down to ``Graph[CudaOp]``.

Each ``LoopOp`` gets normalized (structural passes + Expr simplification)
inside its own ``__post_init__``, so the pipeline doesn't need a separate
simplify stage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.rewriter import run_pass

if TYPE_CHECKING:
    from deplodock.compiler.dump import CompilerDump

_PASSES_DIR = Path(__file__).parent / "passes"

logger = logging.getLogger(__name__)


def run_pipeline(graph: Graph, passes: list[str], dump: CompilerDump | None = None) -> Graph:
    """Run each named pass in order, dispatching ``dump.on_pass`` after each."""
    t_start = time.monotonic()
    for name in passes:
        t0 = time.monotonic()
        n_before = len(graph.nodes)
        graph = run_pass(graph, _PASSES_DIR / name)
        logger.info("compile: %-18s %.2fs (%d -> %d nodes)", name, time.monotonic() - t0, n_before, len(graph.nodes))
        if dump is not None:
            dump.on_pass(name, graph)
    logger.info("compile: total %.2fs", time.monotonic() - t_start)
    return graph
