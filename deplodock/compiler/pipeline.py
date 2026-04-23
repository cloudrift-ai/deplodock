"""Compiler pipeline entry points.

Lowers a traced ``Graph`` through the rewriter's passes:

    1. **Decomposition** — rewrites high-level ops to primitives. Each rule
       emits already-broadcast-explicit IR (every ElementwiseOp input has
       the op's output shape; broadcasts live in ``IndexMapOp`` wrappers).
    2. **Optimization** — compose adjacent ``IndexMapOp`` chains so that
       pure layout ops fold into a single coord_map before they lift to
       trivial ``LoopOp`` copies.
    3. **Fusion** — assembles primitives into ``LoopOp`` nodes.

Each ``LoopOp`` gets normalized (structural passes + Expr simplification)
inside its own ``__post_init__``, so the pipeline doesn't need a
separate simplify stage.

The result is a ``Graph[LoopOp + InputOp + ConstantOp]``. Backend-specific
lowering (``Graph[LoopOp] → Graph[KernelOp] → Graph[CudaOp]``) happens in
each backend's ``compile``.
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


def compile_graph(graph: Graph, dump: CompilerDump | None = None) -> Graph:
    """Run decomposition → optimization → fusion on a traced graph.

    Returns a fused graph whose compute nodes are all ``LoopOp`` instances.
    """
    t_start = time.monotonic()
    n_in = len(graph.nodes)

    t0 = time.monotonic()
    graph = run_pass(graph, _PASSES_DIR / "decomposition")
    graph = run_pass(graph, _PASSES_DIR / "optimization")
    logger.info("compile: decompose+optimize %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_in, len(graph.nodes))

    if dump is not None:
        dump.on_pass("optimization", graph)

    t0 = time.monotonic()
    n_before_fusion = len(graph.nodes)
    graph = run_pass(graph, _PASSES_DIR / "fusion")
    logger.info("compile: fuse %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_before_fusion, len(graph.nodes))

    if dump is not None:
        dump.on_pass("fusion", graph)

    logger.info("compile: total %.2fs", time.monotonic() - t_start)
    return graph
