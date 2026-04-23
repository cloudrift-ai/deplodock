"""Compiler pipeline entry points.

Lowers a traced ``Graph`` through the rewriter's passes:

    1. **Decomposition** — rewrites high-level ops to primitives. Each rule
       emits already-broadcast-explicit IR (every ElementwiseOp input has
       the op's output shape; broadcasts live in ``IndexMapOp`` wrappers).
    2. **Optimization** — compose adjacent ``IndexMapOp`` chains so that
       pure layout ops fold into a single coord_map before they lift to
       trivial ``LoopOp`` copies.
    3. **Fusion** — assembles primitives into ``LoopOp`` nodes.

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
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.simplify import simplify_loop_op
from deplodock.compiler.rewriter import Rewriter

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
    rewriter_pre = Rewriter.from_directory(_PASSES_DIR, pass_order=["decomposition", "optimization"])
    graph = rewriter_pre.apply(graph)
    logger.info("compile: decompose+optimize %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_in, len(graph.nodes))

    if dump is not None:
        dump.dump_tensor_ir(graph)

    t0 = time.monotonic()
    n_before_fusion = len(graph.nodes)
    rewriter_fusion = Rewriter.from_directory(_PASSES_DIR, pass_order=["fusion"])
    graph = rewriter_fusion.apply(graph)
    logger.info("compile: fuse %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_before_fusion, len(graph.nodes))

    t0 = time.monotonic()
    n_loop_ops = 0
    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op = simplify_loop_op(node.op)
            n_loop_ops += 1
    logger.info("compile: simplify_loop_op %.2fs (%d LoopOp nodes)", time.monotonic() - t0, n_loop_ops)

    if dump is not None:
        dump.dump_fused_graph(graph)
        dump.dump_loop_ir(graph)

    logger.info("compile: total %.2fs", time.monotonic() - t_start)
    return graph
