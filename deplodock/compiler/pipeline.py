"""Compiler pipeline entry points.

Lowers a traced ``Graph`` to a ``LoopProgram`` (the post-fusion program
form) via three rewriter stages:

    1. **Decomposition** — rewrites high-level ops to primitives. Each rule
       emits already-broadcast-explicit IR (every ElementwiseOp input has
       the op's output shape; broadcasts live in ``IndexMapOp`` wrappers).
    2. **Optimization** — compose adjacent ``IndexMapOp`` chains so that
       pure layout ops fold into a single coord_map before they lift to
       trivial ``LoopOp`` copies.
    3. **Fusion** — assembles primitives into ``LoopOp`` nodes.

The resulting ``LoopProgram`` is the single input to backend codegen
(``backend/cuda/emit.compile_kernels``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.simplify import simplify_loop_op
from deplodock.compiler.program.loop import LoopProgram
from deplodock.compiler.rewriter import Rewriter

if TYPE_CHECKING:
    from deplodock.compiler.dump import CompilerDump

_RULES_DIR = Path(__file__).parent / "rules"

logger = logging.getLogger(__name__)


def compile_graph(graph: Graph, name: str = "prog", dump: CompilerDump | None = None) -> LoopProgram:
    """Lower a traced ``Graph`` to a ``LoopProgram``.

    The returned program is authoritative for buffer shapes and launch
    order; downstream codegen reads shapes from it and never recomputes
    them.
    """
    t_start = time.monotonic()
    n_in = len(graph.nodes)

    t0 = time.monotonic()
    rewriter_pre = Rewriter.from_directory(_RULES_DIR, pass_order=["decomposition", "optimization"])
    graph = rewriter_pre.apply(graph)
    logger.info("compile: decompose+optimize %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_in, len(graph.nodes))

    if dump is not None:
        dump.dump_tensor_ir(graph)

    t0 = time.monotonic()
    n_before_fusion = len(graph.nodes)
    rewriter_fusion = Rewriter.from_directory(_RULES_DIR, pass_order=["fusion"])
    graph = rewriter_fusion.apply(graph)
    logger.info("compile: fuse %.2fs (%d -> %d nodes)", time.monotonic() - t0, n_before_fusion, len(graph.nodes))

    t0 = time.monotonic()
    n_loop_ops = 0
    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op = simplify_loop_op(node.op)
            n_loop_ops += 1
    logger.info("compile: simplify_loop_op %.2fs (%d LoopOp nodes)", time.monotonic() - t0, n_loop_ops)

    t0 = time.monotonic()
    program = LoopProgram.from_graph(graph, name=name)
    logger.info("compile: LoopProgram.from_graph %.2fs (%d launches)", time.monotonic() - t0, len(program.launches))

    if dump is not None:
        dump.dump_loop_program(program)

    logger.info("compile: total %.2fs", time.monotonic() - t_start)
    return program
