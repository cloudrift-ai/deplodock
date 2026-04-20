"""Compiler pipeline entry points.

Lowers a traced ``Graph`` to a ``LoopProgram`` (the post-fusion program
form) via two rewriter stages:

    1. **Decomposition** — rewrites high-level ops to primitives. Each rule
       emits already-broadcast-explicit IR (every ElementwiseOp input has
       the op's output shape; broadcasts live in ``IndexMapOp`` wrappers).
    2. **Fusion** — assembles primitives into ``LoopOp`` nodes.

The resulting ``LoopProgram`` is the single input to backend codegen
(``backend/cuda/emit.compile_kernels``).
"""

from __future__ import annotations

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


def compile_graph(graph: Graph, name: str = "prog", dump: CompilerDump | None = None) -> LoopProgram:
    """Lower a traced ``Graph`` to a ``LoopProgram``.

    The returned program is authoritative for buffer shapes and launch
    order; downstream codegen reads shapes from it and never recomputes
    them.
    """
    rewriter_pre = Rewriter.from_directory(_RULES_DIR, pass_order=["decomposition"])
    graph = rewriter_pre.apply(graph)

    if dump is not None:
        dump.dump_tensor_ir(graph)

    rewriter_fusion = Rewriter.from_directory(_RULES_DIR, pass_order=["fusion"])
    graph = rewriter_fusion.apply(graph)

    for node in graph.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op = simplify_loop_op(node.op)

    program = LoopProgram.from_graph(graph, name=name)
    if dump is not None:
        dump.dump_loop_program(program)
    return program
