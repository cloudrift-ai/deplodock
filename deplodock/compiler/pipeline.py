"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of ``KernelInfo`` (KernelOp + buffer names):

    1. **Decomposition** — rewrites high-level ops to primitives.
    2. **Optimization** — canonicalizes primitive graph.
    3. **Fusion** — assembles primitives into ``KernelOp`` nodes.
    4. **Extraction** — collects KernelInfos in topo order for backend codegen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.lower import KernelInfo, extract_kernels
from deplodock.compiler.ops import ConstantOp
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent / "rules"


@dataclass
class CompileResult:
    """Output of ``compile_graph``."""

    kernels: list[KernelInfo]
    buf_shapes: dict[str, tuple] = field(default_factory=dict)
    graph_inputs: list[str] = field(default_factory=list)
    graph_outputs: list[str] = field(default_factory=list)
    graph_constants: list[str] = field(default_factory=list)
    constant_values: dict[str, float] = field(default_factory=dict)


def compile_graph(graph: Graph) -> CompileResult:
    """Lower a traced ``Graph`` to a ``CompileResult``."""
    graph_inputs = list(graph.inputs)

    rewriter_pre = Rewriter.from_directory(_RULES_DIR, pass_order=["decomposition", "optimization"])
    graph = rewriter_pre.apply(graph)
    buf_shapes = {nid: tuple(n.output.shape) for nid, n in graph.nodes.items()}

    rewriter_fusion = Rewriter.from_directory(_RULES_DIR, pass_order=["fusion"])
    graph = rewriter_fusion.apply(graph)

    for nid, n in graph.nodes.items():
        if nid not in buf_shapes:
            buf_shapes[nid] = tuple(n.output.shape)

    graph_constants = [nid for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp)]
    constant_values = {nid: n.op.value for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp) and n.op.value is not None}

    kernels = extract_kernels(graph)

    # Graph outputs: the KernelOp graph node IDs that are graph outputs.
    graph_outputs = [info.output_name for info in kernels if info.output_name in set(graph.outputs)]

    return CompileResult(
        kernels=kernels,
        buf_shapes=buf_shapes,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        graph_constants=graph_constants,
        constant_values=constant_values,
    )
