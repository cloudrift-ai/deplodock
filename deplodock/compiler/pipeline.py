"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of structural ``KernelOp``s:

    1. **Decomposition** — rewrites high-level ops to primitives.
    2. **Optimization** — canonicalizes primitive graph (e.g. merge IndexMaps).
    3. **Fusion** — assembles primitives into ``KernelOp`` nodes using the
       chain grammar.
    4. **Extraction** — collects ``KernelOp``s in topo order for backend codegen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.lower import extract_kernels
from deplodock.compiler.ops import ConstantOp, KernelOp
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent / "rules"


@dataclass
class CompileResult:
    """Output of ``compile_graph``: kernels plus graph-level metadata.

    ``buf_shapes`` maps every buffer_id to its shape (from the post-rewrite
    graph). ``graph_inputs``, ``graph_outputs``, and ``graph_constants``
    record the buffer roles from the original graph so the backend can
    mark buffer roles (input / output / constant / scratch) without the
    caller having to thread them manually.
    """

    kernels: list[KernelOp]
    buf_shapes: dict[str, tuple] = field(default_factory=dict)
    graph_inputs: list[str] = field(default_factory=list)
    graph_outputs: list[str] = field(default_factory=list)
    graph_constants: list[str] = field(default_factory=list)


def compile_graph(graph: Graph) -> CompileResult:
    """Lower a traced ``Graph`` to a ``CompileResult``.

    Runs decomposition → optimization → fusion (via the rewriter), then
    extracts the resulting ``KernelOp`` nodes together with graph-level
    metadata (inputs, outputs, constants).
    """
    # Capture metadata from the pre-rewrite graph.
    graph_inputs = list(graph.inputs)
    graph_outputs = list(graph.outputs)

    rewriter = Rewriter.from_directory(_RULES_DIR)
    graph = rewriter.apply(graph)

    # Constants survive rewriting as ConstantOp nodes.
    graph_constants = [nid for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp)]

    # Extract buffer shapes from the post-rewrite graph.
    buf_shapes = {nid: tuple(n.output.shape) for nid, n in graph.nodes.items()}

    kernels = extract_kernels(graph)

    return CompileResult(
        kernels=kernels,
        buf_shapes=buf_shapes,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        graph_constants=graph_constants,
    )
