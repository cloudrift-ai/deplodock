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
    # Capture inputs before rewriting (InputOp nodes keep their ids).
    graph_inputs = list(graph.inputs)

    rewriter = Rewriter.from_directory(_RULES_DIR)
    graph = rewriter.apply(graph)

    graph_constants = [nid for nid, n in graph.nodes.items() if isinstance(n.op, ConstantOp)]

    # Extract buffer shapes from the post-rewrite graph.
    buf_shapes = {nid: tuple(n.output.shape) for nid, n in graph.nodes.items()}

    kernels = extract_kernels(graph)

    # Graph outputs: the output Port buffer_ids of the KernelOp nodes
    # that correspond to graph.outputs entries.
    graph_outputs = []
    for nid in graph.outputs:
        node = graph.nodes.get(nid)
        if node is not None and isinstance(node.op, KernelOp):
            for out in node.op.outputs:
                if hasattr(out, "buffer_id"):
                    graph_outputs.append(out.buffer_id)

    return CompileResult(
        kernels=kernels,
        buf_shapes=buf_shapes,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        graph_constants=graph_constants,
    )


def _all_leaf_ports(kernel: KernelOp) -> list:
    """Collect all leaf Port objects from a kernel's input trees."""
    from deplodock.compiler.ops import Combine, Mux, Port

    ports: list = []

    def walk(inp):
        if isinstance(inp, Port):
            ports.append(inp)
        elif isinstance(inp, Mux):
            for b in inp.branches:
                walk(b.input)
        elif isinstance(inp, Combine):
            for s in inp.sources:
                walk(s)

    for inp in kernel.inputs:
        walk(inp)
    return ports
