"""Lower each ``LoopOp`` node to a ``TileOp``.

Mechanical translation via ``ir.tile.lower.lower_naive``: outer free-Loop
chain becomes ``Enclosure(thread_axes=...)``, leaves pass through. The
node's id, inputs, and output tensor are preserved — only the op changes.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import Accum, LoopOp
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, LoopOp):
        return None
    kname = _kernel_name_for(node.op, match.root_node_id)
    node.op = lower_naive(node.op, kname)
    return None


def _kernel_name_for(loop: LoopOp, node_id: str) -> str:
    if any(isinstance(s, Accum) for s in loop):
        return f"k_{node_id}_reduce"
    return f"k_{node_id}_pointwise"
