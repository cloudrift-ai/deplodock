"""Tensor IR: Tensor, Node, and Graph."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from deplodock.compiler.ops import Op


@dataclass
class Tensor:
    """Multidimensional array descriptor."""

    name: str
    shape: tuple[int | str, ...]  # concrete ints or symbolic dim names
    dtype: str = "f32"


@dataclass
class Node:
    """A single operation in the compute graph."""

    id: str
    op: Op
    inputs: list[str]  # node ids
    output: Tensor


class Graph:
    """Directed acyclic compute graph of tensor operations."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self._id_counter = itertools.count()

    def _next_id(self) -> str:
        return f"n{next(self._id_counter)}"

    def add_node(
        self,
        op: Op,
        inputs: list[str],
        output: Tensor,
        *,
        node_id: str | None = None,
    ) -> str:
        """Add a node to the graph. Returns the node id."""
        nid = node_id or self._next_id()
        if nid in self.nodes:
            raise ValueError(f"Node id {nid!r} already exists")
        for inp in inputs:
            if inp not in self.nodes:
                raise ValueError(f"Input node {inp!r} does not exist")
        self.nodes[nid] = Node(id=nid, op=op, inputs=inputs, output=output)
        return nid

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} not found")
        self.inputs = [i for i in self.inputs if i != node_id]
        self.outputs = [o for o in self.outputs if o != node_id]
        del self.nodes[node_id]

    def replace_node(self, old_id: str, new_id: str) -> None:
        """Rewire all references from old_id to new_id."""
        if new_id not in self.nodes:
            raise KeyError(f"Replacement node {new_id!r} not found")
        for node in self.nodes.values():
            node.inputs = [new_id if i == old_id else i for i in node.inputs]
        self.inputs = [new_id if i == old_id else i for i in self.inputs]
        self.outputs = [new_id if o == old_id else o for o in self.outputs]

    def consumers(self, node_id: str) -> list[str]:
        """Return ids of nodes that consume node_id as an input."""
        return [n.id for n in self.nodes.values() if node_id in n.inputs]

    def topological_order(self) -> list[str]:
        """Return node ids in topological order (inputs before consumers)."""
        # Count unique input edges (deduplicate for fan-out).
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for inp in set(node.inputs):
                if inp in in_degree:
                    in_degree[node.id] += 1

        # Kahn's algorithm — deterministic ordering via insertion order.
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            for consumer_id in self.consumers(nid):
                in_degree[consumer_id] -= 1
                if in_degree[consumer_id] == 0:
                    queue.append(consumer_id)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle")
        return result

    def copy(self) -> Graph:
        """Return a deep copy of the graph."""
        g = Graph()
        g._id_counter = itertools.count(next(self._id_counter))
        for nid, node in self.nodes.items():
            g.nodes[nid] = Node(
                id=node.id,
                op=node.op,
                inputs=list(node.inputs),
                output=Tensor(
                    name=node.output.name,
                    shape=node.output.shape,
                    dtype=node.output.dtype,
                ),
            )
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        return g

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-compatible dict."""
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": {
                nid: {
                    "op": type(node.op).__name__,
                    "op_fields": {k: v for k, v in node.op.__dict__.items() if not k.startswith("_")},
                    "inputs": node.inputs,
                    "output": {
                        "name": node.output.name,
                        "shape": list(node.output.shape),
                        "dtype": node.output.dtype,
                    },
                }
                for nid, node in self.nodes.items()
            },
        }
