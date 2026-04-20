"""Graph container + ``Hints`` metadata.

The graph is a directed acyclic dataflow graph (Kahn-style) over tensor
values. Each ``Node`` wraps one primitive ``Op`` subclass, so nodes are
parameterized by their op type — a ``Node[ElementwiseOp]`` is statically
distinguishable from a ``Node[ReduceOp]``. The structural compiler IR
(``LoopOp`` trees in ``ir/block.py``) relies on that distinction to carry
typed chains like ``tuple[Node[ElementwiseOp], ...]``.

The same ``Graph`` hosts nodes from every IR level as the rewriter runs:
frontend ops during tracing → tensor/minimal ops after decomposition →
``LoopOp`` nodes after fusion. Shapes, inputs, outputs, and hints live
on nodes; ops themselves are shape-free and shared-free by convention.

``Hints`` is the advisory metadata bag attached to individual nodes and
to the graph as a whole. It lives here because ``Node`` and ``Graph`` are
its only users.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from deplodock.compiler.ir.base import Op

# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------


@dataclass
class Hints:
    """Advisory metadata bag keyed by dotted namespace strings.

    Hints do not affect computation semantics — backends may ignore unknown
    hints. Keys use dotted namespaces (e.g. ``cuda.matmul.strategy``).
    """

    _data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a hint value."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Return whether *key* is present."""
        return key in self._data

    def remove(self, key: str) -> None:
        """Remove *key* if present."""
        self._data.pop(key, None)

    def merge(self, other: Hints) -> None:
        """Merge *other* into self.  Other's values win on conflict."""
        self._data.update(other._data)

    def prefix(self, ns: str) -> dict[str, Any]:
        """Return all hints under *ns* as a flat dict with the prefix stripped.

        >>> h = Hints(); h.set("cuda.matmul.strategy", "naive")
        >>> h.prefix("cuda.matmul")
        {'strategy': 'naive'}
        """
        dot = ns if ns.endswith(".") else ns + "."
        return {k[len(dot) :]: v for k, v in self._data.items() if k.startswith(dot)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dict(self._data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Hints:
        """Deserialize from a dict."""
        return Hints(_data=dict(data))

    def __bool__(self) -> bool:
        return bool(self._data)

    def __repr__(self) -> str:
        return f"Hints({self._data!r})"


def resolve_hints(graph: Graph, node_id: str) -> Hints:
    """Merge graph-level hints with node-level hints (node wins)."""
    merged = Hints()
    merged.merge(graph.hints)
    merged.merge(graph.nodes[node_id].hints)
    return merged


# ---------------------------------------------------------------------------
# Tensor + Node + Graph
# ---------------------------------------------------------------------------


@dataclass
class Tensor:
    """Multidimensional array descriptor."""

    name: str
    shape: tuple[int | str, ...]  # concrete ints or symbolic dim names
    dtype: str = "f32"


@dataclass
class Node[T_Op: Op]:
    """A single operation in the compute graph.

    Generic on the op type (PEP 695) so consumers can narrow to
    ``Node[ElementwiseOp]`` or ``Node[ReduceOp]`` where the surrounding
    structure demands it (e.g. a body-chain slot that admits only
    elementwise ops, or a ``ReduceStage`` whose ``reduce`` field must be
    a reduction). The parameter is erased at runtime; owning dataclasses
    enforce the invariant via ``__post_init__``.
    """

    id: str
    op: T_Op
    inputs: list[str]  # node ids
    output: Tensor
    hints: Hints = field(default_factory=Hints)


def _lookup_op_class(name: str) -> type[Op] | None:
    """Find an Op subclass by name across all IR dialect modules.

    Used by ``Graph.from_dict`` to reconstruct nodes. Lazy-imports each
    module to avoid pulling them all in at graph import time.
    """
    from deplodock.compiler.ir import base as _base
    from deplodock.compiler.ir import frontend_ir as _frontend
    from deplodock.compiler.ir import tensor_ir as _tensor
    from deplodock.compiler.ir.loop import ir as _loop

    for module in (_base, _tensor, _frontend, _loop):
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, Op):
            return cls
    return None


class Graph:
    """Directed acyclic compute graph of tensor operations."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.hints: Hints = Hints()
        self._id_counter = itertools.count()

    def _next_id(self) -> str:
        while True:
            nid = f"n{next(self._id_counter)}"
            if nid not in self.nodes:
                return nid

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
        g.hints = Hints.from_dict(self.hints.to_dict())
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
                hints=Hints.from_dict(node.hints.to_dict()),
            )
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        return g

    @staticmethod
    def from_dict(data: dict) -> Graph:
        """Deserialize a graph from a JSON-compatible dict (inverse of to_dict)."""
        g = Graph()
        g.hints = Hints.from_dict(data.get("hints", {}))
        # First pass: create all nodes (inputs first, then in order).
        for nid, ndata in data["nodes"].items():
            op_cls_name = ndata["op"]
            op_cls = _lookup_op_class(op_cls_name)
            if op_cls is None:
                raise ValueError(f"Unknown op class: {op_cls_name}")

            fields = ndata.get("op_fields", {})
            # Convert list fields to tuples for dataclass ops that expect tuples.
            for k, v in fields.items():
                if isinstance(v, list):
                    fields[k] = tuple(v)

            op = op_cls(**fields) if fields else op_cls()
            out = ndata["output"]
            tensor = Tensor(
                name=out["name"],
                shape=tuple(out["shape"]),
                dtype=out.get("dtype", "f32"),
            )
            node_hints = Hints.from_dict(ndata.get("hints", {}))
            # Add directly to bypass input validation (nodes may reference later nodes).
            g.nodes[nid] = Node(id=nid, op=op, inputs=list(ndata["inputs"]), output=tensor, hints=node_hints)

        g.inputs = list(data["inputs"])
        g.outputs = list(data["outputs"])
        return g

    def pretty_print(self) -> str:
        """Render the graph as readable text (topological order + sections)."""
        from deplodock.compiler.ir.base import ConstantOp, InputOp

        order = self.topological_order()
        lines: list[str] = [
            f"# Graph: {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs",
            "",
        ]

        if self.inputs:
            lines.append("inputs:")
            for nid in self.inputs:
                lines.append(f"  {_fmt_tensor(self.nodes[nid].output)}")
            lines.append("")

        const_ids = [nid for nid in order if isinstance(self.nodes[nid].op, ConstantOp)]
        if const_ids:
            lines.append("constants:")
            for nid in const_ids:
                n = self.nodes[nid]
                val = getattr(n.op, "value", None)
                suffix = f" = {val}" if val is not None else ""
                lines.append(f"  {_fmt_tensor(n.output)}{suffix}")
            lines.append("")

        compute_ids = [nid for nid in order if not isinstance(self.nodes[nid].op, (InputOp, ConstantOp))]
        if compute_ids:
            name_w = max(len(self.nodes[nid].output.name) for nid in compute_ids)
            op_w = max(len(_fmt_op(self.nodes[nid], self)) for nid in compute_ids)
            for nid in compute_ids:
                n = self.nodes[nid]
                op_str = _fmt_op(n, self)
                shape_str = _fmt_shape(n.output.shape)
                lines.append(f"{n.output.name:<{name_w}}  =  {op_str:<{op_w}}  -> {shape_str} {n.output.dtype}")
            lines.append("")

        if self.outputs:
            lines.append("outputs:")
            for nid in self.outputs:
                lines.append(f"  {_fmt_tensor(self.nodes[nid].output)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-compatible dict."""
        result: dict = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": {},
        }
        if self.hints:
            result["hints"] = self.hints.to_dict()
        for nid, node in self.nodes.items():
            entry: dict = {
                "op": type(node.op).__name__,
                "op_fields": {k: v for k, v in node.op.__dict__.items() if not k.startswith("_")},
                "inputs": node.inputs,
                "output": {
                    "name": node.output.name,
                    "shape": list(node.output.shape),
                    "dtype": node.output.dtype,
                },
            }
            if node.hints:
                entry["hints"] = node.hints.to_dict()
            result["nodes"][nid] = entry
        return result


# ---------------------------------------------------------------------------
# Pretty-print helpers (used by Graph.pretty_print)
# ---------------------------------------------------------------------------


def _fmt_shape(shape: tuple) -> str:
    inside = ", ".join(str(d) for d in shape)
    if len(shape) == 1:
        inside += ","
    return f"({inside})"


def _fmt_tensor(t: Tensor) -> str:
    return f"{t.name}: {_fmt_shape(t.shape)} {t.dtype}"


_DIM_NAMES = ("i", "j", "k", "l", "m", "n")


def _fmt_op(node: Node, graph: Graph) -> str:
    """Render `node` as a function-call-like string using input tensor names."""
    op = node.op
    cls = type(op).__name__
    arg_names = [graph.nodes[inp].output.name for inp in node.inputs]

    if cls == "ElementwiseOp":
        return f"{op.fn}({', '.join(arg_names)})"
    if cls == "ReduceOp":
        return f"{op.fn}({', '.join(arg_names)}, axis={op.axis})"
    if cls == "ScanOp":
        return f"scan_{op.fn}({', '.join(arg_names)}, axis={op.axis})"
    if cls == "GatherOp":
        return f"gather({', '.join(arg_names)}, axis={op.axis})"
    if cls == "ScatterOp":
        red = f", reduce={op.reduce_fn}" if getattr(op, "reduce_fn", None) else ""
        return f"scatter({', '.join(arg_names)}, axis={op.axis}{red})"
    if cls == "IndexMapOp":
        return _fmt_indexmap(node, graph)

    # Generic fallback (frontend ops like MeanOp, LinearOp, SdpaOp, or LoopOp).
    fields = {k: v for k, v in op.__dict__.items() if not k.startswith("_") and k != "name"}
    label = cls.removesuffix("Op").lower() or cls
    parts = list(arg_names) + [f"{k}={v}" for k, v in fields.items()]
    return f"{label}({', '.join(parts)})"


def _fmt_indexmap(node: Node, graph: Graph) -> str:
    """Render IndexMapOp concisely when its coord expression is simple.

    Single-source, no select, ≤6 output dims → ``src[coord_0, coord_1, ...]``
    with ``out_coord_N`` placeholders replaced by ``i``/``j``/``k``/``l``/``m``/``n``.
    A ``Literal(0)`` at a position where the input's corresponding dim has
    extent 1 is rendered as ``na`` (numpy-style newaxis), making broadcasts
    visually distinct from scalar slices. Everything else (multi-source,
    selected reads, many dims) falls back to ``indexmap(src0, src1, ...)``.
    """
    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Literal, Var, render

    op = node.op
    sources = op.sources
    out_shape = op.out_shape
    arg_names = [graph.nodes[inp].output.name for inp in node.inputs]

    if len(sources) != 1 or sources[0].select is not None or len(out_shape) > len(_DIM_NAMES):
        return f"indexmap({', '.join(arg_names)})"

    src = sources[0]
    input_name = arg_names[src.input_idx] if src.input_idx < len(arg_names) else f"${src.input_idx}"
    input_id = node.inputs[src.input_idx] if src.input_idx < len(node.inputs) else None
    input_shape = graph.nodes[input_id].output.shape if input_id and input_id in graph.nodes else ()

    def rename(e):
        if isinstance(e, Var) and e.name.startswith(PLACEHOLDER_PREFIX):
            try:
                n = int(e.name[len(PLACEHOLDER_PREFIX) :])
                if 0 <= n < len(_DIM_NAMES):
                    return _DIM_NAMES[n]
            except ValueError:
                pass
        return None

    coord_strs: list[str] = []
    for dim, e in enumerate(src.coord_map):
        # Literal(0) at an extent-1 input dim is a broadcast — render as `na`.
        if isinstance(e, Literal) and e.value == 0 and dim < len(input_shape) and input_shape[dim] == 1:
            coord_strs.append("na")
        else:
            coord_strs.append(render(e, rename))
    return f"{input_name}[{', '.join(coord_strs)}]"
