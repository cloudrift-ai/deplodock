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
    from deplodock.compiler.ir.cuda import ir as _cuda
    from deplodock.compiler.ir.frontend import ir as _frontend
    from deplodock.compiler.ir.kernel import ir as _kernel
    from deplodock.compiler.ir.loop import ir as _loop
    from deplodock.compiler.ir.tensor import ir as _tensor
    from deplodock.compiler.ir.tile import ir as _tile

    for module in (_base, _tensor, _frontend, _loop, _tile, _kernel, _cuda):
        cls = getattr(module, name, None)
        if isinstance(cls, type) and issubclass(cls, Op):
            return cls


def _serialize_field(v):
    """Flatten non-JSON-friendly op field values to a JSON-compatible form.

    Special cases:

    - ``ir.elementwise.ElementwiseImpl`` → its ``name`` string.
    - ``ir.stmt.Body`` → the underlying ``stmts`` tuple. Body is an
      in-memory wrapper; on disk we keep the tuple-of-stmts form so
      ``_deserialize_field``'s list-of-Stmt-reprs path keeps working
      unchanged. ``LoopOp.__post_init__`` / ``TileOp.__post_init__``
      coerce the tuple back to Body on load.

    Everything else passes through. ``_deserialize_field`` reverses
    this on load.
    """
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.stmt import Body

    if isinstance(v, ElementwiseImpl):
        return v.name
    if isinstance(v, Body):
        # Body is a ``tuple`` subclass; downcast to plain tuple so
        # JSON encodes element-by-element rather than via ``__repr__``
        # (which would produce a single ``"Body((...))"`` string).
        return tuple(v)
    return v


def _deserialize_field(k, v):
    """Reverse of ``_serialize_field``: rehydrate the ``op`` field name
    back to ``ElementwiseImpl``, and eval Stmt-repr strings (produced by
    ``json.dumps(..., default=str)`` against dataclass Stmts) back into
    Stmt instances. The eval scope mirrors the IR's ``__all__`` exports
    — same classes the Stmt reprs reference."""
    from deplodock.compiler.ir.elementwise import ElementwiseImpl

    if k == "op" and isinstance(v, str):
        return ElementwiseImpl(v)
    if k == "body" and isinstance(v, list) and v and all(isinstance(e, str) for e in v):
        return tuple(_eval_stmt(e) for e in v)
    if isinstance(v, list):
        return tuple(v)
    return v


def _eval_stmt(s: str):
    scope = _stmt_eval_scope()
    return eval(s, scope)


_STMT_EVAL_SCOPE: dict | None = None


def _stmt_eval_scope() -> dict:
    """Lazy-built eval scope for Stmt-repr strings."""
    global _STMT_EVAL_SCOPE
    if _STMT_EVAL_SCOPE is not None:
        return _STMT_EVAL_SCOPE
    from deplodock.compiler.ir.axis import (
        BIND_BLOCK,
        BIND_THREAD,
        Axis,
        BoundAxis,
    )
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.expr import (
        BinaryExpr,
        Builtin,
        CastExpr,
        FuncCallExpr,
        Literal,
        TernaryExpr,
        Var,
    )
    from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve
    from deplodock.compiler.ir.stmt import (
        Accum,
        Assign,
        Cond,
        Init,
        Load,
        Loop,
        Select,
        SelectBranch,
        StridedLoop,
        Tile,
        Write,
    )
    from deplodock.compiler.ir.tile.ir import AsyncWait, Combine, Stage

    _STMT_EVAL_SCOPE = {
        "Axis": Axis,
        "BoundAxis": BoundAxis,
        "BIND_BLOCK": BIND_BLOCK,
        "BIND_THREAD": BIND_THREAD,
        "Var": Var,
        "Literal": Literal,
        "BinaryExpr": BinaryExpr,
        "Builtin": Builtin,
        "FuncCallExpr": FuncCallExpr,
        "TernaryExpr": TernaryExpr,
        "CastExpr": CastExpr,
        "Load": Load,
        "Assign": Assign,
        "Accum": Accum,
        "Init": Init,
        "Write": Write,
        "Select": Select,
        "SelectBranch": SelectBranch,
        "Loop": Loop,
        "StridedLoop": StridedLoop,
        "Cond": Cond,
        "Tile": Tile,
        "Stage": Stage,
        "Combine": Combine,
        "AsyncWait": AsyncWait,
        "Smem": Smem,
        "Sync": Sync,
        "TreeHalve": TreeHalve,
        "ElementwiseImpl": ElementwiseImpl,
        "__builtins__": {},
    }
    return _STMT_EVAL_SCOPE


class Graph:
    """Directed acyclic compute graph of tensor operations.

    Stores both directions of each edge: every ``Node`` carries its
    backward edges as ``node.inputs`` (producer ids), and the graph
    maintains a forward-edge index ``_users`` (consumer id set per node).
    Mutation methods keep both sides consistent, so forward walks
    (``users`` / ``consumers``) are O(1) per hop.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.hints: Hints = Hints()
        self._id_counter = itertools.count()
        self._users: dict[str, set[str]] = {}

    def _next_id(self) -> str:
        while True:
            nid = f"n{next(self._id_counter)}"
            if nid not in self.nodes:
                return nid

    def add_node(
        self,
        op: Op,
        inputs: list[Node | str],
        output: Tensor,
        *,
        node_id: str | None = None,
    ) -> str:
        """Add a node to the graph. Returns the node id.

        ``inputs`` accepts a mix of ids and ``Node`` objects (Nodes get
        their ``id`` extracted) — convenient for decomposition rules that
        thread ``Node`` values through pipelines of helpers.

        When ``node_id`` is omitted, defaults to ``output.name`` if that
        name is non-empty and not already taken; otherwise falls back to
        an auto-generated ``n<i>`` id. This keeps semantic names visible
        in kernel buf refs (Load.input / Write.output) without forcing
        every caller to repeat ``node_id=name``.
        """
        if node_id is None:
            if output.name and output.name not in self.nodes:
                nid = output.name
            else:
                nid = self._next_id()
        else:
            nid = node_id
        if nid in self.nodes:
            raise ValueError(f"Node id {nid!r} already exists")
        input_ids = [inp.id if isinstance(inp, Node) else inp for inp in inputs]
        for inp in input_ids:
            if inp not in self.nodes:
                raise ValueError(f"Input node {inp!r} does not exist")
        self.nodes[nid] = Node(id=nid, op=op, inputs=input_ids, output=output)
        self._users[nid] = set()
        for inp in input_ids:
            self._users[inp].add(nid)
        return nid

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} not found")
        node = self.nodes[node_id]
        for inp in node.inputs:
            users = self._users.get(inp)
            if users is not None:
                users.discard(node_id)
        self.inputs = [i for i in self.inputs if i != node_id]
        self.outputs = [o for o in self.outputs if o != node_id]
        del self.nodes[node_id]
        self._users.pop(node_id, None)

    def rename_node(self, old_id: str, new_id: str) -> None:
        """Change a node's id in place, updating every reference.

        Updates ``self.nodes`` keying, ``self.inputs`` / ``self.outputs``
        membership, the ``_users`` index, every consumer's ``node.inputs``
        list, and every consumer LoopOp's internal buf refs
        (``Load.source`` / ``Write.output`` track the same identity as
        graph node ids).
        """
        if old_id == new_id:
            return
        if old_id not in self.nodes:
            raise KeyError(f"Node {old_id!r} not found")
        if new_id in self.nodes:
            raise ValueError(f"Node id {new_id!r} already exists")
        node = self.nodes.pop(old_id)
        node.id = new_id
        self.nodes[new_id] = node
        consumers = self._users.pop(old_id, set())
        self._users[new_id] = consumers
        for consumer_id in consumers:
            consumer = self.nodes[consumer_id]
            consumer.inputs = [new_id if i == old_id else i for i in consumer.inputs]
            consumer.op = _rename_buf_in_op(consumer.op, old_id, new_id)
        for inp in node.inputs:
            users = self._users.get(inp)
            if users is not None and old_id in users:
                users.discard(old_id)
                users.add(new_id)
        self.inputs = [new_id if i == old_id else i for i in self.inputs]
        self.outputs = [new_id if o == old_id else o for o in self.outputs]
        node.op = _rename_buf_in_op(node.op, old_id, new_id)

    def replace_node(self, old_id: str, new_id: str) -> None:
        """Rewire all references from old_id to new_id.

        Updates graph-level edges (consumer ``node.inputs`` and
        ``graph.inputs/outputs``) AND any LoopOp's internal buf references
        (``Load.source`` / ``Write.output`` are buf names tracking the same
        identity as the graph node ids).
        """
        if new_id not in self.nodes:
            raise KeyError(f"Replacement node {new_id!r} not found")
        old_users = self._users.get(old_id, set())
        for consumer_id in list(old_users):
            node = self.nodes[consumer_id]
            node.inputs = [new_id if i == old_id else i for i in node.inputs]
            node.op = _rename_buf_in_op(node.op, old_id, new_id)
            self._users[new_id].add(consumer_id)
        if old_id in self._users:
            self._users[old_id] = set()
        self.inputs = [new_id if i == old_id else i for i in self.inputs]
        self.outputs = [new_id if o == old_id else o for o in self.outputs]

    def users(self, node_id: str) -> set[str]:
        """Return the set of node ids that consume ``node_id``. O(1)."""
        return self._users.get(node_id, set())

    def consumers(self, node_id: str) -> list[str]:
        """List form of :meth:`users`, preserved for existing callers."""
        return list(self._users.get(node_id, ()))

    def topological_order(self) -> list[str]:
        """Return node ids in topological order (inputs before consumers).

        Kahn's algorithm in O(N+E) using the maintained ``_users`` index.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for inp in set(node.inputs):
                if inp in in_degree:
                    in_degree[node.id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: list[str] = []
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            for consumer_id in self._users.get(nid, ()):
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
        g._users = {nid: set(users) for nid, users in self._users.items()}
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

            fields = {k: _deserialize_field(k, v) for k, v in ndata.get("op_fields", {}).items()}
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

        # Rebuild the forward-edge index now that every node is present.
        g._users = {nid: set() for nid in g.nodes}
        for nid, node in g.nodes.items():
            for inp in node.inputs:
                if inp in g._users:
                    g._users[inp].add(nid)

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
                "op_fields": {k: _serialize_field(v) for k, v in node.op.__dict__.items() if not k.startswith("_")},
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
# Op buf-rename helper — keeps LoopOp's internal buf refs aligned with
# graph node ids when ``replace_node`` rewires consumers.
# ---------------------------------------------------------------------------


def _rename_buf_in_op(op, old: str, new: str):
    """Rewrite ``Load.source`` / ``Write.output`` references inside a
    ``LoopOp`` body from ``old`` to ``new`` (recursively into nested Loops).
    Pass-through for op types without internal buf refs."""
    from deplodock.compiler.ir.loop import Load, LoopOp, Write

    if not isinstance(op, LoopOp):
        return op

    def fn(s):
        if isinstance(s, Load) and s.input == old:
            return Load(name=s.name, input=new, index=s.index)
        if isinstance(s, Write) and s.output == old:
            return Write(output=new, index=s.index, value=s.value)
        return s

    return LoopOp(body=op.body.map(fn))


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
        return f"{op.name}({', '.join(arg_names)})"
    if cls == "ReduceOp":
        return f"{op.name}({', '.join(arg_names)}, axis={op.axis})"
    if cls == "ScanOp":
        return f"scan_{op.name}({', '.join(arg_names)}, axis={op.axis})"
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
    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Literal, Var

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

    placeholder_rename = {f"{PLACEHOLDER_PREFIX}{n}": Var(name) for n, name in enumerate(_DIM_NAMES)}

    coord_strs: list[str] = []
    for dim, e in enumerate(src.coord_map):
        # Literal(0) at an extent-1 input dim is a broadcast — render as `na`.
        if isinstance(e, Literal) and e.value == 0 and dim < len(input_shape) and input_shape[dim] == 1:
            coord_strs.append("na")
        else:
            coord_strs.append(e.substitute(placeholder_rename).pretty())
    return f"{input_name}[{', '.join(coord_strs)}]"
