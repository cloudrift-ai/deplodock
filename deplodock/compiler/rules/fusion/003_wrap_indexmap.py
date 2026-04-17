"""Wrap standalone IndexMapOps as copy kernels.

An IndexMapOp whose consumers are all already-fused LoopOps gets wrapped
as a single-launch copy kernel.

Single-source IndexMapOps become a single-Port kernel that reads via the
absorbed ``coord_map`` and writes the result. Multi-source IndexMapOps
(cat / concat) become a Select-based copy kernel: one Port per source,
each guarded by the source's ``select`` predicate, the chosen value
written to the output.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Literal, Var, substitute
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Axis, LoopOp, Port, Select, SelectBranch, Write
from deplodock.compiler.ir.tensor import IndexMapOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("wrap", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, IndexMapOp):
        return None
    consumers = graph.consumers(nid)
    if not all(isinstance(graph.nodes[c].op, LoopOp) for c in consumers if c in graph.nodes):
        return None

    op = node.op

    # Free-axis names matching the output shape positions.
    axes = tuple(Axis(name=f"a{i}", extent=int(d), kind="free") for i, d in enumerate(op.out_shape))
    mapping = {f"{PLACEHOLDER_PREFIX}{i}": Var(a.name) for i, a in enumerate(axes)}
    write_index = tuple(Var(a.name) for a in axes)

    ports: list[Port] = []
    input_names: list[str] = []
    body: list = []

    if len(op.sources) == 1:
        src = op.sources[0]
        src_id = node.inputs[src.input_idx]
        src_shape = graph.nodes[src_id].output.shape if src_id in graph.nodes else ()
        port_index = _substituted_index(src.coord_map, mapping, src_shape)
        ports.append(Port(index=port_index))
        input_names.append(src_id)
        body.append(Write(output=0, index=write_index, value="$0"))
    else:
        branches = []
        for i, src in enumerate(op.sources):
            src_id = node.inputs[src.input_idx]
            src_shape = graph.nodes[src_id].output.shape if src_id in graph.nodes else ()
            port_index = _substituted_index(src.coord_map, mapping, src_shape)
            ports.append(Port(index=port_index))
            input_names.append(src_id)
            # Catch-all branch (select=None) gets a trivially-true Literal so
            # the interpreter's Select fold doesn't need a None special case.
            select = substitute(src.select, mapping) if src.select is not None else Literal(1, "int")
            branches.append(SelectBranch(value=f"${i}", select=select))
        body.append(Select(name="v", branches=tuple(branches)))
        body.append(Write(output=0, index=write_index, value="v"))

    kernel = LoopOp(axes=axes, inputs=tuple(ports), body=tuple(body))

    frag = Graph()
    for inp_id in input_names:
        if inp_id not in frag.nodes:
            ext = graph.nodes.get(inp_id)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(kernel, input_names, Tensor(f"kernel_{nid}", tuple(op.out_shape), node.output.dtype))
    frag.outputs = [out_id]
    return frag


def _substituted_index(coord_map: tuple, mapping: dict, src_shape: tuple) -> tuple:
    """Build a Port.index by substituting placeholders; force Literal(0) for size-1 dims."""
    from deplodock.compiler.ir.expr import Literal

    out = []
    for i, c in enumerate(coord_map):
        if i < len(src_shape) and isinstance(src_shape[i], int) and src_shape[i] == 1:
            out.append(Literal(0, "int"))
        else:
            out.append(substitute(c, mapping))
    return tuple(out)
