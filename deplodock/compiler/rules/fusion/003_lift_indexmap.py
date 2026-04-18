"""Lift ``IndexMapOp`` to a single-op ``LoopOp`` copy kernel.

Single-source IndexMapOps become a one-Port kernel whose ``Port.index`` is
the IndexMapOp's ``coord_map`` with placeholders substituted by axis Vars.
Multi-source IndexMapOps (cat / concat) become a Select-based kernel: one
Port per source, each guarded by the source's ``select`` predicate, the
chosen value written to the output. Downstream merging folds the copy
kernel into its consumer whenever their axes align via σ.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Literal, Var, substitute
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Axis, Loop, LoopOp, Port, Select, SelectBranch, Stmt, Write
from deplodock.compiler.ir.tensor import IndexMapOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    op = node.op
    if not isinstance(op, IndexMapOp):
        return None

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
        ports.append(Port(index=_substituted_index(src.coord_map, mapping, src_shape)))
        input_names.append(src_id)
        body.append(Write(output=0, index=write_index, value="$0"))
    else:
        branches: list[SelectBranch] = []
        for i, src in enumerate(op.sources):
            src_id = node.inputs[src.input_idx]
            src_shape = graph.nodes[src_id].output.shape if src_id in graph.nodes else ()
            ports.append(Port(index=_substituted_index(src.coord_map, mapping, src_shape)))
            input_names.append(src_id)
            select_expr = substitute(src.select, mapping) if src.select is not None else Literal(1, "int")
            branches.append(SelectBranch(value=f"${i}", select=select_expr))
        body.append(Select(name="v", branches=tuple(branches)))
        body.append(Write(output=0, index=write_index, value="v"))

    # Wrap in nested free Loops.
    nested: tuple[Stmt, ...] = tuple(body)
    for a in reversed(axes):
        nested = (Loop(axis=a, body=nested),)
    kernel = LoopOp(axes=axes, inputs=tuple(ports), body=nested)

    frag = Graph()
    for inp_id in input_names:
        if inp_id in frag.nodes:
            continue
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(kernel, input_names, Tensor(f"lift_{nid}", tuple(op.out_shape), node.output.dtype))
    frag.outputs = [out_id]
    return frag


def _substituted_index(coord_map: tuple, mapping: dict, src_shape: tuple) -> tuple:
    """Substitute placeholders in ``coord_map``; force ``Literal(0)`` for size-1 source dims."""
    out = []
    for i, c in enumerate(coord_map):
        if i < len(src_shape) and isinstance(src_shape[i], int) and src_shape[i] == 1:
            out.append(Literal(0, "int"))
        else:
            out.append(substitute(c, mapping))
    return tuple(out)
