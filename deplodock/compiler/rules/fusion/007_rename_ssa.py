"""Rename SSA names in a fused LoopOp body to sequential ``v0``, ``v1``, ...

After merge + copy-elimination the surviving SSA names look like
``v_p_cp0_cp1`` / ``v_cp0_1_cp1_cp0`` — the merge accumulates ``_cp{k}``
suffixes and collision-avoidance ``_N`` increments at every hop. Kernel
semantics are unaffected but the LoopIR pretty-print is unreadable.

This pass walks each LoopOp's flat body in definition order, assigns
``v0``, ``v1``, ... to each Assign / Select output, and rewrites every
downstream reference. Accumulator names (``acc``, ``acc_1``) are already
short and semantically meaningful — left untouched. Idempotent: a body
already in ``v0..vN`` form returns None and the rewriter moves on.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop_ir import (
    Assign,
    LoopOp,
    Select,
    SelectBranch,
    Stmt,
    Update,
    Write,
    flat_body_to_nested,
    flatten_body,
)
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", LoopOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    loop = node.op
    if not isinstance(loop, LoopOp):
        return None

    flat = list(flatten_body(loop.body))
    acc_names = {acc.name for acc in loop.accumulators}

    rename: dict[str, str] = {}
    counter = 0
    for stmt in flat:
        if isinstance(stmt, (Assign, Select)):
            if stmt.name in acc_names or stmt.name in rename:
                continue
            rename[stmt.name] = f"v{counter}"
            counter += 1

    if all(old == new for old, new in rename.items()):
        return None  # already canonical

    def rn(arg: str) -> str:
        return rename.get(arg, arg)

    new_body: list[Stmt] = []
    for stmt in flat:
        if isinstance(stmt, Assign):
            new_body.append(Assign(name=rn(stmt.name), op=stmt.op, args=tuple(rn(a) for a in stmt.args)))
        elif isinstance(stmt, Update):
            new_body.append(Update(target=stmt.target, value=rn(stmt.value)))
        elif isinstance(stmt, Write):
            new_body.append(Write(output=stmt.output, index=stmt.index, value=rn(stmt.value)))
        elif isinstance(stmt, Select):
            new_body.append(
                Select(
                    name=rn(stmt.name),
                    branches=tuple(SelectBranch(value=rn(br.value), select=br.select) for br in stmt.branches),
                )
            )
        else:
            new_body.append(stmt)

    new_loop = LoopOp(
        inputs=loop.inputs,
        accumulators=loop.accumulators,
        body=flat_body_to_nested(loop.axes, tuple(new_body)),
    )

    frag = Graph()
    for inp_id in node.inputs:
        if inp_id in frag.nodes:
            continue
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(
        new_loop,
        list(node.inputs),
        Tensor(node.output.name, node.output.shape, node.output.dtype),
    )
    frag.outputs = [out_id]
    return frag
