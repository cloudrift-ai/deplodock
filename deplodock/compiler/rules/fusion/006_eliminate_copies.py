"""Collapse ``y = copy(x)`` Assign chains inside a fused LoopOp body.

The merge rule introduces identity Assigns as bridges between each producer's
write value and the matching consumer port reference. After a long merge chain
those bridges stack: ``y = copy(copy(copy(x)))`` with three intermediate SSA
names. The generated CUDA is unaffected (the codegen collapses copies), but
the LoopIR pretty-print becomes unreadable.

This pass walks each LoopOp's flat body, treats every ``Assign(name=Y,
op=ElementwiseOp("copy"), args=(X,))`` as ``Y aliases X``, rewrites every
downstream reference through the alias chain to its root, and drops the copy
Assigns themselves. Purely cosmetic — copy is semantically the identity
function and has no runtime side-effects.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import (
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
from deplodock.compiler.ir.tensor import ElementwiseOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", LoopOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    loop = node.op
    if not isinstance(loop, LoopOp):
        return None

    flat = list(flatten_body(loop.body))
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    new_body: list[Stmt] = []
    dropped = 0
    for stmt in flat:
        if isinstance(stmt, Assign) and isinstance(stmt.op, ElementwiseOp) and stmt.op.fn == "copy" and len(stmt.args) == 1:
            # Record alias and drop the copy — downstream refs will resolve through it.
            # Args include both SSA names and Port references ($N); both flow fine
            # through the alias (an Assign's `args` field accepts either form).
            alias[stmt.name] = stmt.args[0]
            dropped += 1
            continue

        if isinstance(stmt, Assign):
            new_body.append(Assign(name=stmt.name, op=stmt.op, args=tuple(resolve(a) for a in stmt.args)))
        elif isinstance(stmt, Update):
            new_body.append(Update(target=stmt.target, value=resolve(stmt.value)))
        elif isinstance(stmt, Write):
            new_body.append(Write(output=stmt.output, index=stmt.index, value=resolve(stmt.value)))
        elif isinstance(stmt, Select):
            new_body.append(
                Select(
                    name=stmt.name,
                    branches=tuple(SelectBranch(value=resolve(br.value), select=br.select) for br in stmt.branches),
                )
            )
        else:
            new_body.append(stmt)

    if dropped == 0:
        return None

    new_loop = LoopOp(
        inputs=loop.inputs,
        locals=loop.locals,
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
