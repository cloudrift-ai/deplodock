"""Collapse ``y = copy(x)`` Assign chains inside a fused LoopOp body.

The merge rule introduces identity Assigns as bridges between each producer's
write value and the matching consumer port reference. After a long merge chain
those bridges stack: ``y = copy(copy(copy(x)))`` with three intermediate SSA
names. The generated CUDA is unaffected (the codegen collapses copies), but
the LoopIR pretty-print becomes unreadable.

This pass walks each LoopOp's body **preserving tree shape**, treats every
``Assign(name=Y, op=ElementwiseOp("copy"), args=(X,))`` as ``Y aliases X``,
rewrites every downstream reference through the alias chain to its root, and
drops the copy Assigns themselves. Purely cosmetic — copy is semantically
the identity function and has no runtime side-effects.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Load,
    Loop,
    LoopOp,
    Select,
    SelectBranch,
    Stmt,
    Write,
)
from deplodock.compiler.ir.tensor_ir import ElementwiseOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", LoopOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    loop = node.op
    if not isinstance(loop, LoopOp):
        return None

    alias: dict[str, str] = {}
    dropped = [0]

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def walk(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for stmt in stmts:
            if isinstance(stmt, Loop):
                out.append(Loop(axis=stmt.axis, body=walk(stmt.body)))
                continue
            if isinstance(stmt, Assign) and isinstance(stmt.op, ElementwiseOp) and stmt.op.fn == "copy" and len(stmt.args) == 1:
                alias[stmt.name] = stmt.args[0]
                dropped[0] += 1
                continue
            if isinstance(stmt, Assign):
                out.append(Assign(name=stmt.name, op=stmt.op, args=tuple(resolve(a) for a in stmt.args)))
            elif isinstance(stmt, Load):
                out.append(stmt)
            elif isinstance(stmt, Accum):
                out.append(Accum(name=stmt.name, value=resolve(stmt.value), op=stmt.op))
            elif isinstance(stmt, Write):
                out.append(Write(output=stmt.output, index=stmt.index, value=resolve(stmt.value)))
            elif isinstance(stmt, Select):
                out.append(
                    Select(
                        name=stmt.name,
                        branches=tuple(SelectBranch(value=resolve(br.value), select=br.select) for br in stmt.branches),
                    )
                )
            else:
                out.append(stmt)
        return tuple(out)

    new_body = walk(loop.body)
    if dropped[0] == 0:
        return None

    new_loop = LoopOp(body=new_body)

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
