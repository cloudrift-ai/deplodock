"""Rename SSA names in a fused LoopOp body to sequential ``v0``, ``v1``, ...

After merge + copy-elimination the surviving SSA names look like
``v_p_cp0_cp1`` / ``v_cp0_1_cp1_cp0`` — the merge accumulates ``_cp{k}``
suffixes and collision-avoidance ``_N`` increments at every hop. Kernel
semantics are unaffected but the LoopIR pretty-print is unreadable.

This pass walks each LoopOp's body **preserving its tree shape**, assigns
``v0``, ``v1``, ... to each Assign / Select / Load output in definition
order, and rewrites every downstream reference. Accum names (``acc``,
``acc_1``) are already short and semantically meaningful — left untouched.
Idempotent: a body already in ``v0..vN`` form returns None and the
rewriter moves on.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop_ir import (
    Accum,
    Assign,
    Load,
    Loop,
    LoopOp,
    Select,
    SelectBranch,
    Stmt,
    Write,
    flatten_body,
)
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", LoopOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    loop = node.op
    if not isinstance(loop, LoopOp):
        return None

    # Build rename map by pre-order walk of the body (flatten preserves
    # definition order — that's all we need from it; we don't reshape).
    acc_names = {decl.name for decl in loop.accums}
    rename: dict[str, str] = {}
    counter = 0
    for stmt in flatten_body(loop.body):
        if isinstance(stmt, (Assign, Select)):
            if stmt.name in acc_names or stmt.name in rename:
                continue
            rename[stmt.name] = f"v{counter}"
            counter += 1

    if all(old == new for old, new in rename.items()):
        return None  # already canonical

    def rn(name: str) -> str:
        return rename.get(name, name)

    def walk(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for stmt in stmts:
            if isinstance(stmt, Loop):
                out.append(Loop(axis=stmt.axis, body=walk(stmt.body)))
            elif isinstance(stmt, Assign):
                out.append(Assign(name=rn(stmt.name), op=stmt.op, args=tuple(rn(a) for a in stmt.args)))
            elif isinstance(stmt, Load):
                out.append(Load(name=stmt.name, source=stmt.source, index=stmt.index))
            elif isinstance(stmt, Accum):
                out.append(Accum(name=stmt.name, value=rn(stmt.value), op=stmt.op))
            elif isinstance(stmt, Write):
                out.append(Write(output=stmt.output, index=stmt.index, value=rn(stmt.value)))
            elif isinstance(stmt, Select):
                out.append(
                    Select(
                        name=rn(stmt.name),
                        branches=tuple(SelectBranch(value=rn(br.value), select=br.select) for br in stmt.branches),
                    )
                )
            else:
                out.append(stmt)
        return tuple(out)

    new_loop = LoopOp(body=walk(loop.body))

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
