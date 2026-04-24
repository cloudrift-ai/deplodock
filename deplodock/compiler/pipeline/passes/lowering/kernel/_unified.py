"""Unified emit skeleton — recursive walk over the LoopOp body.

One ``emit_body`` function handles every shape (pointwise, RMSNorm, softmax,
matmul, SDPA's flash-style nested-reduce-in-output-loop) by recursing on the
body tree. At each statement:

- ``Loop`` containing an ``Accum`` → reduce block: declare a register
  accumulator, emit ``for (k = 0; k < extent; ++k) { body }``, the inner
  ``Accum`` folds.
- ``Loop`` without ``Accum`` → free Loop: if its axis is bound to a thread
  coord (in ``live_axes``) the for-loop is elided and the body is inlined;
  otherwise emit a serial ``for (v = 0; v < extent; ++v) { body }``.
- Leaf ``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` → render
  through ``_common.emit_stmt``.

Live axes are the outer free-Loop chain that wraps every reduce. Each thread
binds these axes from the flat tid; everything below runs serially per
thread. K is serial per thread, so no cross-thread combine is needed
(split-K is a follow-up that would specialize this skeleton).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import ForLoop, GpuKernel, IfStmt, Stmt, VarDecl
from deplodock.compiler.ir.loop import Accum, Axis, Loop, LoopOp
from deplodock.compiler.ir.loop import Stmt as LoopStmt
from deplodock.compiler.pipeline.passes.lowering.kernel._common import (
    BLOCK,
    Ctx,
    axis_env_for_flat,
    build_params,
    emit_stmt,
    numel_axes,
)


def emit_unified(node: Node, kernel_name: str, graph: Graph) -> tuple[GpuKernel, list[str], tuple[int, int, int], tuple[int, int, int]]:
    params, arg_order = build_params(node)
    loop: LoopOp = node.op

    top_level, leaf_body, live_axes = _split_outer(loop.body)

    n_threads = numel_axes(live_axes)
    grid = (max((n_threads + BLOCK - 1) // BLOCK, 1), 1, 1)
    block = (BLOCK, 1, 1)
    ctx = Ctx(graph=graph, node=node, env=axis_env_for_flat(live_axes, Var("tid")))

    stmts: list[Stmt] = [
        *emit_body(top_level, ctx),
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinaryExpr(
                "+",
                BinaryExpr("*", Builtin("block_idx.x"), Builtin("block_dim.x")),
                Builtin("thread_idx.x"),
            ),
        ),
        IfStmt(cond=BinaryExpr("<", Var("tid"), Literal(n_threads, "int")), body=emit_body(leaf_body, ctx)),
    ]
    return GpuKernel(name=kernel_name, params=params, body=stmts, block_size=block), arg_order, grid, block


# ---------------------------------------------------------------------------
# Outer-chain split
# ---------------------------------------------------------------------------


def _split_outer(body: tuple[LoopStmt, ...]) -> tuple[tuple[LoopStmt, ...], tuple[LoopStmt, ...], tuple[Axis, ...]]:
    """Pull off scalar top-level stmts and walk the outer free-Loop chain.

    Returns ``(top_level, leaf_body, live_axes)``:
    - ``top_level``: leading non-Loop stmts (typically scalar Loads — must
      not reference any axis Var).
    - ``leaf_body``: the body after descending through the outer free-Loop
      chain. Stops at a level that is non-trivial (has Accum, Write, multiple
      sibling Loops, or a reduce Loop).
    - ``live_axes``: the axes of the descended free Loops, in order.
    """
    top: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = body
    while rest and not isinstance(rest[0], Loop):
        top.append(rest[0])
        rest = rest[1:]
    if any(isinstance(s, Loop) for s in rest[1:]):
        return tuple(top), rest, ()

    cur = rest
    live: list[Axis] = []
    while True:
        # Only descend when this level is exactly ONE stmt and that stmt is a
        # non-reduce Loop. If there are sibling stmts (e.g. a Load hoisted to
        # an outer scope) we can't strip the loop without losing them.
        if len(cur) != 1 or not isinstance(cur[0], Loop):
            return tuple(top), cur, tuple(live)
        only = cur[0]
        if any(isinstance(x, Accum) for x in only.body):
            return tuple(top), cur, tuple(live)
        live.append(only.axis)
        cur = only.body


# ---------------------------------------------------------------------------
# Recursive body walk — every helper returns list[Stmt]
# ---------------------------------------------------------------------------


def emit_body(stmts: tuple[LoopStmt, ...], ctx: Ctx) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            if any(isinstance(x, Accum) for x in s.body):
                out.extend(_emit_reduce_loop(s, ctx))
            else:
                out.extend(_emit_free_loop(s, ctx))
        else:
            emit_stmt(s, ctx, out)
    return out


def _emit_reduce_loop(loop: Loop, ctx: Ctx) -> list[Stmt]:
    """Declare register accumulator, emit serial for-loop, fold inside.

    Multiple ``Accum`` stmts in one Loop body are allowed (rare); each gets
    its own register. The accumulator binding survives in ``ctx.values``
    after this function returns so downstream stmts can read it.
    """
    decls: list[Stmt] = []
    for a in (s for s in loop.body if isinstance(s, Accum)):
        if a.name in ctx.values:
            continue
        acc_var = ctx.fresh()
        identity = a.op.identity if a.op.identity is not None else 0.0
        decls.append(VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")))
        ctx.values[a.name] = Var(acc_var)

    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    ctx.env[loop.axis.name] = Var(k_name)
    inner = emit_body(loop.body, ctx)
    ctx.env = saved_env

    return [*decls, ForLoop(var=k_name, start=Literal(0, "int"), end=Literal(int(loop.axis.extent), "int"), body=inner)]


def _emit_free_loop(loop: Loop, ctx: Ctx) -> list[Stmt]:
    """Free Loop: bound axis → inline body; unbound axis → serial for-loop."""
    if loop.axis.name in ctx.env:
        return emit_body(loop.body, ctx)
    name = f"o{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    ctx.env[loop.axis.name] = Var(name)
    inner = emit_body(loop.body, ctx)
    ctx.env = saved_env
    return [ForLoop(var=name, start=Literal(0, "int"), end=Literal(int(loop.axis.extent), "int"), body=inner)]


__all__ = ["emit_unified"]
