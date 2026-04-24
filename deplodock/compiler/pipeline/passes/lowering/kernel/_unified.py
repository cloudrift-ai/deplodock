"""Unified emit skeleton — replaces Strategy A for pointwise / per-row reduce.

Two shapes, one function shape:

- **pointwise** (``|live|=0``): flat tid over every free axis; one thread per
  output element. No smem, no syncs.
- **per_row_reduce** (``|live|=1``): flat tid over the live axis only; one
  thread per output row. Each thread runs every reduce block serially (k is
  serial per thread), then iterates the output axis serially, emitting the
  output_chain and Write for each element.

Deferred (still falls through via classifier returning ``None``): matmul
(``|live|=2``, handled by the annotated template), multi-live-axis
reductions, non-matching live-axis sets across accums, K-split.

Reuses ``_common.emit_stmt`` for Load / Assign / Select / Write / Accum
rendering so the generated CUDA is byte-identical to what the legacy
emitter produced for the same body.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import ForLoop, GpuKernel, IfStmt, Stmt, VarDecl
from deplodock.compiler.ir.loop import Axis
from deplodock.compiler.pipeline.passes.lowering.kernel._classify import ReduceBlock, UnifiedSig
from deplodock.compiler.pipeline.passes.lowering.kernel._common import (
    BLOCK,
    Ctx,
    axis_env_for_flat,
    build_params,
    emit_stmt,
    emit_stmts,
    numel_axes,
)


def emit_unified(
    node: Node, kernel_name: str, graph: Graph, sig: UnifiedSig
) -> tuple[GpuKernel, list[str], tuple[int, int, int], tuple[int, int, int]]:
    params, arg_order = build_params(node)
    if sig.kind == "pointwise":
        body, grid, block = _emit_pointwise(node, graph, sig)
    elif sig.kind == "per_row_reduce":
        body, grid, block = _emit_per_row_reduce(node, graph, sig)
    else:
        raise NotImplementedError(f"unified emit: shape {sig.kind!r} not supported")
    kd = GpuKernel(name=kernel_name, params=params, body=body, block_size=block)
    return kd, arg_order, grid, block


# ---------------------------------------------------------------------------
# Pointwise: one thread per output element
# ---------------------------------------------------------------------------


def _emit_pointwise(node: Node, graph: Graph, sig: UnifiedSig) -> tuple[list[Stmt], tuple[int, int, int], tuple[int, int, int]]:
    free_axes = sig.free_axes
    n_threads = numel_axes(free_axes)
    grid = (max((n_threads + BLOCK - 1) // BLOCK, 1), 1, 1)
    block = (BLOCK, 1, 1)
    tid = Var("tid")
    ctx = Ctx(graph=graph, node=node, env=axis_env_for_flat(free_axes, tid))

    top_level = emit_stmts(sig.top_level, ctx)
    body_stmts = emit_stmts(sig.output_chain, ctx)

    return _wrap_with_tid_guard(n_threads, [*top_level, *body_stmts]), grid, block


# ---------------------------------------------------------------------------
# Per-row reduce: one thread per live-axis tuple
# ---------------------------------------------------------------------------


def _emit_per_row_reduce(node: Node, graph: Graph, sig: UnifiedSig) -> tuple[list[Stmt], tuple[int, int, int], tuple[int, int, int]]:
    live_axes = sig.live_axes
    assert len(live_axes) >= 1, "per_row_reduce expects at least one live axis"
    n_threads = numel_axes(live_axes)
    grid = (max((n_threads + BLOCK - 1) // BLOCK, 1), 1, 1)
    block = (BLOCK, 1, 1)
    tid = Var("tid")
    ctx = Ctx(graph=graph, node=node, env=axis_env_for_flat(live_axes, tid))

    body: list[Stmt] = emit_stmts(sig.top_level, ctx)
    body.extend(emit_stmts(sig.pre_reduce, ctx))
    for block_idx, rb in enumerate(sig.reduce_blocks):
        body.extend(_emit_reduce_block(rb, ctx, block_idx))
    body.extend(emit_stmts(sig.interlude, ctx))
    if sig.output_axis is None:
        # No output Loop — emit the single Write (and any chain stmts) directly.
        body.extend(emit_stmts(sig.output_chain, ctx))
    else:
        body.append(_emit_output_loop(sig.output_axis, sig.output_chain, ctx))

    return _wrap_with_tid_guard(n_threads, body), grid, block


def _emit_reduce_block(rb: ReduceBlock, ctx: Ctx, block_idx: int) -> list[Stmt]:
    """Declare the accumulator, emit ``for (k = 0; k < extent; ++k) { body }``.

    The single Accum inside the block folds into the declared register; its
    binding survives in ``ctx.values`` so interlude / output chain can read it.
    """
    acc = rb.accum
    acc_var = f"acc{block_idx}"
    identity = acc.op.identity if acc.op.identity is not None else 0.0
    ctx.values[acc.name] = Var(acc_var)

    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[rb.axis.name] = Var(k_name)
    inner: list[Stmt] = []
    for s in rb.body_stmts:
        emit_stmt(s, ctx, inner)
    ctx.env = saved_env
    saved_values[acc.name] = Var(acc_var)
    ctx.values = saved_values

    return [
        VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")),
        ForLoop(var=k_name, start=Literal(0, "int"), end=Literal(int(rb.axis.extent), "int"), body=inner),
    ]


def _emit_output_loop(axis: Axis, chain: tuple, ctx: Ctx) -> Stmt:
    """``for (out = 0; out < extent; ++out) { chain... }`` — chain ends with Write."""
    loop_name = f"o{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[axis.name] = Var(loop_name)
    inner: list[Stmt] = []
    for s in chain:
        emit_stmt(s, ctx, inner)
    ctx.env = saved_env
    ctx.values = saved_values
    return ForLoop(var=loop_name, start=Literal(0, "int"), end=Literal(int(axis.extent), "int"), body=inner)


def _wrap_with_tid_guard(n_threads: int, body: list[Stmt]) -> list[Stmt]:
    tid = Var("tid")
    return [
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinaryExpr(
                "+",
                BinaryExpr("*", Builtin("block_idx.x"), Builtin("block_dim.x")),
                Builtin("thread_idx.x"),
            ),
        ),
        IfStmt(cond=BinaryExpr("<", tid, Literal(n_threads, "int")), body=body),
    ]


__all__ = ["emit_unified"]
