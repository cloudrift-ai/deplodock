"""Unified emit skeleton — replaces Strategy A for pointwise / per-row reduce.

Two shapes, one function shape:

- **pointwise** (``|live|=0``): flat tid over every free axis; one thread per
  output element. No smem, no syncs.
- **per_row_reduce** (``|live|=1``): flat tid over the live axis only; one
  thread per output row. Each thread runs every reduce block serially (k is
  serial per thread), then iterates the output axis serially, emitting the
  output_chain and Write for each element.

Deferred (still falls through to legacy emitter via ``classify() → None``):
matmul (``|live|=2``), multi-live-axis reductions, non-matching live-axis
sets across accums, K-split.

Reuses ``_emit``'s ``_axis_env_for_flat``, ``_flatten_coords``,
``_apply_elementwise``, ``_emit_reduce_accum``, ``_build_params``, and the
stmt walker ``_emit_stmt`` so Load / Assign / Select / Write rendering stays
identical between the unified and legacy paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import ForLoop, GpuKernel, IfStmt, Stmt, VarDecl
from deplodock.compiler.ir.loop import Axis
from deplodock.compiler.pipeline.passes.lowering.kernel._classify import ReduceBlock, UnifiedSig
from deplodock.compiler.pipeline.passes.lowering.kernel._emit import (
    _BLOCK,
    _apply_elementwise,
    _axis_env_for_flat,
    _build_params,
    _Ctx,
    _emit_stmt,
    _emit_stmts,
)

# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


@dataclass
class _Strategy:
    """Minimal strategy stub so ``_Ctx`` can carry state — the unified path
    ignores the strategy kind, but ``_emit_stmt`` reads ``ctx.strategy`` when
    it walks a reduce Loop (we never hit that path here because reduce Loops
    are emitted directly via ``_emit_reduce_block``)."""

    kind: str = "unified"
    outer_free: tuple = ()
    inner_free: tuple = ()
    reduce_axes: tuple = ()
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (_BLOCK, 1, 1)


def emit_unified(
    node: Node, kernel_name: str, graph: Graph, sig: UnifiedSig
) -> tuple[GpuKernel, list[str], tuple[int, int, int], tuple[int, int, int]]:
    params, arg_order = _build_params(node)
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
    n_threads = _numel(free_axes)
    grid = (max((n_threads + _BLOCK - 1) // _BLOCK, 1), 1, 1)
    block = (_BLOCK, 1, 1)
    tid = Var("tid")
    strategy = _Strategy(grid=grid, block=block, outer_free=free_axes)
    ctx = _Ctx(graph=graph, node=node, strategy=strategy, env=_axis_env_for_flat(free_axes, tid))

    # Emit top-level loads first (scalar inputs that aren't axis-indexed).
    top_level = _emit_stmts(sig.top_level, ctx)
    # Emit the output chain (Loads / Assigns / Selects ending with Write).
    body_stmts = _emit_stmts(sig.output_chain, ctx)

    stmts: list[Stmt] = [
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinaryExpr("+", BinaryExpr("*", Builtin("block_idx.x"), Builtin("block_dim.x")), Builtin("thread_idx.x")),
        ),
        IfStmt(cond=BinaryExpr("<", tid, Literal(n_threads, "int")), body=[*top_level, *body_stmts]),
    ]
    return stmts, grid, block


# ---------------------------------------------------------------------------
# Per-row reduce: one thread per live-axis tuple
# ---------------------------------------------------------------------------


def _emit_per_row_reduce(node: Node, graph: Graph, sig: UnifiedSig) -> tuple[list[Stmt], tuple[int, int, int], tuple[int, int, int]]:
    live_axes = sig.live_axes
    assert len(live_axes) >= 1, "per_row_reduce expects at least one live axis"
    n_threads = _numel(live_axes)
    grid = (max((n_threads + _BLOCK - 1) // _BLOCK, 1), 1, 1)
    block = (_BLOCK, 1, 1)
    tid = Var("tid")
    strategy = _Strategy(grid=grid, block=block, outer_free=live_axes)
    ctx = _Ctx(graph=graph, node=node, strategy=strategy, env=_axis_env_for_flat(live_axes, tid))

    # Top-level Loads (scalar broadcasts).
    body: list[Stmt] = _emit_stmts(sig.top_level, ctx)

    # One reduce block at a time: declare acc, walk reduce axis serially.
    for block_idx, rb in enumerate(sig.reduce_blocks):
        body.extend(_emit_reduce_block(rb, ctx, block_idx))

    # Interlude: per-row Assigns between last reduce and output Loop.
    body.extend(_emit_stmts(sig.interlude, ctx))

    # Output free Loop: iterate the output axis serially and emit Write per iter.
    assert sig.output_axis is not None
    body.append(_emit_output_loop(sig.output_axis, sig.output_chain, ctx))

    stmts: list[Stmt] = [
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinaryExpr("+", BinaryExpr("*", Builtin("block_idx.x"), Builtin("block_dim.x")), Builtin("thread_idx.x")),
        ),
        IfStmt(cond=BinaryExpr("<", tid, Literal(n_threads, "int")), body=body),
    ]
    return stmts, grid, block


def _emit_reduce_block(rb: ReduceBlock, ctx: _Ctx, block_idx: int) -> list[Stmt]:
    """Declare the accumulator, emit ``for (k = 0; k < extent; ++k) { body }``.

    ``body`` walks the ReduceBlock's stmts through ``_emit_stmt`` with the
    reduce axis bound to the loop var. The single Accum inside the block
    folds into the declared accumulator register.
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
        _emit_stmt(s, ctx, inner)
    ctx.env = saved_env
    # Preserve the accumulator binding so interlude / output chain can read it.
    saved_values[acc.name] = Var(acc_var)
    ctx.values = saved_values

    return [
        VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")),
        ForLoop(var=k_name, start=Literal(0, "int"), end=Literal(int(rb.axis.extent), "int"), body=inner),
    ]


def _emit_output_loop(axis: Axis, chain: tuple, ctx: _Ctx) -> Stmt:
    """Emit ``for (out = 0; out < extent; ++out) { chain... }`` with the
    output axis bound to the loop var. Chain ends with the single Write."""
    loop_name = f"o{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[axis.name] = Var(loop_name)
    inner: list[Stmt] = []
    for s in chain:
        _emit_stmt(s, ctx, inner)
    ctx.env = saved_env
    ctx.values = saved_values
    return ForLoop(var=loop_name, start=Literal(0, "int"), end=Literal(int(axis.extent), "int"), body=inner)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numel(axes: tuple[Axis, ...]) -> int:
    return int(math.prod(int(a.extent) for a in axes) or 1)


__all__ = ["emit_unified"]


# Silence unused-import warnings for helpers re-used only indirectly via ctx.
_ = (_apply_elementwise, field)
