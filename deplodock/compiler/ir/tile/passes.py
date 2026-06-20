"""Tile-IR Stmt registrations for the shared rewrite / simplify dispatch.

``StageBundle`` carries its ``sources`` directly (the gmem transport
operands) plus the consumer ``body``, an optional cooperative ``compute``
phase, and the transport policy fields. It goes through the
introspection-based handler — adding a new ``Expr`` / ``Axis`` field is
picked up automatically without an override.

StageBundle's ``body`` and optional ``compute`` phase are explicit-recursion
fields: ``_stage_kwargs`` walks ``Expr`` / ``Axis`` content (including the
``sources``' origins / template exprs) but stops at ``Stmt`` boundaries, so
we recurse into the nested bodies ourselves.

Registration runs at module import (loaded from the bottom of
``tile/ir.py`` after class definitions, breaking the tile→stmt→tile cycle).
"""

from __future__ import annotations

from dataclasses import replace as _replace

from deplodock.compiler.ir.axis import extend_simplify_ctx
from deplodock.compiler.ir.expr import Expr, SimplifyCtx
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.passes import AxisFn, Rename, _stage_kwargs, rewrite, simplify
from deplodock.compiler.ir.tile.ir import (
    AsyncWait,
    AtomTile,
    GridTile,
    ParallelTile,
    RegisterTile,
    SerialTile,
    StageBundle,
    StridedTile,
    ThreadTile,
    WarpSpecialize,
    WarpTile,
)


@rewrite.register
def _(s: StageBundle, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    # ``sources`` (Expr in origins / template exprs) ride through ``_stage_kwargs``;
    # ``body`` / ``compute`` are nested Stmt bodies we recurse into explicitly.
    kwargs = _stage_kwargs(s, on_expr=sigma.apply, on_axis=axis_fn)
    kwargs["body"] = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    if s.compute is not None:
        kwargs["compute"] = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.compute)
    return type(s)(**kwargs)


@simplify.register
def _(s: StageBundle, ctx: SimplifyCtx) -> Stmt:
    kwargs = _stage_kwargs(s, on_expr=lambda e: e.simplify(ctx), on_axis=lambda a: a)
    kwargs["body"] = tuple(simplify(c, ctx) for c in s.body)
    if s.compute is not None:
        kwargs["compute"] = tuple(simplify(c, ctx) for c in s.compute)
    return type(s)(**kwargs)


@rewrite.register
def _(s: AsyncWait, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return AsyncWait(
        keep=s.keep,
        phase=sigma.apply(s.phase) if s.phase is not None else None,
        slot=sigma.apply(s.slot) if s.slot is not None else None,
    )


@simplify.register
def _(s: AsyncWait, ctx: SimplifyCtx) -> Stmt:
    return AsyncWait(
        keep=s.keep,
        phase=s.phase.simplify(ctx) if s.phase is not None else None,
        slot=s.slot.simplify(ctx) if s.slot is not None else None,
    )


@rewrite.register
def _(s: WarpSpecialize, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return WarpSpecialize(
        producer_body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.producer_body),
        consumer_body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.consumer_body),
        ring_depth=s.ring_depth,
        n_producer_threads=s.n_producer_threads,
        consumer_thread_axes=tuple(axis_fn(ax) for ax in s.consumer_thread_axes),
        consumer_is_warp=s.consumer_is_warp,
    )


@simplify.register
def _(s: WarpSpecialize, ctx: SimplifyCtx) -> Stmt:
    return WarpSpecialize(
        producer_body=tuple(simplify(c, ctx) for c in s.producer_body),
        consumer_body=tuple(simplify(c, ctx) for c in s.consumer_body),
        ring_depth=s.ring_depth,
        n_producer_threads=s.n_producer_threads,
        consumer_thread_axes=s.consumer_thread_axes,
        consumer_is_warp=s.consumer_is_warp,
    )


# Monoid has no Expr fields — default ``simplify`` (identity) handles it.


# ---------------------------------------------------------------------------
# New tile flavor hierarchy (GridTile / ThreadTile / RegisterTile / SerialTile
# / StridedTile). Each carries a single body (or axes + body); rewrite descends
# into the body recursively, simplify extends the SimplifyCtx with axis ranges.
# ---------------------------------------------------------------------------


def _parallel_rewrite(s: ParallelTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(axis_fn(ax) for ax in s.axes)
    new_body = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    return _replace(s, axes=new_axes, body=new_body)


def _parallel_simplify(s: ParallelTile, ctx: SimplifyCtx) -> Stmt:
    inner = ctx
    for ax in s.axes:
        inner = extend_simplify_ctx(inner, ax)
    new_body = tuple(simplify(c, inner) for c in s.body)
    return _replace(s, body=new_body)


@rewrite.register
def _(s: GridTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    new_axes = tuple(axis_fn(ax) for ax in s.axes)
    new_body = tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body)
    return GridTile(axes=new_axes, body=new_body, swizzle_group_m=s.swizzle_group_m)


@simplify.register
def _(s: GridTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: ThreadTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return _parallel_rewrite(s, rename, sigma, axis_fn)


@simplify.register
def _(s: ThreadTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: RegisterTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return _parallel_rewrite(s, rename, sigma, axis_fn)


@simplify.register
def _(s: RegisterTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: WarpTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return _parallel_rewrite(s, rename, sigma, axis_fn)


@simplify.register
def _(s: WarpTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: AtomTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return _parallel_rewrite(s, rename, sigma, axis_fn)


@simplify.register
def _(s: AtomTile, ctx: SimplifyCtx) -> Stmt:
    return _parallel_simplify(s, ctx)


@rewrite.register
def _(s: SerialTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    return SerialTile(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        kind=s.kind,
        unroll=s.unroll,
    )


@simplify.register
def _(s: SerialTile, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    return SerialTile(
        axis=s.axis,
        body=tuple(simplify(c, inner) for c in s.body),
        kind=s.kind,
        unroll=s.unroll,
    )


@rewrite.register
def _(s: StridedTile, rename: Rename, sigma: Sigma, axis_fn: AxisFn) -> Stmt:
    step = sigma.apply(s.step) if isinstance(s.step, Expr) else s.step
    return StridedTile(
        axis=axis_fn(s.axis),
        body=tuple(rewrite(c, rename, sigma, axis_fn) for c in s.body),
        start=sigma.apply(s.start),
        step=step,
        unroll=s.unroll,
    )


@simplify.register
def _(s: StridedTile, ctx: SimplifyCtx) -> Stmt:
    inner = extend_simplify_ctx(ctx, s.axis)
    step = s.step.simplify(ctx) if isinstance(s.step, Expr) else s.step
    return StridedTile(
        axis=s.axis,
        body=tuple(simplify(c, inner) for c in s.body),
        start=s.start.simplify(ctx),
        step=step,
        unroll=s.unroll,
    )
