"""Stamp per-statement dtypes on every Load / Assign / Write / Source.

Runs once over a ``TileOp`` body before any analytical pass (demote,
split_register_axes, vectorize_loads, permute, pack_fp16, vectorize_stores)
sees the IR. After this pass, downstream passes read dtypes directly
off the IR instead of reaching for matcher-populated ``KernelOp``
side channels (``inputs`` / ``outputs`` / ``smem_buffers``) or walking
the graph node table.

Stamping rules:

- ``Source(name, buf, ...)`` — ``dtype = graph.nodes[buf].output.dtype``
  when ``buf`` names a graph node (the usual transport-Stage case);
  resolves through a sibling-Stage's ``source.dtype`` when ``buf``
  names another Stage's smem slab (``ComputeStage``).
- ``Load(input=B)`` — ``dtype`` resolves to:
  - the enclosing Stage's matching ``source.dtype`` if ``B`` matches
    a Stage source name in scope (the body's Loads against staged smem),
  - else ``graph.nodes[B].output.dtype`` (graph-buffer Load — typical
    inside a Stage's cooperative-load body that we don't model
    structurally, or top-level Loads outside any Stage).
- ``Assign(op, args)`` — ``dtype = dtype_promote(op.name, [ssa[a] for a
  in args])``. Lifted out of the renderer; same rule as today's
  ``Assign.render`` inline check.
- ``Write(output, value)`` — ``value_dtype = ssa[value]``. The
  destination buffer's dtype is render-time concern (still read off
  ``ctx.buffer_dtypes``); only the value-side dtype is stamped.
- ``Accum`` / ``Init`` / ``Pack`` / ``Unpack`` are already typed.
  Register them in the running ``ssa_dtypes`` so downstream Assigns
  / Writes pick up the right arg dtype.

Idempotent. ``None``-stamped fields are filled; already-stamped ones
are kept unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Init,
    Load,
    Loop,
    Pack,
    Stmt,
    StridedLoop,
    Unpack,
    Write,
)
from deplodock.compiler.ir.stmt.base import dtype_promote
from deplodock.compiler.ir.tile.ir import Source, Stage, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


@dataclass
class _StampCtx:
    """Per-walk state. ``ssa_dtypes`` accumulates as Stmts are stamped;
    ``source_dtypes`` maps in-scope Stage source ``name``s to the
    stamped Source dtype (so body Loads referencing the smem slab
    resolve to the same dtype as the producer-side Load)."""

    graph: Graph
    ssa_dtypes: dict[str, DataType] = field(default_factory=dict)
    source_dtypes: dict[str, DataType] = field(default_factory=dict)


def rewrite(match: Match, root: Node) -> Graph | None:
    ctx = _StampCtx(graph=match.graph)
    new_body = _stamp_body(root.op.body, ctx)
    if new_body == root.op.body:
        raise RuleSkipped("every Load/Assign/Write/Source already stamped")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _stamp_body(body: Body, ctx: _StampCtx) -> Body:
    return Body(_stamp_stmt(s, ctx) for s in body)


def _stamp_stmt(s: Stmt, ctx: _StampCtx) -> Stmt:
    if isinstance(s, Stage):
        return _stamp_stage(s, ctx)
    if isinstance(s, Load):
        return _stamp_load(s, ctx)
    if isinstance(s, Assign):
        return _stamp_assign(s, ctx)
    if isinstance(s, Write):
        return _stamp_write(s, ctx)
    if isinstance(s, (Accum, Init)):
        ctx.ssa_dtypes[s.name] = s.dtype or F32
        return s
    if isinstance(s, Pack):
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    if isinstance(s, Unpack):
        ctx.ssa_dtypes[s.low_name] = s.lane_dtype
        ctx.ssa_dtypes[s.high_name] = s.lane_dtype
        return s
    # Block-structured stmts: recurse through children. Use ``nested()`` /
    # ``with_bodies()`` so every block flavor (Loop, StridedLoop, Cond,
    # GridTile / ThreadTile / RegisterTile / SerialTile / StridedTile, etc.)
    # works without an isinstance ladder.
    nested = s.nested()
    if not nested:
        return s
    new_bodies = tuple(_stamp_body(b, ctx) for b in nested)
    return s.with_bodies(new_bodies)


def _stamp_stage(s: Stage, ctx: _StampCtx) -> Stmt:
    """Stamp each Source.dtype, then recurse into Stage bodies with the
    source map pushed onto ``ctx.source_dtypes``. Restores the prior
    source map on exit so sibling Stages with the same source name don't
    leak into each other."""
    new_sources = tuple(_stamp_source(src, ctx) for src in s.sources)
    saved = dict(ctx.source_dtypes)
    for src in new_sources:
        if src.dtype is not None:
            ctx.source_dtypes[src.name] = src.dtype
    new_bodies = tuple(_stamp_body(b, ctx) for b in s.nested())
    ctx.source_dtypes = saved
    new_stage = replace(s, sources=new_sources)
    return new_stage.with_bodies(new_bodies)


def _stamp_source(src: Source, ctx: _StampCtx) -> Source:
    if src.dtype is not None:
        return src
    # ``src.buf`` typically names a graph node (transport Stage); for a
    # ComputeStage it can name a sibling Stage's smem slab instead.
    dt = ctx.source_dtypes.get(src.buf)
    if dt is None:
        node = ctx.graph.nodes.get(src.buf)
        if node is not None:
            dt = node.output.dtype
    if dt is None:
        # No information available — leave unstamped so the render-time
        # fallback applies. Should be rare in production paths.
        return src
    return replace(src, dtype=dt)


def _resolve_buf_dtype(buf: str, ctx: _StampCtx) -> DataType | None:
    dt = ctx.source_dtypes.get(buf)
    if dt is not None:
        return dt
    node = ctx.graph.nodes.get(buf)
    if node is not None:
        return node.output.dtype
    return None


def _stamp_load(s: Load, ctx: _StampCtx) -> Load:
    dt = s.dtype if s.dtype is not None else _resolve_buf_dtype(s.input, ctx)
    if dt is not None:
        for n in s.names:
            ctx.ssa_dtypes[n] = dt
        if s.dtype is None:
            return Load(names=s.names, input=s.input, index=s.index, dtype=dt)
    return s


def _stamp_assign(s: Assign, ctx: _StampCtx) -> Assign:
    if s.dtype is not None:
        ctx.ssa_dtypes[s.name] = s.dtype
        return s
    arg_dtypes = [(ctx.ssa_dtypes.get(a) or F32).name for a in s.args]
    result_name = dtype_promote(s.op.name, arg_dtypes)
    # Resolve canonical name back to a DataType. F32 / F16 are the only
    # ones promote returns today; lazy import to avoid cycle with dtype.get.
    from deplodock.compiler.dtype import get as _get  # noqa: PLC0415

    result_dt = _get(result_name)
    ctx.ssa_dtypes[s.name] = result_dt
    return replace(s, dtype=result_dt)


def _stamp_write(s: Write, ctx: _StampCtx) -> Write:
    if s.value_dtype is not None:
        return s
    # In a multi-value (vector) Write all values share the same SSA dtype
    # by construction (the vectorize pass widens runs of same-dtype Loads).
    primary = s.values[0]
    dt = ctx.ssa_dtypes.get(primary)
    if dt is None:
        return s
    return Write(
        output=s.output,
        index=s.index,
        values=s.values,
        value_dtype=dt,
    )


# Silence unused-import warnings — these symbols are referenced via
# isinstance only and not all imports are exercised on every IR.
_ = (Loop, StridedLoop, Cond)
