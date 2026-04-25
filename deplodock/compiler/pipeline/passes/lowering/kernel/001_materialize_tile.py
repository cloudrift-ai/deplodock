"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

Reads each ``Tile`` in the TileOp body and emits the concrete
hardware shape — ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop`` — that ``render_kernelop`` consumes.

The Tile→Enclosure mapping is structural: both nodes carry
``axes: tuple[BoundAxis, ...]``. Materialization is then:

- All ``BoundAxis`` in the Tile are ``BIND_THREAD`` (pointwise / per-
  thread serial) → ``Enclosure(axes=blk.axes)``. Inner ``BoundLoop``s
  fall back to serial Loop-IR ``Loop``s.
- Any ``BoundAxis`` is ``BIND_BLOCK`` (cooperative) →
  ``Enclosure(axes=(BoundAxis(t, BIND_THREAD), *blk.axes))``, where
  ``t`` is the synthetic cooperative thread axis introduced here.
  Inner ``BoundLoop`` with ``bind=BIND_BLOCK_STRIDED`` becomes ``StridedLoop`` driven
  by ``t``; ``Combine`` siblings emit the smem tree-halve phase and
  broadcast loads; ``Stmt.rewrite`` renames subsequent Accum reads to
  ``<name>_b``.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BoundLoop,
    Combine,
    Stage,
    Tile,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

BLOCK_SIZE = 256


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body: list[Stmt] = []
    for s in tile_op.body:
        if isinstance(s, Tile):
            new_body.append(_materialize(s))
        else:
            new_body.append(s)

    node.op = KernelOp(body=tuple(new_body), name=tile_op.name)
    return None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Tile) -> Stmt:
    if blk.block_axes:
        return _materialize_cooperative(blk.axes, blk.body)
    return _materialize_thread_per_output(blk.axes, blk.body)


def _materialize_thread_per_output(axes: tuple, body: tuple) -> Stmt:
    """One thread per output element. ``axes`` is passed through
    unchanged — every BoundAxis is already ``BIND_THREAD``."""
    lowered = tuple(_lower_uncooperative(s) for s in body)
    return Enclosure(axes=axes, body=lowered)


def _lower_uncooperative(s: Stmt) -> Stmt:
    """Translate a ``BoundLoop(bind=SERIAL)`` tree to Loop-IR ``Loop``.
    Leaves pass through. ``Combine`` must not appear in a non-cooperative
    Tile (no strategy places it without setting ``block_axes``)."""
    if isinstance(s, BoundLoop):
        if s.bind != BIND_SERIAL:
            raise ValueError(f"non-cooperative Tile cannot contain BoundLoop with bind={s.bind!r}")
        return Loop(axis=s.axis, body=tuple(_lower_uncooperative(c) for c in s.body))
    if isinstance(s, Combine):
        raise ValueError("Combine not allowed in non-cooperative Tile (block_axes must be populated)")
    return s


def _materialize_cooperative(axes: tuple, body: tuple) -> Stmt:
    """Cooperative materialization: one CUDA block per output point;
    a synthetic ``t`` thread axis drives cooperation. ``axes`` carries
    the output BoundAxes (all ``BIND_BLOCK``); the synthetic ``t`` is
    prepended as ``BIND_THREAD``."""
    t_axis = Axis(name="t", extent=BLOCK_SIZE)
    rename: dict[str, str] = {}
    # buf -> (stage_buf, cache_positions): record Stage redirects so
    # subsequent Loads of `buf` get rewritten to read from the cached
    # smem buffer with only the cache-dimension positions of their index.
    redirects: dict[str, tuple[str, tuple[int, ...]]] = {}

    def transform(s: Stmt) -> Stmt:
        """Apply Stage load-redirects + Combine SSA renames to a stmt tree."""
        if redirects:
            s = _redirect_loads(s, redirects)
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    pending_reduce: tuple[BoundLoop, Accum] | None = None

    for stmt in body:
        if isinstance(stmt, Stage):
            stage_buf = f"{stmt.buf}_stage"
            new_body.extend(_emit_stage(stmt, stage_buf, t_axis.name))
            stage_axis_names = {ax.name for ax in stmt.axes}
            cache_positions = tuple(i for i, e in enumerate(stmt.index) if isinstance(e, Var) and e.name in stage_axis_names)
            redirects[stmt.buf] = (stage_buf, cache_positions)
            pending_reduce = None
        elif isinstance(stmt, BoundLoop):
            pending_reduce = None
            if _is_reduce(stmt):
                accum = next(a for a in stmt.body if isinstance(a, Accum))
                new_body.append(_emit_strided(stmt, t_axis.name, transform))
                pending_reduce = (stmt, accum)
            else:
                new_body.append(_emit_strided(stmt, t_axis.name, transform))
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce BoundLoop")
            reduce_loop, accum = pending_reduce
            if accum.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({accum.name!r})")
            # Combine scope is derived from the surrounding BoundLoop's bind:
            # BIND_BLOCK_STRIDED → smem tree-halve at block scope.
            phase = _emit_combine(stmt, accum, reduce_loop.bind, t_axis.name)
            new_body.extend(phase)
            if reduce_loop.bind == BIND_BLOCK_STRIDED:
                rename[accum.name] = f"{accum.name}_b"
            pending_reduce = None
        elif isinstance(stmt, Write):
            new_body.append(
                Cond(
                    cond=BinaryExpr("==", Var(t_axis.name), Literal(0, "int")),
                    body=(transform(stmt),),
                    else_body=(),
                )
            )
        else:
            new_body.append(transform(stmt))

    # Cooperative thread axis ``t`` (BIND_THREAD) plus the original output
    # axes — but BIND_BLOCK_STRIDED axes are filtered out because they
    # don't contribute to launch geometry (the body's strided BoundLoops
    # handle their iteration).
    launch_axes = tuple(ba for ba in axes if ba.bind != BIND_BLOCK_STRIDED)
    new_axes = (BoundAxis(axis=t_axis, bind=BIND_THREAD), *launch_axes)
    return Enclosure(axes=new_axes, body=tuple(new_body))


def _emit_strided(loop: BoundLoop, t: str, renamed) -> Stmt:
    body = tuple(_lower_inner(c, renamed) for c in loop.body)
    if loop.bind == BIND_BLOCK_STRIDED:
        return StridedLoop(axis=loop.axis, start=Var(t), step=BLOCK_SIZE, body=body)
    if loop.bind == BIND_SERIAL:
        return Loop(axis=loop.axis, body=body)
    raise NotImplementedError(f"BoundLoop bind={loop.bind!r} inside cooperative Tile not yet handled")


def _lower_inner(s: Stmt, renamed) -> Stmt:
    if isinstance(s, BoundLoop):
        return Loop(axis=s.axis, body=tuple(_lower_inner(c, renamed) for c in s.body))
    return renamed(s)


def _emit_combine(combine: Combine, accum: Accum, scope: str, t: str) -> list[Stmt]:
    """Emit the cross-thread combine. ``scope`` is the surrounding reduce
    BoundLoop's bind value, which drives the combine mechanism:

    - ``BIND_BLOCK_STRIDED`` → smem tree-halve at block scope.
    - ``BIND_SERIAL`` → no combine (each thread's partial is already
      the final value; legal but unused today since strategies don't
      emit Combine after a serial loop).
    - Future: ``BIND_WARP_STRIDED`` → warp-shuffle (no smem).
    """
    if scope == BIND_SERIAL:
        return []
    if scope == BIND_BLOCK_STRIDED:
        smem_name = f"{accum.name}_smem"
        broadcast_name = f"{accum.name}_b"
        return [
            Smem(name=smem_name, extents=(BLOCK_SIZE,)),
            Write(output=smem_name, index=(Var(t),), value=accum.name),
            Sync(),
            TreeHalve(buf=smem_name, op=accum.op, length=BLOCK_SIZE, tid_var=t),
            Sync(),
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ]
    raise NotImplementedError(f"Combine for surrounding bind={scope!r} not yet handled")


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


# ---------------------------------------------------------------------------
# Stage expansion + load redirect
# ---------------------------------------------------------------------------


def _emit_stage(stage: Stage, stage_buf: str, t: str) -> list[Stmt]:
    """Expand a ``Stage`` Stmt into the smem decl + cooperative load +
    sync sequence that loads the operand once into per-block smem.

    Single-axis stage emits a direct ``StridedLoop(cache_axis, start=t,
    step=BLOCK_SIZE)`` (each thread handles every BLOCK_SIZE-th cache
    position). Multi-axis stage flattens the cache axes into one
    synthetic linear axis of extent ``prod(extents)``; each thread owns
    every BLOCK_SIZE-th flat position, decoded back into the per-axis
    coordinates for the source Load and smem Write indices.
    """
    extents = tuple(int(ax.extent) for ax in stage.axes)
    if not extents:
        raise ValueError(f"Stage {stage.buf!r} has no cache axes")
    smem = Smem(name=stage_buf, extents=extents)
    load_name = f"_stage_{stage.buf}_v"

    if len(stage.axes) == 1:
        cache_axis = stage.axes[0]
        cooperative_load = StridedLoop(
            axis=cache_axis,
            start=Var(t),
            step=BLOCK_SIZE,
            body=(
                Load(name=load_name, input=stage.buf, index=stage.index),
                Write(output=stage_buf, index=(Var(cache_axis.name),), value=load_name),
            ),
        )
        return [smem, cooperative_load, Sync()]

    # Multi-axis: row-major flatten cache_axes into a synthetic linear axis.
    total = 1
    for e in extents:
        total *= e
    flat_name = f"_stage_{stage.buf}_flat"
    flat_axis = Axis(name=flat_name, extent=total)
    decode_sigma = _flat_decode_sigma(stage.axes, flat_name)

    source_index = tuple(decode_sigma.apply(e) for e in stage.index)
    smem_index = tuple(decode_sigma.apply(Var(ax.name)) for ax in stage.axes)
    cooperative_load = StridedLoop(
        axis=flat_axis,
        start=Var(t),
        step=BLOCK_SIZE,
        body=(
            Load(name=load_name, input=stage.buf, index=source_index),
            Write(output=stage_buf, index=smem_index, value=load_name),
        ),
    )
    return [smem, cooperative_load, Sync()]


def _flat_decode_sigma(cache_axes: tuple[Axis, ...], flat_name: str) -> Sigma:
    """Decode a flat row-major index back into per-axis coordinates.

    For ``cache_axes = (m_i ext=BM, k_i ext=BK)`` and a flat var ``F``::

        m_i = F / BK
        k_i = F % BK

    Generalizes to N axes — innermost gets ``F % extent``, the outer
    chain gets ``(F / inner_stride) % extent`` for middle axes and
    ``F / inner_stride`` for the outermost (no mod needed)."""
    flat = Var(flat_name)
    mapping: dict = {}
    inner_stride = 1
    for ax in reversed(cache_axes):
        ext = int(ax.extent)
        if inner_stride == 1:
            mapping[ax.name] = flat % Literal(ext, "int")
        else:
            mapping[ax.name] = (flat / Literal(inner_stride, "int")) % Literal(ext, "int")
        inner_stride *= ext
    # Simplify the outermost: it's flat / outer_stride with no mod (the
    # mod by its own extent is redundant since flat < total = outer_extent * outer_stride).
    outer = cache_axes[0]
    outer_stride = 1
    for ax in cache_axes[1:]:
        outer_stride *= int(ax.extent)
    if outer_stride == 1:
        mapping[outer.name] = flat
    else:
        mapping[outer.name] = flat / Literal(outer_stride, "int")
    return Sigma(mapping)


def _redirect_loads(stmt: Stmt, redirects: dict[str, tuple[str, tuple[int, ...]]]) -> Stmt:
    """Recursively rewrite ``Load(buf, ...)`` to ``Load(stage_buf, projected_index)``
    for every ``buf`` in ``redirects``. ``projected_index`` keeps only
    the cache-position entries of the original index (block-bound
    positions are dropped because the staged buffer is per-block)."""
    if isinstance(stmt, Load) and stmt.input in redirects:
        stage_buf, cache_positions = redirects[stmt.input]
        new_index = tuple(stmt.index[i] for i in cache_positions)
        return Load(name=stmt.name, input=stage_buf, index=new_index)
    if isinstance(stmt, BoundLoop):
        return BoundLoop(
            axis=stmt.axis,
            body=tuple(_redirect_loads(c, redirects) for c in stmt.body),
        )
    if isinstance(stmt, Loop):
        return Loop(axis=stmt.axis, body=tuple(_redirect_loads(c, redirects) for c in stmt.body))
    if isinstance(stmt, StridedLoop):
        return StridedLoop(
            axis=stmt.axis,
            start=stmt.start,
            step=stmt.step,
            body=tuple(_redirect_loads(c, redirects) for c in stmt.body),
        )
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_redirect_loads(c, redirects) for c in stmt.body),
            else_body=tuple(_redirect_loads(c, redirects) for c in stmt.else_body),
        )
    return stmt
