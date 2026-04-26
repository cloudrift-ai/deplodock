"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

Reads each ``Tile`` in the TileOp body and emits the concrete hardware
shape — ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop`` — that ``render_kernelop`` consumes.

The Tile→Enclosure mapping is structural: both nodes carry
``axes: tuple[BoundAxis, ...]``. Two paths:

- **Non-cooperative** (no ``BIND_BLOCK`` axes): every BoundAxis is
  ``BIND_THREAD`` (pointwise / per-thread serial) → ``Enclosure`` with
  ``axes`` passed through. Inner ``BoundLoop``s become serial Loops.

- **Cooperative** (one or more ``BIND_BLOCK`` axes): the Tile's THREAD
  axes are the cooperative thread set (synthesized by the strategy:
  ``cooperative-reduce`` adds a single ``t`` axis; ``blockify`` uses
  the per-block tile dims ``m_i`` / ``n_i``). Materialization passes
  ``Tile.axes`` through to the Enclosure, computes a linear thread
  index ``tid_expr`` from the THREAD axes, then walks the body:

    * ``Stage`` → smem decl + cooperative load driven by ``tid_expr``
      (multi-axis stages flatten via row-major decode).
    * ``BoundLoop(BIND_BLOCK_STRIDED)`` → ``StridedLoop(start=tid_expr,
      step=BLOCK_SIZE)`` over the axis. ``Combine`` siblings emit smem
      tree-halve + broadcast.
    * ``BoundLoop(BIND_SERIAL)`` → plain ``Loop``.
    * ``Write`` whose index references a THREAD axis is emitted
      unconditionally (each thread owns a unique output slot). Writes
      that don't reference any THREAD axis are guarded by ``tid==0`` so
      only one thread writes.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var, free_vars
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BLOCK_SIZE,
    BoundLoop,
    Combine,
    Stage,
    Tile,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]


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
    """Cooperative materialization: one CUDA block per ``BIND_BLOCK`` axis
    coordinate; the THREAD axes carried in ``Tile.axes`` are the
    cooperative thread set. Strategies populate THREAD axes upfront —
    this pass commits no axis decisions of its own."""
    thread_axes = tuple(ba for ba in axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise ValueError("cooperative Tile must have at least one BIND_THREAD axis")
    thread_axis_names = {ba.axis.name for ba in thread_axes}
    tid_expr = _build_linear_tid(thread_axes)

    rename: dict[str, str] = {}
    # buf -> (stage_buf, stage): subsequent Loads of `buf` get rewritten to
    # read from the smem cache via _smem_coords_for_load.
    redirects: dict[str, tuple[str, Stage]] = {}

    def transform(s: Stmt) -> Stmt:
        if redirects:
            s = _redirect_loads(s, redirects)
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    pending_reduce: tuple[BoundLoop, Accum] | None = None
    # Axes whose Var distinguishes threads at the current scope. Starts as
    # the THREAD axes; extended by ``_emit_loop`` when descending into a
    # BIND_BLOCK_STRIDED loop (each thread visits distinct iterations).
    distinguishing = set(thread_axis_names)

    for stmt in body:
        if isinstance(stmt, Stage):
            stage_buf = f"{stmt.buf}_stage"
            new_body.extend(_emit_stage(stmt, stage_buf, tid_expr))
            redirects[stmt.buf] = (stage_buf, stmt)
            pending_reduce = None
        elif isinstance(stmt, BoundLoop):
            pending_reduce = None
            new_body.append(_emit_loop(stmt, tid_expr, thread_axes, distinguishing, redirects, transform))
            if stmt.bind == BIND_BLOCK_STRIDED and _is_reduce(stmt):
                accum = next(a for a in stmt.body if isinstance(a, Accum))
                pending_reduce = (stmt, accum)
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce BoundLoop")
            reduce_loop, accum = pending_reduce
            if accum.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({accum.name!r})")
            phase = _emit_combine(stmt, accum, reduce_loop.bind, _single_thread_var(thread_axes))
            new_body.extend(phase)
            if reduce_loop.bind == BIND_BLOCK_STRIDED:
                rename[accum.name] = f"{accum.name}_b"
            pending_reduce = None
        elif isinstance(stmt, Write):
            new_body.append(_emit_write(stmt, distinguishing, thread_axes, transform))
        else:
            new_body.append(transform(stmt))

    # Hoist Accum inits to Enclosure scope so nested-reduce shapes (matmul
    # ``Loop(k_o) > Loop(k_i) > Accum``) don't reset per outer iteration.
    # The renderer's explicit_inits suppression makes this a no-op for
    # softmax-style single-Loop reductions — same emitted CUDA either way.
    inits = _collect_init_stmts(new_body)
    new_body = [*inits, *new_body]

    # Pass Tile.axes through — strategies committed the launch layout
    # (THREAD + BLOCK only). Cooperatively-walked body axes live on their
    # BoundLoop's bind, not here.
    return Enclosure(axes=axes, body=tuple(new_body))


def _emit_loop(
    loop: BoundLoop,
    tid_expr,
    thread_axes: tuple,
    distinguishing: set,
    redirects: dict,
    transform,
) -> Stmt:
    """Translate a body BoundLoop. Recurses so nested staging / loops /
    writes inside the loop body get the same uniform treatment.

    A BIND_BLOCK_STRIDED loop adds its axis to ``distinguishing`` for
    nested-scope Writes — threads visit distinct iterations of the
    strided axis, so a Write whose index references it gets distinct
    output positions per thread (no guard needed)."""
    inner_distinguishing = distinguishing | ({loop.axis.axis.name} if loop.bind == BIND_BLOCK_STRIDED else set())
    inner: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, Stage):
            stage_buf = f"{s.buf}_stage"
            inner.extend(_emit_stage(s, stage_buf, tid_expr))
            redirects[s.buf] = (stage_buf, s)
        elif isinstance(s, BoundLoop):
            inner.append(_emit_loop(s, tid_expr, thread_axes, inner_distinguishing, redirects, transform))
        elif isinstance(s, Write):
            inner.append(_emit_write(s, inner_distinguishing, thread_axes, transform))
        else:
            inner.append(transform(s))
    body = tuple(inner)

    if loop.bind == BIND_BLOCK_STRIDED:
        return StridedLoop(axis=loop.axis, start=tid_expr, step=BLOCK_SIZE, body=body)
    if loop.bind == BIND_SERIAL:
        return Loop(axis=loop.axis, body=body)
    if loop.bind == BIND_THREAD:
        # The thread axis is bound by the Enclosure decode — strip the
        # BoundLoop and inline its body.
        return _StripThreadLoop(body=body)
    raise NotImplementedError(f"BoundLoop bind={loop.bind!r} inside cooperative Tile not handled")


class _StripThreadLoop:
    """Sentinel returned by ``_emit_loop`` when a BoundLoop iterates a
    THREAD axis — the loop is dropped and its body inlined into the
    surrounding scope. (Today's strategies don't emit BoundLoops over
    THREAD axes, but the path exists for future strategies.)"""

    def __init__(self, body):
        self.body = body


def _emit_write(
    write: Write,
    distinguishing: set,
    thread_axes: tuple,
    transform,
) -> Stmt:
    """Emit a Write — guarded by ``all thread_axes == 0`` if the Write
    doesn't address any *distinguishing* axis (THREAD axes plus any
    enclosing BIND_BLOCK_STRIDED loop axes). Otherwise unconditional —
    each thread writes a unique output position."""
    write = transform(write)
    write_free = set()
    for e in write.index:
        write_free |= free_vars(e)
    if distinguishing & write_free:
        return write
    return Cond(cond=_all_threads_zero(thread_axes), body=(write,), else_body=())


def _all_threads_zero(thread_axes: tuple) -> BinaryExpr:
    """Build ``t0 == 0 && t1 == 0 && ...`` over the THREAD axes."""
    cond = None
    for ba in thread_axes:
        eq = BinaryExpr("==", Var(ba.axis.name), Literal(0, "int"))
        cond = eq if cond is None else BinaryExpr("&&", cond, eq)
    return cond  # type: ignore[return-value]


def _single_thread_var(thread_axes: tuple) -> str:
    """Combine + TreeHalve emit a single ``tid_var`` string. Only valid
    when there's exactly one THREAD axis — softmax-style cooperation
    (matmul has multi-axis THREAD set but doesn't emit Combine)."""
    if len(thread_axes) != 1:
        raise ValueError(f"Combine requires a single THREAD axis; got {len(thread_axes)}")
    return thread_axes[0].axis.name


def _collect_init_stmts(stmts: list[Stmt]) -> list[Stmt]:
    """Walk transitively for distinct Accums; return one Init per name."""

    def walk(ss):
        for s in ss:
            if isinstance(s, Accum):
                yield s
            for child_body in s.nested():
                yield from walk(child_body)

    seen: dict[str, Accum] = {}
    for accum in walk(stmts):
        seen.setdefault(accum.name, accum)
    return [Init(name=name, op=accum.op) for name, accum in seen.items()]


def _build_linear_tid(thread_axes: tuple[BoundAxis, ...]):
    """Linear row-major thread index from the THREAD axes.

    Single-axis (softmax) → ``Var(name)``.
    Multi-axis (matmul) → ``m_i * BN + n_i`` for ``(m_i, n_i)``."""
    if len(thread_axes) == 1:
        return Var(thread_axes[0].axis.name)
    inner_stride = 1
    parts: list = []
    for ba in reversed(thread_axes):
        ext = int(ba.extent)
        if inner_stride == 1:
            parts.append(Var(ba.axis.name))
        else:
            parts.append(Var(ba.axis.name) * Literal(inner_stride, "int"))
        inner_stride *= ext
    expr = parts[0]
    for p in parts[1:]:
        expr = p + expr
    return expr


def _emit_combine(combine: Combine, accum: Accum, scope: str, t: str) -> list[Stmt]:
    """Emit the cross-thread combine. ``scope`` is the surrounding reduce
    BoundLoop's bind value, which drives the combine mechanism:

    - ``BIND_BLOCK_STRIDED`` → smem tree-halve at block scope.
    - ``BIND_SERIAL`` → no combine (each thread's partial is already
      the final value; legal but unused today).
    - Future: ``BIND_WARP_STRIDED`` → warp-shuffle (no smem)."""
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


def _emit_stage(stage: Stage, stage_buf: str, tid_expr) -> list[Stmt]:
    """Expand a ``Stage`` Stmt into smem decl + cooperative load + sync.

    Single-axis stage emits a direct ``StridedLoop(cache_axis,
    start=tid_expr, step=BLOCK_SIZE)``. Multi-axis stage flattens the
    cache axes into one synthetic linear axis of extent
    ``prod(extents)``; each thread owns every BLOCK_SIZE-th flat
    position, decoded back into per-axis coords for source Load and
    smem Write indices.

    Emits a leading ``Sync`` so iterations 2+ of an enclosing serial
    loop (typical chunked-K matmul) wait for the prior iteration's
    compute to finish reading smem before this iteration overwrites it.
    Iteration 1's leading Sync is harmless (no prior state)."""
    extents = tuple(int(ax.extent) for ax in stage.axes)
    if not extents:
        raise ValueError(f"Stage {stage.buf!r} has no cache axes")
    smem = Smem(name=stage_buf, extents=extents)
    load_name = f"_stage_{stage.buf}_v"

    if len(stage.axes) == 1:
        cache_axis = stage.axes[0]
        cooperative_load = StridedLoop(
            axis=cache_axis,
            start=tid_expr,
            step=BLOCK_SIZE,
            body=(
                Load(name=load_name, input=stage.buf, index=stage.index),
                Write(output=stage_buf, index=(Var(cache_axis.name),), value=load_name),
            ),
        )
        return [Sync(), smem, cooperative_load, Sync()]

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
        start=tid_expr,
        step=BLOCK_SIZE,
        body=(
            Load(name=load_name, input=stage.buf, index=source_index),
            Write(output=stage_buf, index=smem_index, value=load_name),
        ),
    )
    return [Sync(), smem, cooperative_load, Sync()]


def _flat_decode_sigma(cache_axes: tuple[Axis, ...], flat_name: str) -> Sigma:
    """Decode a flat row-major index back into per-axis coordinates."""
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
    outer = cache_axes[0]
    outer_stride = 1
    for ax in cache_axes[1:]:
        outer_stride *= int(ax.extent)
    if outer_stride == 1:
        mapping[outer.name] = flat
    else:
        mapping[outer.name] = flat / Literal(outer_stride, "int")
    return Sigma(mapping)


def _smem_coords_for_load(stage: Stage, load_index: tuple) -> tuple:
    """Build the smem read index for a redirected Load against ``stage``.

    Per cache axis, find its position in ``stage.index`` and extract the
    *local* (cache-coord) value from the matching ``load_index`` position:

    - **Pure-Var case** (softmax-style, ``stage.index[p] == Var(ax)``):
      smem coord = the load's expression verbatim (axis var possibly
      named differently per Load site).
    - **Affine case** (matmul-style, ``stage.index[p] == outer*F + ax``):
      strip ``outer → 0`` from the load's index expression."""
    cache_axis_names = {ax.name for ax in stage.axes}
    coords_by_axis: dict = {}
    for p, e in enumerate(stage.index):
        match = cache_axis_names & free_vars(e)
        if not match:
            continue
        if len(match) > 1:
            raise ValueError(f"Stage {stage.buf!r} index position {p} matches multiple cache axes: {match}")
        cache_name = next(iter(match))
        stage_free = free_vars(e)
        load_e = load_index[p]
        if stage_free == {cache_name}:
            coords_by_axis[cache_name] = load_e
        else:
            non_cache = stage_free - {cache_name}
            sigma = Sigma({nc: Literal(0, "int") for nc in non_cache})
            coords_by_axis[cache_name] = sigma.apply(load_e)
    return tuple(coords_by_axis[ax.name] for ax in stage.axes)


def _redirect_loads(stmt: Stmt, redirects: dict[str, tuple[str, Stage]]) -> Stmt:
    """Recursively rewrite ``Load(buf, ...)`` to ``Load(stage_buf,
    smem_index)`` for every ``buf`` in ``redirects``."""
    if isinstance(stmt, Load) and stmt.input in redirects:
        stage_buf, stage = redirects[stmt.input]
        return Load(name=stmt.name, input=stage_buf, index=_smem_coords_for_load(stage, stmt.index))
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
