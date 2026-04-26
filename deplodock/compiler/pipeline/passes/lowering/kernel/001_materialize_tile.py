"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

Reads each ``Tile`` in the TileOp body and emits the concrete hardware
shape — ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop`` — that ``render_kernelop`` consumes.

The Tile→Enclosure mapping is structural: both nodes carry
``axes: tuple[BoundAxis, ...]``. Two paths:

- **Non-cooperative** (no ``BIND_BLOCK`` axes): every BoundAxis is
  ``BIND_THREAD`` (pointwise / per-thread serial) → ``Enclosure`` with
  ``axes`` passed through. Inner ``Loop``s pass through.

- **Cooperative** (one or more ``BIND_BLOCK`` axes): the Tile's THREAD
  axes are the cooperative thread set (synthesized by the strategy:
  ``cooperative-reduce`` adds a single ``t`` axis; ``blockify`` uses
  the per-block tile dims ``m_i`` / ``n_i``). Materialization passes
  ``Tile.axes`` through to the Enclosure, computes a linear thread
  index ``tid_expr`` from the THREAD axes, then walks the body:

    * ``Stage`` → smem decl + cooperative load driven by ``tid_expr``
      (multi-axis stages flatten via row-major decode).
    * ``Loop`` / ``StridedLoop`` → passed through (recursive walk for
      Stage / Write handling inside).
    * ``Combine`` after a reduce loop → smem tree-halve + broadcast.
    * ``Write`` whose index references a THREAD axis is emitted
      unconditionally (each thread owns a unique output slot). Writes
      that don't reference any THREAD axis are guarded by ``tid==0`` so
      only one thread writes.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var, free_vars
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Load, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import (
    BLOCK_SIZE,
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
    """Walk an uncooperative-Tile body. Loops pass through; Combine /
    StridedLoop must not appear (no strategy emits them without setting
    ``block_axes``)."""
    if isinstance(s, Combine):
        raise ValueError("Combine not allowed in non-cooperative Tile (block_axes must be populated)")
    if isinstance(s, StridedLoop):
        raise ValueError("StridedLoop not allowed in non-cooperative Tile (no thread axis to drive it)")
    if isinstance(s, Loop):
        return Loop(axis=s.axis, body=tuple(_lower_uncooperative(c) for c in s.body))
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

    def transform(s: Stmt) -> Stmt:
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    pending_reduce: Accum | None = None
    # Axes whose Var distinguishes threads at the current scope. Starts as
    # the THREAD axes; extended by ``_emit_loop`` when descending into a
    # StridedLoop (each thread visits distinct iterations).
    distinguishing = set(thread_axis_names)

    for stmt in body:
        if isinstance(stmt, Stage):
            new_body.extend(_emit_stage(stmt, tid_expr))
            pending_reduce = None
        elif isinstance(stmt, (Loop, StridedLoop)):
            new_body.append(_emit_loop(stmt, tid_expr, thread_axes, distinguishing, transform))
            if _is_reduce(stmt):
                pending_reduce = next(a for a in stmt.body if isinstance(a, Accum))
            else:
                pending_reduce = None
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce loop")
            if pending_reduce.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({pending_reduce.name!r})")
            new_body.extend(_emit_combine(pending_reduce, _single_thread_var(thread_axes)))
            rename[pending_reduce.name] = f"{pending_reduce.name}_b"
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
    # (THREAD + BLOCK only).
    return Enclosure(axes=axes, body=tuple(new_body))


def _emit_loop(
    loop,
    tid_expr,
    thread_axes: tuple,
    distinguishing: set,
    transform,
) -> Stmt:
    """Translate a body Loop or StridedLoop. Recurses so nested staging /
    loops / writes inside the body get the same uniform treatment.

    A ``StridedLoop`` adds its axis to ``distinguishing`` for nested-scope
    Writes — threads visit distinct iterations of the strided axis, so a
    Write whose index references it gets distinct output positions per
    thread (no guard needed)."""
    inner_distinguishing = distinguishing | ({loop.axis.name} if isinstance(loop, StridedLoop) else set())
    inner: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, Stage):
            inner.extend(_emit_stage(s, tid_expr))
        elif isinstance(s, (Loop, StridedLoop)):
            inner.append(_emit_loop(s, tid_expr, thread_axes, inner_distinguishing, transform))
        elif isinstance(s, Write):
            inner.append(_emit_write(s, inner_distinguishing, thread_axes, transform))
        else:
            inner.append(transform(s))
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=tuple(inner))
    return Loop(axis=loop.axis, body=tuple(inner))


def _emit_write(
    write: Write,
    distinguishing: set,
    thread_axes: tuple,
    transform,
) -> Stmt:
    """Emit a Write — unconditional if the index references any
    *distinguishing* axis (THREAD axes plus any enclosing StridedLoop
    axes); otherwise wrapped in ``Cond(all THREAD axes == 0)`` so only
    one thread writes a shared output slot."""
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


def _emit_combine(accum: Accum, t: str) -> list[Stmt]:
    """Emit the cross-thread combine: each thread writes its per-thread
    accumulator partial to smem indexed by ``t``, then a tree-halve
    reduces over ``t`` and broadcasts the final value via a load from
    ``smem[0]``."""
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


def _is_reduce(loop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


# ---------------------------------------------------------------------------
# Stage expansion
# ---------------------------------------------------------------------------


def _emit_stage(stage: Stage, tid_expr) -> list[Stmt]:
    """Expand a ``Stage`` Stmt into ``Smem`` decl + cooperative load + sync.

    The strategy that emits the Stage is responsible for rewriting
    body Loads of ``stage.buf`` to target ``stage.name`` with
    cache-local indices — materialization just emits the smem buffer
    and the cooperative load that fills it.

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
        raise ValueError(f"Stage {stage.name!r} has no cache axes")
    smem = Smem(name=stage.name, extents=extents)
    load_name = f"{stage.name}_v"

    if len(stage.axes) == 1:
        cache_axis = stage.axes[0]
        cooperative_load = StridedLoop(
            axis=cache_axis,
            start=tid_expr,
            step=BLOCK_SIZE,
            body=(
                Load(name=load_name, input=stage.buf, index=stage.index),
                Write(output=stage.name, index=(Var(cache_axis.name),), value=load_name),
            ),
        )
        return [Sync(), smem, cooperative_load, Sync()]

    total = 1
    for e in extents:
        total *= e
    flat_name = f"{stage.name}_flat"
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
            Write(output=stage.name, index=smem_index, value=load_name),
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
