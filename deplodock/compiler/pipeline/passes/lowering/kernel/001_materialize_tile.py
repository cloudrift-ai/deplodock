"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

The wrapper stays as ``Tile`` (shared with Tile IR via ``ir.stmt``);
only the body content changes — ``Stage`` becomes
``Smem`` + cooperative load, ``Combine`` becomes smem tree-halve,
``Loop`` / ``StridedLoop`` pass through. Two paths:

- **Non-cooperative** (no ``BIND_BLOCK`` axes): every BoundAxis is
  ``BIND_THREAD`` (pointwise / per-thread serial) — ``axes`` passed
  through, inner ``Loop``s pass through.

- **Cooperative** (one or more ``BIND_BLOCK`` axes): the Tile's THREAD
  axes are the cooperative thread set (synthesized by the strategy:
  ``cooperative-reduce`` adds a single ``t`` axis; ``blockify`` uses
  the per-block tile dims ``m_i`` / ``n_i``). Materialization passes
  ``Tile.axes`` through, computes a linear thread index ``tid_expr``
  from the THREAD axes, then walks the body:

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

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import KernelOp, Smem, Sync, TreeHalve
from deplodock.compiler.ir.stmt import Accum, Load, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import Combine, Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body: list[Stmt] = []
    for s in root.op.body:
        if isinstance(s, Tile):
            new_body.append(_materialize(s))
        else:
            new_body.append(s)

    root.op = KernelOp(body=tuple(new_body), name=root.op.name)
    return None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Tile) -> Stmt:
    """Materialize a Tile. ``Tile.axes`` carries the launch geometry
    (THREAD + optional BLOCK axes); strategies set this up — this pass
    commits no axis decisions of its own. The body is walked uniformly
    whether or not the Tile is cooperative: pointwise (no BLOCK axes)
    is the degenerate case where Stage / Combine / StridedLoop don't
    appear and no Accum nesting exists. The same walker handles both.

    Strategies that need single-thread Writes (e.g. cooperative scalar
    output) wrap them in ``Cond(thread_var == 0)`` themselves —
    materialization passes Writes through unchanged."""
    axes = blk.axes
    body = blk.body
    thread_axes = tuple(ba for ba in axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise ValueError("Tile must have at least one BIND_THREAD axis")
    tid_expr = _build_linear_tid(thread_axes)
    n_threads = 1
    for ba in thread_axes:
        n_threads *= int(ba.extent)

    rename: dict[str, str] = {}

    def transform(s: Stmt) -> Stmt:
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    pending_reduce: Accum | None = None

    for stmt in body:
        if isinstance(stmt, Stage):
            new_body.extend(_emit_stage(stmt, tid_expr, n_threads))
            pending_reduce = None
        elif isinstance(stmt, (Loop, StridedLoop)):
            new_body.append(_emit_loop(stmt, tid_expr, n_threads, transform))
            if stmt.is_reduce:
                # Combine matching needs the Accum at the immediate-body
                # level (single-loop reduce). For nested-reduce shapes
                # (K-chunked matmul: outer reduce wraps inner reduce, no
                # immediate Accum) there's no Combine to match anyway, so
                # ``None`` here is safe — a stray Combine would error in
                # the Combine branch as before.
                pending_reduce = next((a for a in stmt.body if isinstance(a, Accum)), None)
            else:
                pending_reduce = None
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce loop")
            if pending_reduce.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({pending_reduce.name!r})")
            new_body.extend(_emit_combine(pending_reduce, _single_thread_var(thread_axes), n_threads))
            rename[pending_reduce.name] = f"{pending_reduce.name}_b"
            pending_reduce = None
        else:
            new_body.append(transform(stmt))

    # Init scoping for accumulators is handled by the upstream
    # ``000_place_inits`` pass — explicit ``Init`` Stmts already sit at
    # the correct scope (Tile body head for reduce-only nesting, inside
    # a free Loop body when one wraps the Accum). Materialize is purely
    # mechanical from here.
    return Tile(axes=axes, body=tuple(new_body))


def _emit_loop(loop, tid_expr, n_threads, transform) -> Stmt:
    """Translate a body Loop or StridedLoop. Recurses so nested staging
    / loops / writes inside the body get the same uniform treatment.
    The wrapper type (Loop vs StridedLoop) is preserved — strategies
    decided the iteration shape; materialization just walks."""
    inner: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, Stage):
            inner.extend(_emit_stage(s, tid_expr, n_threads))
        elif isinstance(s, (Loop, StridedLoop)):
            inner.append(_emit_loop(s, tid_expr, n_threads, transform))
        else:
            inner.append(transform(s))
    return replace(loop, body=tuple(inner))


def _single_thread_var(thread_axes: tuple) -> str:
    """Combine + TreeHalve emit a single ``tid_var`` string. Only valid
    when there's exactly one THREAD axis — softmax-style cooperation
    (matmul has multi-axis THREAD set but doesn't emit Combine)."""
    if len(thread_axes) != 1:
        raise ValueError(f"Combine requires a single THREAD axis; got {len(thread_axes)}")
    return thread_axes[0].axis.name


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


def _emit_combine(accum: Accum, t: str, n_threads: int) -> list[Stmt]:
    """Emit the cross-thread combine: each thread writes its per-thread
    accumulator partial to smem indexed by ``t``, then a tree-halve
    reduces over ``t`` and broadcasts the final value via a load from
    ``smem[0]``."""
    smem_name = f"{accum.name}_smem"
    broadcast_name = f"{accum.name}_b"
    return [
        Smem(name=smem_name, extents=(n_threads,)),
        Write(output=smem_name, index=(Var(t),), value=accum.name),
        Sync(),
        TreeHalve(buf=smem_name, op=accum.op, length=n_threads, tid_var=t),
        Sync(),
        Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
    ]


# ---------------------------------------------------------------------------
# Stage expansion
# ---------------------------------------------------------------------------


def _emit_stage(stage: Stage, tid_expr, n_threads: int) -> list[Stmt]:
    """Expand a ``Stage`` Stmt into ``Smem`` decl + cooperative load + sync.

    The cooperative load reads a contiguous slab of ``stage.buf``
    starting at ``stage.origin`` (block-uniform) and spanning
    ``stage.axes`` extents. Each thread fetches one or more elements
    via a StridedLoop driven by ``tid_expr``: for a 1D slab, the loop
    iterates the cache axis directly; for N-D, it iterates a synthetic
    flat axis decoded into per-axis coords.

    Source index per source-buffer dim ``d`` =
    ``origin[d] + decoded[d]`` (the decoded slab coord, if any axis
    maps to ``d`` via ``slab_dims``); else just ``origin[d]``.

    Emits a leading ``Sync`` so iterations 2+ of an enclosing serial
    loop (chunked-K matmul) wait for the prior iteration's compute to
    finish reading smem before this iteration overwrites it. Iteration
    1's leading Sync is harmless (no prior state)."""
    if not stage.axes:
        raise ValueError(f"Stage {stage.name!r} has no cache axes")
    extents = tuple(int(ax.extent) for ax in stage.axes)

    # Iteration axis + per-cache-axis coord. Always synthesize a fresh
    # iter axis name so the cooperative-load ``for`` variable can't
    # collide with a same-named outer thread-decode variable (C++
    # init-expression scoping rules read the freshly-declared inner var,
    # giving an undefined initial value when the inner name shadows an
    # outer one). 1D cache: trivial map to the single cache axis. N-D:
    # row-major decode of the flat iter into per-axis coords.
    total = 1
    for e in extents:
        total *= e
    iter_axis = Axis(name=f"{stage.name}_flat", extent=total)
    if len(stage.axes) == 1:
        coord_for = {stage.axes[0].name: Var(iter_axis.name)}
    else:
        coord_for = _flat_decode(stage.axes, iter_axis.name)

    smem_index = tuple(coord_for[ax.name] for ax in stage.axes)
    if stage.source_index_template is not None:
        # Non-affine path (``/``, ``%`` from collapsed-reshape views).
        # Cache extent equals the raw axis extent (no F-scale baked in),
        # so substituting cache-axis Vars with their iter coords into
        # the original Load index gives the right source address per
        # cache position. Affine cases stay on the additive
        # ``origin + decoded`` path because their cache extent IS scaled
        # by F, so iter coord directly equals the cache-relative source
        # position — no need to re-multiply via F.
        from deplodock.compiler.ir.sigma import Sigma as _Sigma

        cache_sigma = _Sigma({ax.name: coord_for[ax.name] for ax in stage.axes})
        source_index = tuple(cache_sigma.apply(e) for e in stage.source_index_template)
    else:
        decoded_per_dim = {dim: coord_for[ax.name] for dim, ax in zip(stage.slab_dims, stage.axes, strict=True)}
        source_index = tuple(o if d not in decoded_per_dim else o + decoded_per_dim[d] for d, o in enumerate(stage.origin))

    load_name = f"{stage.name}_v"
    cooperative_load = StridedLoop(
        axis=iter_axis,
        start=tid_expr,
        step=Literal(n_threads, "int"),
        body=(
            Load(name=load_name, input=stage.buf, index=source_index),
            Write(output=stage.name, index=smem_index, value=load_name),
        ),
    )
    return [Sync(), Smem(name=stage.name, extents=extents), cooperative_load, Sync()]


def _flat_decode(cache_axes: tuple[Axis, ...], flat_name: str) -> dict:
    """Row-major decode of a flat index into per-axis coordinates.

    Innermost axis: ``flat % extent``. Middle axes:
    ``(flat / inner_stride) % extent``. Outermost axis: ``flat /
    outer_stride`` (mod is redundant — flat < total)."""
    flat = Var(flat_name)
    coords: dict = {}
    inner_stride = 1
    for ax in reversed(cache_axes):
        ext = int(ax.extent)
        coords[ax.name] = flat % Literal(ext, "int") if inner_stride == 1 else (flat / Literal(inner_stride, "int")) % Literal(ext, "int")
        inner_stride *= ext
    outer = cache_axes[0]
    outer_stride = inner_stride // int(outer.extent)
    coords[outer.name] = flat if outer_stride == 1 else flat / Literal(outer_stride, "int")
    return coords
