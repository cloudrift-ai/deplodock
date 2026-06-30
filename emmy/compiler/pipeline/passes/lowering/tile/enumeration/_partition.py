"""``partition_reduce`` — the split-K combine block, rebuilt against the block-DAG IR.

A cross-CTA split-K matmul partitions its
contraction axis ``K`` across ``SPLITK`` CTAs (the ``K_s`` GRID axis the reduce-decomp
body move already binds); the per-partition partials must then be combined. Two
realizations:

- **atomic** (the default, already working): each CTA ``atomicAdd``s its partial into
  the output — ``K_s`` stays out of the Write index, so ``escape_analysis`` emits the
  atomic. Nothing here.
- **atomic-free** (this module): the matmul writes its partial into a workspace
  ``partial[K_s, M, N]`` (``K_s`` now in the Write index ⇒ a plain store), and a
  sibling **combine kernel** folds the ``K_s`` axis into the original output. The
  combine is a separate launch group — a second kernel spliced beside the matmul.

The combine kernel is built here as a **fully-tiled** single-``Block`` ``TileGraph``
(``assembly/010_assemble`` materializes it directly): one CTA per ``16×16`` output
tile, one thread per output cell, the partition axis ``K_s`` a serial reduce loop. ONE
builder — :func:`deferred_combine_tilegraph` — serves every carrier: the additive matmul
case folds with a bit-identical ``Accum`` sum, a non-additive carrier (flash split-KV's
online ``(m, l)`` monoid) folds via the carrier's ``combine_states``; the carrier dispatch
is the carrier's ``combine_states`` (an ``Accum`` folded as a degenerate ``Monoid`` via
``Accum.as_monoid``), realized by ``render_merge_program`` — no cross-CTA "combiner" class. This mirrors the deleted
``017_atomic_free_splitk``'s ``_build_reduce_tileop`` / ``build_monoid_reduce_tileop``,
now emitting the block-DAG IR instead of a pre-tiled ``TileOp`` tower.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import to_dim
from emmy.compiler.dtype import F32, DataType
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Monoid, Stmt, Write
from emmy.compiler.ir.tile.ir import Binding, Block, Buffer, Schedule, SerialTile, Space, TileGraph, TileGraphOp

# Fixed combine schedule — bandwidth-bound at any realistic shape, so per-shape
# autotune would only re-discover it (the deleted 017's _BM_RED / _BN_RED).
_BM_RED = 16
_BN_RED = 16

# The combine kernel is fixed-schedule (no fork): it is a fully-tiled, dag-less
# ``TileGraphOp`` (``dag=None``), so the geometry passes (090 / 100 / 110) skip it on the
# dag-None guard and ``assembly/010_assemble`` materializes the already-tiled block
# directly. It carries no free-axis ``SPLIT@<axis>`` / ``REDUCE@<axis>`` knob (nothing to
# enumerate); the tensorize sentinel (``ATOM@out=scalar``) stays so 020/050 skip. It
# carries no ``PLACE@<edge>``: 120 skips it on the dag-None guard and 130 on the absence of
# any untransported smem edge (nothing staged), so the placement passes pass it through.
_COMBINE_KNOBS = {
    "ATOM@out": "scalar",
}


def reduce_tilegraphop(tg: TileGraph, *, extra_knobs: dict | None = None) -> TileGraphOp:
    """Wrap a fully-tiled combine ``TileGraph`` into the ``TileGraphOp`` the assembly
    pass consumes. ``algebra=MAP`` + the stamped :data:`_COMBINE_KNOBS` make every
    enumeration fork ``RuleSkipped`` (the kernel is fixed-schedule, not searched)."""
    return TileGraphOp(
        name=tg.name,
        tilegraph=tg,
        algebra=AlgebraKind.MAP,
        target_names=frozenset(),
        knobs={**_COMBINE_KNOBS, **(extra_knobs or {})},
    )


def _grid_thread_axes(m_extent: int, n_extent: int) -> tuple[Axis, Axis, Axis, Axis, BinaryExpr, BinaryExpr]:
    """The 2-D combine kernel's free-axis tower: ``GridTile(M_b, N_b) > ThreadTile(M_t,
    N_t)`` over ``16×16`` tiles with ceil-div grid extents (a boundary ``Cond`` guards
    a non-divisor M/N). Returns the four axes + the flattened ``(m_idx, n_idx)``. The
    matmul split-K combine is the rank-2 case — kept byte-identical (axis names + order)
    by :func:`_out_axes`; a general reduction's output rank routes through the rank-N path."""
    m_blocks = -(-m_extent // _BM_RED)
    n_blocks = -(-n_extent // _BN_RED)
    M_b = Axis("M_b_red", to_dim(m_blocks))
    N_b = Axis("N_b_red", to_dim(n_blocks))
    M_t = Axis("M_t_red", to_dim(_BM_RED))
    N_t = Axis("N_t_red", to_dim(_BN_RED))
    m_idx = Var(M_b.name) * Literal(_BM_RED, "int") + Var(M_t.name)
    n_idx = Var(N_b.name) * Literal(_BN_RED, "int") + Var(N_t.name)
    return M_b, N_b, M_t, N_t, m_idx, n_idx


def _out_axes(out_shape: tuple[int, ...]) -> tuple[tuple[Axis, ...], dict, tuple, BinaryExpr]:
    """The combine kernel's free-axis tower for an **arbitrary output rank** — the
    carrier-generic geometry generalization (a reduction's output is N-D: ``(M,)`` for a
    plain ``x.sum(dim=-1)``, ``(M, N)`` for the matmul split-K, ``(B, M, N)`` for a
    batched reduce). Returns ``(domain, binding, out_index, in_bounds)``: the THREAD-inner
    / GRID-outer domain, the per-axis binding, the per-dim output index exprs, and the
    boundary ``Cond`` guard.

    The **innermost up-to-two** output axes get a ``16``-wide ``ThreadTile`` + ceil-div
    ``GridTile`` (so one CTA owns a ``16×16`` output tile, one thread per cell — the
    matmul's geometry); every leading axis is a pure ``GridTile`` (one block per
    coordinate). The ``K_s`` partition stays the serial fold (added by the caller, not a
    domain axis). The **rank-2** case delegates to :func:`_grid_thread_axes` so the matmul
    split-K combine stays byte-identical (same axis names + domain order)."""
    if len(out_shape) == 2:
        m_extent, n_extent = out_shape
        M_b, N_b, M_t, N_t, m_idx, n_idx = _grid_thread_axes(m_extent, n_extent)
        domain = (M_t, N_t, M_b, N_b)
        binding = {M_t.name: Binding.THREAD, N_t.name: Binding.THREAD, M_b.name: Binding.GRID, N_b.name: Binding.GRID}
        in_bounds = BinaryExpr("&&", BinaryExpr("<", m_idx, Literal(m_extent, "int")), BinaryExpr("<", n_idx, Literal(n_extent, "int")))
        return domain, binding, (m_idx, n_idx), in_bounds

    r = len(out_shape)
    tiled = set(range(max(0, r - 2), r))  # the innermost up-to-two axes get a 16-wide thread tile
    thread_axes: list[Axis] = []
    grid_axes: list[Axis] = []
    binding: dict = {}
    index: list = []
    guards: list = []
    for i, ext in enumerate(out_shape):
        if i in tiled:
            a_b = Axis(f"d{i}_b_red", to_dim(-(-ext // _BN_RED)))
            a_t = Axis(f"d{i}_t_red", to_dim(_BN_RED))
            idx = Var(a_b.name) * Literal(_BN_RED, "int") + Var(a_t.name)
            thread_axes.append(a_t)
            grid_axes.append(a_b)
            binding[a_t.name] = Binding.THREAD
            binding[a_b.name] = Binding.GRID
            index.append(idx)
            guards.append(BinaryExpr("<", idx, Literal(ext, "int")))
        else:
            a = Axis(f"d{i}_red", to_dim(ext))
            grid_axes.append(a)
            binding[a.name] = Binding.GRID
            index.append(Var(a.name))
    domain = (*thread_axes, *grid_axes)  # THREAD inner, GRID outer (``_free_layers`` tier order)
    in_bounds = guards[0]
    for g in guards[1:]:
        in_bounds = BinaryExpr("&&", in_bounds, g)
    return domain, binding, tuple(index), in_bounds


def _combine_block(
    *,
    name: str,
    out_shape: tuple[int, ...],
    inits: tuple[Stmt, ...],
    reduce_inner: tuple[Stmt, ...],
    s_axis: Axis,
    finalize: tuple[Assign, ...],
    written: str,
    out_name: str,
    dtype: DataType,
) -> TileGraph:
    """Assemble the combine ``Block`` from its per-carrier pieces, for an arbitrary
    output rank (the geometry comes from :func:`_out_axes`).

    Shape (after ``assemble``), illustrated for a 2-D output::

        GridTile(M_b, N_b)
          ThreadTile(M_t, N_t)
            <inits>
            SerialTile(K_s, reduce) { <reduce_inner: Load(s) + carrier fold> }
            Cond(in-bounds) { <finalize>; Write(out[d...], written) }

    ``domain`` carries the free output axes (THREAD then GRID); the ``K_s`` serial reduce
    rides ``compute`` (it is not a domain axis), exactly as the matmul's K tower does. The
    ``reduce_inner`` Loads index ``workspace[s, *out_index]``, so the caller must build them
    from the SAME :func:`_out_axes` ``out_index`` this block writes.
    """
    domain, binding, out_index, in_bounds = _out_axes(out_shape)
    write = Write(output=out_name, index=out_index, value=written, value_dtype=dtype)
    guarded = Cond(cond=in_bounds, body=Body((*finalize, write)))
    serial = SerialTile(axis=s_axis, body=Body(reduce_inner), kind="plain", unroll=True)
    compute = Body((*inits, serial, guarded))
    block = Block(name=name, domain=domain, compute=compute)
    buffers = {out_name: Buffer(name=out_name, shape=tuple(to_dim(e) for e in out_shape), dtype=dtype, space=Space.GMEM)}
    return TileGraph(name=name, buffers=buffers, blocks=(block,), schedule=Schedule(binding=binding))


def deferred_combine_tilegraph(
    carrier,
    *,
    workspaces: tuple[str, ...],
    out_name: str,
    s_extent: int,
    out_shape: tuple[int, ...],
    dtype: DataType,
    name: str,
    init_ops: tuple[ElementwiseImpl, ...] = (),
    finalize: tuple[Assign, ...] = (),
    out_value: str | None = None,
) -> TileGraph:
    """Carrier-generic cross-partition (split-K / split-KV) combine — **ONE** combine kernel for
    ANY associative carrier. The schedule + scaffolding (``_combine_block`` / ``_out_axes``: one CTA
    per ``16×16`` output tile, a serial fold over the ``S`` partitions) never change; only the
    carrier fold does. The fold is the **cross-CTA distribution** of the carrier's combine — pure
    geometry (load each partition's state from the workspace(s), then the carrier's ``combine_states``
    via ``as_state_merge``); the combine ALGEBRA is realized by ``Monoid.render`` /
    ``render_merge_program`` (the SAME realizer the scalar combine uses), so there is no separate
    cross-CTA "combiner". A regular sum, online softmax ``(m, d)``, Welford ``(n, μ, M₂)``, and flash
    attention ``(m, d, o)`` are the SAME reduction, differing only in carrier state.

    **ONE carrier path** (no additive special-case): an additive ``Accum`` lowers as the degenerate
    1-component ``Monoid`` it is (``Accum.as_monoid``), folded exactly like a flash ``(m, l, O)``. The
    producer writes one workspace per state component (``workspaces`` aligned to the carrier's state;
    an additive carrier is length 1; or one **packed** ``partial[S, n_state, *out]`` for a multi-state
    carrier a single producer kernel emits). An optional ``finalize`` (e.g. flash's ``O / l``) yields
    ``out_value`` (default ``state[0]``)."""
    if isinstance(carrier, Accum):
        init_ops, carrier = (carrier.op,), carrier.as_monoid()  # the degenerate 1-component monoid — one path
    elif not isinstance(carrier, Monoid):
        raise ValueError(
            f"deferred_combine_tilegraph: a {type(carrier).__name__} carrier needs the Accum (additive) "
            "or Monoid (state, combine_states) form for the cross-partition fold"
        )
    s_axis = Axis("K_s_red", to_dim(s_extent))
    _, _, out_index, _ = _out_axes(out_shape)
    # The cross-CTA distribution: load each partition's state, fold via the carrier's combine_states.
    others = tuple(f"o_{i}" for i in range(len(carrier.state)))
    packed = len(workspaces) == 1 and len(carrier.state) > 1
    loads: tuple[Stmt, ...] = tuple(
        Load(
            name=others[i],
            input=workspaces[0] if packed else workspaces[i],
            index=(Var(s_axis.name), Literal(i, "int"), *out_index) if packed else (Var(s_axis.name), *out_index),
            dtype=dtype,
        )
        for i in range(len(carrier.state))
    )
    fold = replace(carrier.as_state_merge(others), axes=(s_axis.name,))
    inits: tuple[Stmt, ...] = tuple(Init(name=st, op=init_ops[i], dtype=F32) for i, st in enumerate(carrier.state))
    written = out_value if out_value is not None else carrier.state[0]
    reduce_inner = (*loads, fold)
    return _combine_block(
        name=name,
        out_shape=out_shape,
        inits=inits,
        reduce_inner=reduce_inner,
        s_axis=s_axis,
        finalize=finalize,
        written=written,
        out_name=out_name,
        dtype=dtype,
    )
