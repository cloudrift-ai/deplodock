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
tile, one thread per output cell, the partition axis ``K_s`` a serial reduce loop. The
additive matmul case folds with a bit-identical ``Accum`` sum
(:func:`additive_reduce_tilegraph`); a non-additive carrier (flash split-KV's online
``(m, l)`` monoid) folds via the carrier's ``combine_states``
(:func:`monoid_reduce_tilegraph`). This mirrors the deleted
``017_atomic_free_splitk``'s ``_build_reduce_tileop`` / ``build_monoid_reduce_tileop``,
now emitting the block-DAG IR instead of a pre-tiled ``TileOp`` tower.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dim import to_dim
from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Monoid, Stmt, Write
from deplodock.compiler.ir.tile.ir import Binding, Block, Buffer, Schedule, SerialTile, Space, TileGraph, TileGraphOp

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
        in_bounds = BinaryExpr(
            "&&", BinaryExpr("<", m_idx, Literal(m_extent, "int")), BinaryExpr("<", n_idx, Literal(n_extent, "int"))
        )
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


def additive_reduce_tilegraph(
    *, workspace_name: str, out_name: str, s_extent: int, out_shape: tuple[int, ...], dtype: DataType, name: str
) -> TileGraph:
    """The additive split-K / split-reduce combine: ``out[d...] = Σ_s partial[s, d...]``
    via an ``Accum`` sum (bit-identical to the matmul's own ``+`` reduce). One thread per
    output cell serially folds the ``S`` partition slabs. ``out_shape`` is the (static)
    output extent per dim — ``(M, N)`` for the matmul, ``(M,)`` for a plain ``sum(dim)``."""
    s_axis = Axis("K_s_red", to_dim(s_extent))
    _, _, out_index, _ = _out_axes(out_shape)
    reduce_inner = (
        Load(name="p", input=workspace_name, index=(Var(s_axis.name), *out_index), dtype=dtype),
        Accum(name="acc", value="p", dtype=F32, axes=(s_axis.name,)),
    )
    return _combine_block(
        name=name,
        out_shape=out_shape,
        inits=(),
        reduce_inner=reduce_inner,
        s_axis=s_axis,
        finalize=(),
        written="acc",
        out_name=out_name,
        dtype=dtype,
    )


def monoid_reduce_tilegraph(
    *,
    carrier: Monoid,
    init_ops: tuple[ElementwiseImpl, ...],
    workspaces: tuple[str, ...],
    out_name: str,
    s_extent: int,
    out_shape: tuple[int, ...],
    dtype: DataType,
    finalize: tuple[Assign, ...] = (),
    out_value: str | None = None,
    name: str = "monoid__reduce",
) -> TileGraph:
    """Carrier-general cross-partition combine — the monoid sibling of
    :func:`additive_reduce_tilegraph`.

    Each of the carrier's ``state`` components has its own workspace ``workspaces[i]``
    (shaped ``[S, *out_shape]``). One thread per output cell seeds its state from the
    op-identity (``init_ops[i]``: ``maximum`` → −inf, ``add`` → 0) and serially folds
    each ``S`` slice via the carrier's ``combine_states`` (the state-merges-state monoid
    op). An optional ``finalize`` (Assigns over the merged state) produces the single
    output value ``out_value`` (default the first state component), written to
    ``out_name``. The non-additive path a flash split-KV's online ``(m, l)`` LSE takes.
    """
    s_axis = Axis("K_s_red", to_dim(s_extent))
    _, _, out_index, _ = _out_axes(out_shape)
    others = tuple(f"o_{i}" for i in range(len(carrier.state)))
    loads: tuple[Stmt, ...] = tuple(
        Load(name=others[i], input=workspaces[i], index=(Var(s_axis.name), *out_index), dtype=dtype) for i in range(len(workspaces))
    )
    fold = replace(carrier.as_state_merge(others), axes=(s_axis.name,))
    inits: tuple[Stmt, ...] = tuple(Init(name=st, op=init_ops[i], dtype=F32) for i, st in enumerate(carrier.state))
    written = out_value if out_value is not None else carrier.state[0]
    return _combine_block(
        name=name,
        out_shape=out_shape,
        inits=inits,
        reduce_inner=(*loads, fold),
        s_axis=s_axis,
        finalize=finalize,
        written=written,
        out_name=out_name,
        dtype=dtype,
    )


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
    """Carrier-generic cross-partition (split-K / split-KV) **deferred finalize** — ONE combine
    kernel for ANY associative carrier, dispatched on its state. The schedule never changes (one
    CTA per ``16×16`` output tile, a serial fold over the ``S`` partitions); only the carrier's
    combine does. This is the algebraic thesis made concrete: a regular sum, online softmax
    ``(m, d)``, Welford ``(n, μ, M₂)``, and flash attention ``(m, d, o)`` are the SAME reduction,
    differing only in carrier state — so the deferred finalize is one entry point, not a flash
    special case.

    - an **additive 1-component ``Accum``** with no finalize (the matmul split-K, ``state =
      (acc,)``) → :func:`additive_reduce_tilegraph` (the bit-identical ``+`` fold — the trivial
      carrier).
    - a tuple **``Monoid``** (online-softmax / flash) or any multi-component carrier →
      :func:`monoid_reduce_tilegraph`, seeding each state component from ``init_ops`` (the
      carrier's per-component identity) and folding via the carrier's ``combine_states``, with the
      op's ``finalize`` (e.g. flash's ``o / d``) read off at the end.

    The producer (the cross-CTA split) writes one workspace ``partial_i[S, M, N]`` per state
    component (``workspaces`` aligned to the carrier's state); an additive carrier has a single
    component, so ``workspaces`` is length 1."""
    if isinstance(carrier, Accum) and carrier.op.name == "add" and len(workspaces) == 1 and not finalize:
        return additive_reduce_tilegraph(
            workspace_name=workspaces[0],
            out_name=out_name,
            s_extent=s_extent,
            out_shape=out_shape,
            dtype=dtype,
            name=name,
        )
    if not isinstance(carrier, Monoid):
        raise ValueError(
            f"deferred_combine_tilegraph: a non-additive {type(carrier).__name__} carrier needs the "
            "Monoid (state, combine_states) form for the cross-partition fold"
        )
    return monoid_reduce_tilegraph(
        carrier=carrier,
        init_ops=init_ops,
        workspaces=workspaces,
        out_name=out_name,
        s_extent=s_extent,
        out_shape=out_shape,
        dtype=dtype,
        finalize=finalize,
        out_value=out_value,
        name=name,
    )
