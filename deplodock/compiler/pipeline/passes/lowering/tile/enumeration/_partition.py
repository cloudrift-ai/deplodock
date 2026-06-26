"""``partition_reduce`` — the split-K combine block, rebuilt against the block-DAG IR.

R3 of ``plans/tile-ir-block-dag.md``. A cross-CTA split-K matmul partitions its
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
# enumerate); the staging / tensorize sentinels (``MMA=0`` scalar, ``STAGE=`` unstaged)
# stay so the PLACE-tier passes (120 / 130) skip on knob presence.
_COMBINE_KNOBS = {
    "MMA": "0",
    "STAGE": "",
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
    """The combine kernel's free-axis tower: ``GridTile(M_b, N_b) > ThreadTile(M_t,
    N_t)`` over ``16×16`` tiles with ceil-div grid extents (a boundary ``Cond`` guards
    a non-divisor M/N). Returns the four axes + the flattened ``(m_idx, n_idx)``."""
    m_blocks = -(-m_extent // _BM_RED)
    n_blocks = -(-n_extent // _BN_RED)
    M_b = Axis("M_b_red", to_dim(m_blocks))
    N_b = Axis("N_b_red", to_dim(n_blocks))
    M_t = Axis("M_t_red", to_dim(_BM_RED))
    N_t = Axis("N_t_red", to_dim(_BN_RED))
    m_idx = Var(M_b.name) * Literal(_BM_RED, "int") + Var(M_t.name)
    n_idx = Var(N_b.name) * Literal(_BN_RED, "int") + Var(N_t.name)
    return M_b, N_b, M_t, N_t, m_idx, n_idx


def _combine_block(
    *,
    name: str,
    m_extent: int,
    n_extent: int,
    inits: tuple[Stmt, ...],
    reduce_inner: tuple[Stmt, ...],
    s_axis: Axis,
    finalize: tuple[Assign, ...],
    written: str,
    out_name: str,
    dtype: DataType,
) -> TileGraph:
    """Assemble the combine ``Block`` from its per-carrier pieces.

    Shape (after ``assemble``)::

        GridTile(M_b, N_b)
          ThreadTile(M_t, N_t)
            <inits>
            SerialTile(K_s, reduce) { <reduce_inner: Load(s) + carrier fold> }
            Cond(in-bounds) { <finalize>; Write(out[m, n], written) }

    ``domain`` carries the free axes (THREAD then GRID); the ``K_s`` serial reduce
    rides ``compute`` (it is not a domain axis), exactly as the matmul's K tower does.
    """
    M_b, N_b, M_t, N_t, m_idx, n_idx = _grid_thread_axes(m_extent, n_extent)
    in_bounds = BinaryExpr(
        "&&",
        BinaryExpr("<", m_idx, Literal(m_extent, "int")),
        BinaryExpr("<", n_idx, Literal(n_extent, "int")),
    )
    write = Write(output=out_name, index=(m_idx, n_idx), value=written, value_dtype=dtype)
    guarded = Cond(cond=in_bounds, body=Body((*finalize, write)))
    serial = SerialTile(axis=s_axis, body=Body(reduce_inner), kind="plain", unroll=True)
    compute = Body((*inits, serial, guarded))
    # THREAD axes first, then GRID — ``assembly/_assemble._free_layers`` lays the tower
    # in that tier order (THREAD inner, GRID outer).
    domain = (M_t, N_t, M_b, N_b)
    binding = {
        M_t.name: Binding.THREAD,
        N_t.name: Binding.THREAD,
        M_b.name: Binding.GRID,
        N_b.name: Binding.GRID,
    }
    block = Block(name=name, domain=domain, compute=compute)
    buffers = {out_name: Buffer(name=out_name, shape=(to_dim(m_extent), to_dim(n_extent)), dtype=dtype, space=Space.GMEM)}
    return TileGraph(name=name, buffers=buffers, blocks=(block,), schedule=Schedule(binding=binding))


def additive_reduce_tilegraph(
    *, workspace_name: str, out_name: str, s_extent: int, m_extent: int, n_extent: int, dtype: DataType, name: str
) -> TileGraph:
    """The additive split-K combine: ``out[m, n] = Σ_s partial[s, m, n]`` via an
    ``Accum`` sum (bit-identical to the matmul's own ``+`` reduce). One thread per
    output cell serially folds the ``S`` partition slabs."""
    s_axis = Axis("K_s_red", to_dim(s_extent))
    _, _, _, _, m_idx, n_idx = _grid_thread_axes(m_extent, n_extent)
    reduce_inner = (
        Load(name="p", input=workspace_name, index=(Var(s_axis.name), m_idx, n_idx), dtype=dtype),
        Accum(name="acc", value="p", dtype=F32, axes=(s_axis.name,)),
    )
    return _combine_block(
        name=name,
        m_extent=m_extent,
        n_extent=n_extent,
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
    m_extent: int,
    n_extent: int,
    dtype: DataType,
    finalize: tuple[Assign, ...] = (),
    out_value: str | None = None,
    name: str = "monoid__reduce",
) -> TileGraph:
    """Carrier-general cross-partition combine — the monoid sibling of
    :func:`additive_reduce_tilegraph`.

    Each of the carrier's ``state`` components has its own workspace ``workspaces[i]``
    (shaped ``[S, M, N]``). One thread per output cell seeds its state from the
    op-identity (``init_ops[i]``: ``maximum`` → −inf, ``add`` → 0) and serially folds
    each ``S`` slice via the carrier's ``combine_states`` (the state-merges-state monoid
    op). An optional ``finalize`` (Assigns over the merged state) produces the single
    output value ``out_value`` (default the first state component), written to
    ``out_name``. The non-additive path a flash split-KV's online ``(m, l)`` LSE takes.
    """
    s_axis = Axis("K_s_red", to_dim(s_extent))
    _, _, _, _, m_idx, n_idx = _grid_thread_axes(m_extent, n_extent)
    others = tuple(f"o_{i}" for i in range(len(carrier.state)))
    loads: tuple[Stmt, ...] = tuple(
        Load(name=others[i], input=workspaces[i], index=(Var(s_axis.name), m_idx, n_idx), dtype=dtype) for i in range(len(workspaces))
    )
    fold = replace(carrier.as_state_merge(others), axes=(s_axis.name,))
    inits: tuple[Stmt, ...] = tuple(Init(name=st, op=init_ops[i], dtype=F32) for i, st in enumerate(carrier.state))
    written = out_value if out_value is not None else carrier.state[0]
    return _combine_block(
        name=name,
        m_extent=m_extent,
        n_extent=n_extent,
        inits=inits,
        reduce_inner=(*loads, fold),
        s_axis=s_axis,
        finalize=finalize,
        written=written,
        out_name=out_name,
        dtype=dtype,
    )
