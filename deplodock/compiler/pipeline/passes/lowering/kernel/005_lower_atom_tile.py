"""Lower AtomTile to MMA fragment Stmts — M5 of
``plans/mma-fragment-factorization.md`` plus M5 of
``plans/mma-smem-staging.md``.

Runs *before* ``010_split_register_axes`` in the kernel chain. Pattern-
matches the warp-tier matmul shape the partition planner emits
(``RegisterTile > AtomTile > matmul-cell-body`` for the unstaged path,
``RegisterTile > AtomTile > StageBundle > matmul-cell-body`` once
smem-staging is on), replaces the cell body with an ``MmaFragment`` +
``MmaFill`` + per-K_i ``MmaLoad`` / ``MmaSync`` chain plus a final
``MmaStore``, and strips the AtomTile wrapper (its axes encoded the
cell shape, which is now baked into the Mma* Stmts' ``shape`` field).
When a ``StageBundle`` lives inside the AtomTile body, the lowered
Mma chain is re-wrapped in the same bundle so the smem allocation and
cooperative producer survive.

For staged operands, ``MmaLoad.src_index`` is rebuilt to match the
slab's cache-axis rank: one coord per cache axis, each
``Var(ax) * block[i]``. ``MmaLoad.ldm`` is computed from the inner
source-dim's slab extent product so multi-axis-per-source-dim slabs
(N-side FN-fan-out is the common case) feed the right row stride into
``wmma::load_matrix_sync`` — the auto path would have picked the
trailing alloc extent, which collapses to a per-cell width when N
splits across warp + register cache axes.

The ``RegisterTile`` wrapper is left in place — ``010_split_register_axes``
runs next and replicates the Mma* chain per (M_r, N_r) cell. Each
``Mma*`` Stmt's ``rewrite.register`` handler threads the per-cell
``rename`` callback through the fragment SSA names, so the replicator
produces ``c_frag_<i>_<j>`` / ``a_frag_<i>_<j>`` / ``b_frag_<i>_<j>``
per cell without this pass having to know about FM / FN.

Eligibility: ``op.knobs["ATOM_KIND"]`` set (only warp-tier matmul rows
carry this knob — the scalar planner branch leaves it unset and this
pass skips). Idempotence: after this pass the AtomTile is gone, so on
a second visit the pattern doesn't match and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import MmaFill, MmaFragment, MmaLoad, MmaStore, MmaSync
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, AtomTile, SerialTile, Source, StageBundle, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._atom import AtomSpec, atom_spec
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    parallel_tile_of,
    replace_parallel_tile_body,
    single_tile,
)

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    atom_kind = op.knobs.get("ATOM_KIND")
    if not atom_kind:
        raise RuleSkipped("not an MMA TileOp (no ATOM_KIND knob)")
    spec = atom_spec(atom_kind)
    body = op.body
    idx, outer = single_tile(body)
    tt = parallel_tile_of(outer)

    # The c-fragment dtype tracks the output buffer's dtype so the emitted
    # ``wmma::store_matrix_sync(out_ptr, c_frag, ...)`` has a matching
    # overload. WMMA supports ``__half`` and ``float`` accumulators on
    # sm_70+; for F16 outputs we use F16 accumulation (slightly less
    # precision than F32 acc + downconvert via smem scratch, but
    # significantly simpler and fast). F32 outputs keep the spec default
    # (F32 acc).
    c_dtype_override = _resolve_c_dtype(root, spec.operand_dtypes["c"])

    lowered, found = _lower_atom_tiles(tt.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources={})
    if not found:
        # Could happen on a second visit (AtomTile already consumed).
        raise RuleSkipped("no AtomTile in body — already lowered")

    rebuilt = replace_parallel_tile_body(outer, lowered)
    return TileOp(body=body[:idx] + (rebuilt,) + body[idx + 1 :], name=op.name, knobs=op.knobs)


def _resolve_c_dtype(root: Node, spec_c_dtype: DataType) -> DataType:
    """Pick the c-fragment dtype for the WMMA accumulator. WMMA's
    ``store_matrix_sync`` requires the destination pointer's element type
    to match the fragment's element type — so if the output buffer is
    ``__half*``, the C fragment must be ``__half``-accumulator (not
    ``float``-accumulator). Otherwise NVCC fails to find a matching overload.

    Strategy: read the matmul TileOp's output tensor dtype directly.
    F16 output → F16 c-frag (sm_70+ supports both half and float WMMA
    accumulators). F32 output → F32 c-frag (canonical higher-precision).
    Other dtypes fall through to the registry's default.

    Tradeoff: F16 accumulator has ~3-4 ulp drift per accumulation step vs
    F32. For real-world matmuls (small K, fp16 operands at small dynamic
    range) it stays within fp16 tolerance. A future plan can add the
    "F32 acc + smem-scratch fp32→fp16 cooperative downconvert" path for
    kernels needing the precision.
    """
    out_dtype = root.output.dtype
    if out_dtype == F16:
        return F16
    if out_dtype == F32:
        return F32
    return spec_c_dtype


def _lower_atom_tiles(
    body: Body,
    *,
    spec: AtomSpec,
    c_dtype_override: DataType,
    smem_sources: dict[str, Source],
) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile`` encountered, rewrite its
    interior matmul-shape body into an Mma* fragment chain and strip
    the AtomTile wrapper. Recurses into non-AtomTile block stmts so a
    deep-nested AtomTile (under RegisterTile / SerialTile / Cond / ...)
    is reached. Returns ``(new_body, found_any)``.

    ``smem_sources`` is a flat ``smem_name → Source`` map of Sources
    in-scope from enclosing ``StageBundle``s. When ``_atom_body_to_mma``
    encounters a Load reading from a staged smem buffer, it rebuilds
    ``src_index`` via the Source's ``AffineAddressing.source_index`` so
    the MMA fragment lands on the correct (block-scaled) slab offset.
    M2-M5 of ``plans/mma-smem-staging.md``.
    """
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            new_stmts = _atom_body_to_mma(s.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources=smem_sources)
            out.extend(new_stmts)
            found = True
            continue
        if isinstance(s, StageBundle):
            # Register every Source in the bundle so child AtomTile
            # Loads reading from this slab can resolve the addressing.
            # Bundles can nest (cooperative-K + outer staging); we copy
            # the dict so a child bundle's additions don't leak out to
            # sibling subtrees.
            inner_sources = dict(smem_sources)
            for stage in s.stages:
                for src in stage.sources:
                    inner_sources[src.name] = src
            new_bundle_body, body_found = _lower_atom_tiles(
                s.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources=inner_sources
            )
            # ``StageBundle.with_bodies`` expects ``(stages_body, body)`` —
            # stages stay byte-clean (no AtomTile lives inside producer
            # Sources), we only rebuild the consumer body.
            new_stages_body = Body(s.stages)
            out.append(s.with_bodies((new_stages_body, new_bundle_body)))
            found = found or body_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_lowered = False
            for sub in s.nested():
                new_sub, sub_found = _lower_atom_tiles(sub, spec=spec, c_dtype_override=c_dtype_override, smem_sources=smem_sources)
                new_bodies.append(new_sub)
                any_lowered = any_lowered or sub_found
            found = found or any_lowered
            out.append(s.with_bodies(tuple(new_bodies)))
            continue
        out.append(s)
    return Body(out), found


def _atom_body_to_mma(
    body: Body,
    *,
    spec: AtomSpec,
    c_dtype_override: DataType,
    smem_sources: dict[str, Source],
) -> tuple[Stmt, ...]:
    """Pattern-match the AtomTile's interior matmul body and rewrite it
    to an Mma* fragment chain.

    Expected shape (post-partition_loops, MMA path with no smem staging):

        [Init(acc)?]                              # placed by 020_place_inits — absent here, this runs before
        SerialTile(K_o, "serial_outer"):          # may be size-1-filtered away by _wrap_tower
          SerialTile(K_i, "stage_inner", reduce):
            Load a_v <- A[<m_expr>, <k_expr>]
            Load b_v <- B[<k_expr>, <n_expr>]
            Assign p = a_v * b_v
            Accum acc <- p
        Write C[<m_expr>, <n_expr>] = acc

    Emits:

        MmaFragment(c_frag, "c", spec.shape, c_dtype)
        MmaFragment(a_frag, "a", spec.shape, a_dtype)
        MmaFragment(b_frag, "b", spec.shape, b_dtype)
        MmaFill(c_frag, 0.0)
        SerialTile(K_o, "serial_outer"):
          SerialTile(K_i, "stage_inner"):              # no longer reduce
            MmaLoad(a_frag, A, [<m_expr>, <k_expr>])   # ldm=0 → render-time lookup
            MmaLoad(b_frag, B, [<k_expr>, <n_expr>])
            MmaSync(c_frag, a_frag, b_frag)
        MmaStore(C, [<m_expr>, <n_expr>], c_frag)
    """
    # Find the Write + the inner reduce SerialTile (K_i). K_o and K_i may
    # both be size-1 filtered by ``_wrap_tower``, in which case the matmul
    # body sits *directly* inside the AtomTile (no SerialTile wrapper).
    # M5: a ``StageBundle`` between the AtomTile and the K-loop is the
    # MMA-staged shape — look inside its body to find the K-loop, and
    # capture the bundle so the lowered Mma chain re-wraps in it.
    write_stmt: Write | None = None
    reduce_st: SerialTile | None = None
    outer_st: SerialTile | None = None
    enclosing_bundle: StageBundle | None = None
    flat_loads: list[Load] = []
    flat_accum: Accum | None = None
    # Local copy so a bundle nested inside the AtomTile's body contributes
    # its Sources to the lookup `_mma_src_index` consults below — without
    # leaking back to siblings.
    bundle_sources = dict(smem_sources)

    def _walk(stmt) -> None:
        nonlocal write_stmt, reduce_st, outer_st, flat_accum, enclosing_bundle
        if isinstance(stmt, Write):
            write_stmt = stmt
            return
        if isinstance(stmt, StageBundle):
            enclosing_bundle = stmt
            for stage in stmt.stages:
                for src in stage.sources:
                    bundle_sources[src.name] = src
            for inner in stmt.body:
                _walk(inner)
            return
        if isinstance(stmt, SerialTile):
            # Direct K_o > K_i shape (no staging between): record outer +
            # reduce in one shot.
            if any(isinstance(c, SerialTile) and c.is_reduce for c in stmt.body):
                outer_st = stmt
                reduce_st = next(c for c in stmt.body if isinstance(c, SerialTile) and c.is_reduce)
                return
            if stmt.is_reduce:
                reduce_st = stmt
                return
            # Non-reduce serial: K_o wrapping a StageBundle that wraps
            # the K_i SerialTile (the M5-staged shape). Recurse through.
            outer_st = stmt
            for inner in stmt.body:
                _walk(inner)
            return
        if isinstance(stmt, Load):
            flat_loads.append(stmt)
            return
        if isinstance(stmt, Accum):
            flat_accum = stmt
            return

    for s in body:
        _walk(s)

    if write_stmt is None:
        raise RuleSkipped("AtomTile body unrecognised — no Write")

    if reduce_st is not None:
        # Shape A / B: extract Loads + Accum from the reduce body.
        loads: list[Load] = []
        accum: Accum | None = None
        for c in reduce_st.body:
            if isinstance(c, Load):
                loads.append(c)
            elif isinstance(c, Accum):
                accum = c
        if len(loads) != 2 or accum is None:
            raise RuleSkipped(f"AtomTile reduce body unrecognised — expected 2 Loads + Accum, got {len(loads)} Loads")
        K_name = reduce_st.axis.name
    elif flat_accum is not None and len(flat_loads) == 2:
        # Shape C: matmul body inline at the AtomTile level (single-iter K).
        loads = flat_loads
        accum = flat_accum
        # K name comes from the AtomTile's K axis — but AtomTile only
        # carries M_a / N_a; the K axis was the reduce loop the planner
        # built and then size-1 filtered. We identify A vs B by index
        # arity: A indexes [m_expr, k_expr] with K in the inner dim; B
        # indexes [k_expr, n_expr] with K in the outer dim. With K
        # filtered to a Literal(0), the inner dim of A is a constant
        # 0-index plus the per-row stride — distinguishable from B's
        # outer-dim constant 0 by axis ordering in the original LoopOp.
        # Use a heuristic: A's index[0] depends on M_expr (which depends
        # on the Write's index[0]); B's index[-1] depends on N_expr.
        K_name = "__filtered_k__"  # sentinel — not used in shape C
    else:
        raise RuleSkipped("AtomTile body unrecognised — no reduce SerialTile and no inline Load+Load+Accum")

    # Identify A and B operands. For shape A/B (K_name well-defined),
    # match by which index dim carries K. For shape C (K filtered),
    # match by which Load shares an axis with the Write's M (index[0])
    # vs N (index[-1]) Expr.
    a_load: Load | None = None
    b_load: Load | None = None
    if K_name != "__filtered_k__":
        for ld in loads:
            k_in_last = K_name in {v for e in ld.index[-1:] for v in e.free_vars()}
            k_in_first = K_name in {v for e in ld.index[:1] for v in e.free_vars()}
            if k_in_last and not k_in_first:
                a_load = ld  # K is the inner (last) dim → row-major A (M×K).
            elif k_in_first and not k_in_last:
                b_load = ld  # K is the outer dim → row-major B (K×N).
    else:
        # Shape C heuristic: A = the Load whose outer-dim Expr shares
        # free vars with the Write's outer-dim (M); B = whose inner-dim
        # shares free vars with the Write's inner-dim (N).
        w_m_vars = set(write_stmt.index[0].free_vars()) if write_stmt.index else set()
        w_n_vars = set(write_stmt.index[-1].free_vars()) if write_stmt.index else set()
        for ld in loads:
            ld_outer_vars = set(ld.index[0].free_vars()) if ld.index else set()
            ld_inner_vars = set(ld.index[-1].free_vars()) if ld.index else set()
            if w_m_vars & ld_outer_vars and not (w_n_vars & ld_inner_vars):
                a_load = ld
            elif w_n_vars & ld_inner_vars and not (w_m_vars & ld_outer_vars):
                b_load = ld
    if a_load is None or b_load is None:
        raise RuleSkipped("AtomTile Loads didn't match A=[M,K], B=[K,N] shape")

    a_dtype = spec.operand_dtypes["a"]
    b_dtype = spec.operand_dtypes["b"]
    # c_dtype tracks the destination buffer dtype so ``wmma::store_matrix_sync``
    # has a matching overload (the WMMA accumulator type must equal the
    # destination pointer's element type). Falls back to the spec's default
    # (F32) for non-F16/F32 outputs.
    c_dtype = c_dtype_override

    c_frag = f"{accum.name}_frag"
    a_frag = f"{a_load.names[0]}_frag"
    b_frag = f"{b_load.names[0]}_frag"

    a_src_index, a_ldm = _mma_src_index(a_load, bundle_sources)
    b_src_index, b_ldm = _mma_src_index(b_load, bundle_sources)
    inner: tuple[Stmt, ...] = (
        MmaLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_src_index, ldm=a_ldm),
        MmaLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_src_index, ldm=b_ldm),
        MmaSync(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag),
    )
    if reduce_st is not None:
        # Body shape A / B: rebuild the K_i / K_o wrappers around the
        # Mma* chain. The reduce SerialTile loses the ``reduce`` flag
        # (no more Accum inside); its kind stays the same.
        new_reduce_st = SerialTile(axis=reduce_st.axis, body=Body(inner), kind=reduce_st.kind)
        # M5: re-wrap with the StageBundle BETWEEN K_o and K_i — the
        # bundle's ``Source.origin`` may reference the K_o loop var
        # (e.g. ``a[a0*16, K_o*1024]``) so the producer must fire inside
        # K_o's scope. The structural shape we restore is exactly what
        # 020_stage_inputs emitted:
        # ``SerialTile(K_o) > StageBundle > SerialTile(K_i) > Mma chain``.
        if enclosing_bundle is not None:
            staged_inner = enclosing_bundle.with_bodies((Body(enclosing_bundle.stages), Body((new_reduce_st,))))
            inner_for_outer: tuple[Stmt, ...] = (staged_inner,)
        else:
            inner_for_outer = (new_reduce_st,)
        if outer_st is not None:
            k_loop_stmts: tuple[Stmt, ...] = (SerialTile(axis=outer_st.axis, body=Body(inner_for_outer), kind=outer_st.kind),)
        else:
            k_loop_stmts = inner_for_outer
    else:
        # Body shape C: single-iter K, no SerialTile wrapper. Emit the
        # Mma* chain inline (with the bundle re-wrap if applicable —
        # origin can still reference outer scope vars, but no K_o is
        # present so the bundle sits at the AtomTile level).
        if enclosing_bundle is not None:
            k_loop_stmts = (enclosing_bundle.with_bodies((Body(enclosing_bundle.stages), Body(inner))),)
        else:
            k_loop_stmts = inner

    fragments: tuple[Stmt, ...] = (
        MmaFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
        MmaFragment(name=a_frag, role="a", shape=spec.shape, dtype=a_dtype),
        MmaFragment(name=b_frag, role="b", shape=spec.shape, dtype=b_dtype),
    )
    return (
        *fragments,
        MmaFill(frag=c_frag, value=0.0),
        *k_loop_stmts,
        MmaStore(dst_buffer=write_stmt.output, dst_index=write_stmt.index, frag=c_frag, ldm=0),
    )


def _mma_src_index(load: Load, smem_sources: dict[str, Source]) -> tuple:
    """Choose the right ``src_index`` for an MMA fragment Load.

    Unstaged (load.input is the gmem buffer): use the gmem ``load.index``
    verbatim — pre-M5 behavior.

    Staged (load.input is an smem name registered by an enclosing
    ``StageBundle``): the consumer Load index is the bare cache-coord
    tuple (``020_stage_inputs`` rewrites ``smem_index = tuple(Var(ax.name)
    for ax in cache_axes)``). The slab itself is block-scaled per
    ``Source.alloc_extents``; the MMA fragment must read from the
    block-scaled coordinate. ``AffineAddressing.source_index`` threads
    ``block`` through the composite-stride decode so the resulting Expr
    is ``cache_var · block · stride_of_inner_axes`` per cache axis,
    relative to a zero origin (the slab is its own anchor). ``ldm``
    auto-resolves at render time via ``ctx.shapes[smem_name][-1]`` — the
    block-scaled inner extent.
    """
    src = smem_sources.get(load.input)
    if src is None:
        # Unstaged: gmem-direct WMMA load. ``ldm=0`` triggers the
        # render-time ``ctx.shapes[gmem_buf][-1]`` lookup, which is the
        # gmem tensor's inner extent — correct for the rank-2 gmem
        # operand.
        return load.index, 0
    if not isinstance(src.addressing, AffineAddressing):
        # Template-addressed Sources don't carry the block multiplier;
        # the cache vars in load.index decode verbatim through
        # ``addressing.exprs``, which the kernel renderer already
        # handles via the standard Load path. Fall back to the gmem-
        # style passthrough — same as the pre-M5 defensive branch.
        return load.index, 0
    # The smem slab is rank == len(cache_axes); render_index expects an
    # index tuple of the SAME rank so its row-major flatten lines up
    # with ``Source.alloc_extents``. The per-cache-axis slab coord is
    # ``Var(ax) * block_ax`` (scalar paths have block=() → bare Var).
    cache_axes = src.cache_axes
    block = src.addressing.block
    dims = src.addressing.dims
    out: list = []
    for i, ax in enumerate(cache_axes):
        b = block[i] if block else 1
        if b == 1:
            out.append(Var(ax.name))
        else:
            out.append(Var(ax.name) * Literal(b, "int"))
    # ldm for WMMA row_major matrix_a / matrix_b is the row stride along
    # the leading source dim — equivalently, the product of slab dims
    # for the *inner* source dim (dim 1 here). The auto-ldm path picks
    # ``ctx.shapes[name][-1]`` which collapses to the last alloc extent,
    # wrong when several cache axes share a source dim (e.g. an MMA
    # matmul whose N-side splits into warp + register cells). Compute
    # explicitly: ldm = ∏ alloc_extents[i] for i where dims[i] is the
    # inner source dim.
    alloc_extents = src.alloc_extents
    ldm_dim = max(dims) if dims else 0
    ldm = 1
    for i, d in enumerate(dims):
        if d == ldm_dim:
            ldm *= alloc_extents[i]
    return tuple(out), ldm
