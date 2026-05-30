"""Lower AtomTile to MMA fragment Stmts — M5 of
``plans/mma-fragment-factorization.md``.

Runs *before* ``010_split_register_axes`` in the kernel chain. Pattern-
matches the warp-tier matmul shape the partition planner emits
(``RegisterTile > AtomTile > matmul-cell-body``), replaces the cell
body with an ``MmaFragment`` + ``MmaFill`` + per-K_i ``MmaLoad`` /
``MmaSync`` chain plus a final ``MmaStore``, and strips the AtomTile
wrapper (its axes encoded the cell shape, which is now baked into the
Mma* Stmts' ``shape`` field).

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
from deplodock.compiler.ir.kernel.ir import MmaFill, MmaFragment, MmaLoad, MmaStore, MmaSync
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import AtomTile, SerialTile, Stage, StageBundle, TileOp
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

    lowered, found = _lower_atom_tiles(tt.body, spec=spec, c_dtype_override=c_dtype_override)
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


def _lower_atom_tiles(body: Body, *, spec: AtomSpec, c_dtype_override: DataType) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile`` encountered, rewrite its
    interior matmul-shape body into an Mma* fragment chain and strip
    the AtomTile wrapper. Recurses into non-AtomTile block stmts so a
    deep-nested AtomTile (under RegisterTile / SerialTile / Cond / ...)
    is reached. Returns ``(new_body, found_any)``."""
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            new_stmts = _atom_body_to_mma(s.body, spec=spec, c_dtype_override=c_dtype_override)
            out.extend(new_stmts)
            found = True
            continue
        if isinstance(s, (Stage, StageBundle)):
            # MMA TileOps bypass staging in v1 (skip guard at
            # tile/020_stage_inputs); a Stage here would mean the guard
            # failed. Pass through defensively.
            out.append(s)
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_lowered = False
            for sub in s.nested():
                new_sub, sub_found = _lower_atom_tiles(sub, spec=spec, c_dtype_override=c_dtype_override)
                new_bodies.append(new_sub)
                any_lowered = any_lowered or sub_found
            found = found or any_lowered
            out.append(s.with_bodies(tuple(new_bodies)))
            continue
        out.append(s)
    return Body(out), found


def _atom_body_to_mma(body: Body, *, spec: AtomSpec, c_dtype_override: DataType) -> tuple[Stmt, ...]:
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
    write_stmt: Write | None = None
    reduce_st: SerialTile | None = None
    outer_st: SerialTile | None = None
    flat_loads: list[Load] = []
    flat_accum: Accum | None = None

    for s in body:
        if isinstance(s, Write):
            write_stmt = s
            continue
        if isinstance(s, SerialTile):
            if any(isinstance(c, SerialTile) and c.is_reduce for c in s.body):
                outer_st = s
                reduce_st = next(c for c in s.body if isinstance(c, SerialTile) and c.is_reduce)
            elif s.is_reduce:
                reduce_st = s
        # Body shape C: degenerate (both K loops filtered) — matmul body
        # sits directly inside the AtomTile.
        elif isinstance(s, Load):
            flat_loads.append(s)
        elif isinstance(s, Accum):
            flat_accum = s

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

    inner: tuple[Stmt, ...] = (
        MmaLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_load.index, ldm=0),
        MmaLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_load.index, ldm=0),
        MmaSync(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag),
    )
    if reduce_st is not None:
        # Body shape A / B: rebuild the K_i / K_o wrappers around the
        # Mma* chain. The reduce SerialTile loses the ``reduce`` flag
        # (no more Accum inside); its kind stays the same.
        new_reduce_st = SerialTile(axis=reduce_st.axis, body=Body(inner), kind=reduce_st.kind)
        if outer_st is not None:
            k_loop_stmts: tuple[Stmt, ...] = (SerialTile(axis=outer_st.axis, body=Body((new_reduce_st,)), kind=outer_st.kind),)
        else:
            k_loop_stmts = (new_reduce_st,)
    else:
        # Body shape C: single-iter K, no SerialTile wrapper. Emit the
        # Mma* chain inline.
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
