"""Lower KernelOp to LoopIR.

``lower_generic()`` is the single entry point: it reads a ``Schedule`` and
emits LoopIR through five phases (grid → accumulators → reductions →
epilogue → write).  No pattern-matching — all structural decisions live
in the Schedule.

All contraction strategies (naive, tma, smem) route through the Schedule.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from deplodock.compiler.backend.ir.loop_ir import (
    Accum,
    AccumInit,
    Alloc,
    Builtin,
    FuncCall,
    Guard,
    Let,
    Literal,
    Load,
    LoopExpr,
    LoopNest,
    LoopProgram,
    OpCall,
    ParallelAxis,
    RegAccess,
    SetVar,
    ShuffleReduce,
    Store,
    Var,
)
from deplodock.compiler.ops import ElementwiseOp, KernelOp, ReshapeOp, TransposeOp
from deplodock.compiler.ops import _needed_by_ids as _needed_by

if TYPE_CHECKING:
    from deplodock.compiler.backend.cuda.schedule import Schedule


def _phases_view(region: KernelOp):
    """Bundle ``KernelOp.phases()`` tuple as a namespace for ``phases.X`` reads."""
    from types import SimpleNamespace

    p, r, i, e = region.phases()
    return SimpleNamespace(prologue=p, reduces=r, inter_reduce=i, epilogue=e)


def _pattern(region: KernelOp, shapes: dict[str, tuple]) -> str:
    """Derive the tile pattern from ``region`` / ``shapes`` (no schedule dependency)."""
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    return region.tile_pattern(shapes, out_shape)


def _is_batched(region: KernelOp, shapes: dict[str, tuple]) -> bool:
    """True when the region has a batched contraction (cinfo.batch_size > 1)."""
    cinfo = region.contraction_info(shapes)
    return cinfo is not None and cinfo.batch_size > 1


def _epilogue_per_element(region: KernelOp, shapes: dict[str, tuple]) -> bool:
    """Whether the epilogue needs a per-element second pass over inputs."""
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    return region.epilogue_needs_per_element(shapes, out_shape)


def _grid_type(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> str:
    """Derive the grid flavor ("1d" / "1d_contraction" / "2d_swizzle" / "2d_standard")
    from the region pattern + schedule.load_strategy.  Contractions with multi-reduce
    use 1d_contraction (online N-tiled reduction); smem-load contractions use
    2d_standard; all other contractions use 2d_swizzle; everything else is 1d."""
    if _pattern(region, shapes) != "contraction":
        return "1d"
    if len(region.reduce_fn_names()) > 1:
        return "1d_contraction"
    if schedule.load_strategy == "smem":
        return "2d_standard"
    return "2d_swizzle"


def _bound_var(region: KernelOp, shapes: dict[str, tuple]) -> str:
    """Bound variable for the 1D parallel axis: 'n' for pointwise, 'rows' otherwise."""
    return "n" if _pattern(region, shapes) == "pointwise" else "rows"


def _dim_params(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list[tuple[str, str]]:
    """Kernel signature scalar dim args, derived from pattern + batching + k_splits."""
    pattern = _pattern(region, shapes)
    if pattern == "pointwise":
        return [("int", "n")]
    if pattern != "contraction":
        return [("int", "rows"), ("int", "cols")]
    dim_params: list[tuple[str, str]] = [("int", "M"), ("int", "N"), ("int", "K")]
    if _is_batched(region, shapes):
        dim_params.append(("int", "batch_count"))
    elif schedule.k_splits > 1:
        dim_params.append(("int", "k_splits"))
    return dim_params


def _tma_params(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list[str] | None:
    """TMA descriptor param names when load_strategy == 'tma', else None."""
    if schedule.load_strategy != "tma":
        return None
    cinfo = region.contraction_info(shapes)
    if cinfo is None or not cinfo.a_id:
        return None
    return [f"{_safe(cinfo.a_id)}_tma", f"{_safe(cinfo.b_id)}_tma"]


def _tma_config(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]):
    """TMALoadConfig when load_strategy == 'tma', else None."""
    from deplodock.compiler.backend.cuda.schedule import TMALoadConfig

    if schedule.load_strategy != "tma":
        return None
    cinfo = region.contraction_info(shapes)
    if cinfo is None or not cinfo.a_id:
        return None
    is_batched = cinfo.batch_size > 1
    a_name = _safe(cinfo.a_id)
    b_name = _safe(cinfo.b_id)
    a_ref = f"&{a_name}_tma[batch]" if is_batched else f"&{a_name}_tma"
    b_ref = f"&{b_name}_tma[batch]" if is_batched else f"&{b_name}_tma"
    return TMALoadConfig(a_tma_ref=a_ref, b_tma_ref=b_ref)


def _includes(schedule: Schedule) -> list[str] | None:
    """Header includes; TMA kernels need cuda.h for descriptor types."""
    return ["cuda.h"] if schedule.load_strategy == "tma" else None


# ---------------------------------------------------------------------------
# Generic lowering — single entry point driven by Schedule
# ---------------------------------------------------------------------------


def lower_generic(
    region: KernelOp,
    name: str,
    shapes: dict[str, tuple],
    schedule: Schedule,
) -> LoopProgram:
    """Lower a KernelOp to LoopIR using a Schedule.

    Five phases, each parameterized by the Schedule:
    1. Grid setup
    2. Accumulator allocation
    3. Reduction loops
    4. Epilogue ops
    5. Write outputs
    """
    params = _build_params(region)
    params.extend(_dim_params(schedule, region, shapes))

    body: list = []

    # Phase 1: Grid setup
    body.extend(_emit_grid(schedule, region, shapes))

    # Phase 2: Accumulators
    body.extend(_emit_accumulators(schedule, region, shapes))

    # Phase 3: Reduction loops (0 for pointwise, 1+ for reduce/contraction)
    body.extend(_emit_reductions(schedule, region, shapes))

    # Phase 4: Epilogue
    body.extend(_emit_epilogue(schedule, region, shapes))

    # Phase 5: Write outputs
    body.extend(_emit_write(schedule, region, shapes))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        dim_strides=_build_dim_strides(region, shapes),
    )


# ---------------------------------------------------------------------------
# Phase 1: Grid setup
# ---------------------------------------------------------------------------


def _emit_grid(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list:
    grid = schedule.grid
    is_batched = _is_batched(region, shapes)
    gtype = _grid_type(schedule, region, shapes)
    if gtype == "1d":
        axis_name = "i" if _pattern(region, shapes) == "pointwise" else "row"
        return [ParallelAxis(axis_name, "blockIdx.x", _bound_var(region, shapes))]
    if gtype == "1d_contraction":
        # Online reduction: 1D grid over M-tiles, N is tiled sequentially.
        # Emit bm, tr, tc but NOT bn (comes from the N-tile loop).
        I = "int"  # noqa: E741
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        tile_m = schedule.tile_m or 64
        tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
        ops: list = []
        if is_batched:
            ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))
        ops.append(Let("bm", Builtin("blockIdx.x") * tile_m, dtype=I))
        ops.append(Let("tr", tidy * thread_m, dtype=I))
        ops.append(Let("tc", tidx * thread_n, dtype=I))
        ops.append(
            Guard(
                (Var("bm") + Var("tr")).ge(Var("M")),
                [],  # early return
            )
        )
        return ops
    if gtype == "2d_swizzle":
        return _cta_swizzle_grid(
            schedule.thread_m or 8,
            schedule.thread_n or 4,
            schedule.tile_m or 64,
            schedule.tile_n or 128,
            is_batched,
        )
    # 2d_standard (smem): row_base/col_base/sr grid
    if gtype == "2d_standard":
        I = "int"  # noqa: E741
        tx, ty, _ = grid.block_size
        thread_m = schedule.thread_m or 4
        thread_n = schedule.thread_n or 4
        bidy, bidx = Builtin("blockIdx.y"), Builtin("blockIdx.x")
        tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
        ops: list = []
        if is_batched:
            ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))
        ops.append(Let("row_base", (bidy * ty + tidy) * thread_m, dtype=I))
        ops.append(Let("col_base", (bidx * tx + tidx) * thread_n, dtype=I))
        ops.append(Let("sr", tidy * thread_m, dtype=I))
        return ops
    return []


# ---------------------------------------------------------------------------
# Phase 2: Accumulators
# ---------------------------------------------------------------------------


def _emit_accumulators(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list:
    ops: list = []
    pattern = _pattern(region, shapes)

    if pattern == "pointwise":
        return ops

    if pattern != "contraction":
        # Scalar accumulators (one per reduce op)
        for _node in region.phases()[1]:
            acc_name = f"acc_{_safe(_node.id)}"
            ops.append(AccumInit(acc_name, _node.op.fn))
        return ops

    # 2D register tile (contraction)
    ops.append(Alloc("c", "float", (schedule.thread_m, schedule.thread_n), "reg", Literal(0.0)))

    # Batch pointer aliases for contraction
    cinfo = region.contraction_info(shapes)
    if _is_batched(region, shapes) and cinfo is not None and cinfo.a_id:
        A = Var(_safe(cinfo.a_id))  # noqa: N806
        B = Var(_safe(cinfo.b_id))  # noqa: N806
        batch, M, K, N = Var("batch"), Var("M"), Var("K"), Var("N")  # noqa: N806
        # GQA / broadcast batch: one operand may have fewer batch elements.
        # E.g. 28 Q heads vs 4 KV heads → b_batch_group=7, B uses batch//7.
        a_batch_idx = batch / cinfo.a_batch_group if cinfo.a_batch_group > 1 else batch
        b_batch_idx = batch / cinfo.b_batch_group if cinfo.b_batch_group > 1 else batch
        ops.append(Let("Ab", A + a_batch_idx * (M * K), dtype="const float*"))
        ops.append(Let("Bb", B + b_batch_idx * (K * N), dtype="const float*"))

    return ops


# ---------------------------------------------------------------------------
# Phase 3: Reduction loops
# ---------------------------------------------------------------------------


def _emit_reductions(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    pattern = _pattern(region, shapes)
    if pattern == "pointwise":
        # Pointwise: no reduction, just inline all ops inside a guard
        return _emit_pointwise_body(region, shapes)

    if pattern != "contraction":
        # Scalar reduction (row_reduce / reduce_broadcast / multi-reduce)
        return _emit_scalar_reductions(region, shapes)

    # 2D register tile contraction
    if _grid_type(schedule, region, shapes) == "1d_contraction":
        # Contraction + multi-reduce: online N-tiled reduction.
        # Handles phases 3-5 (reductions, epilogue, write) internally.
        return _emit_online_contraction_reduce(schedule, region, shapes)

    # Standard contraction K-loop (single reduce, no multi-reduce)
    return _dispatch_k_loop(schedule, region, shapes)


def _dispatch_k_loop(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Pick the K-loop emitter based on schedule.load_strategy."""
    if schedule.load_strategy == "tma":
        return _emit_tma_k_loop(schedule, region, shapes)
    if schedule.load_strategy == "smem":
        return _emit_smem_k_loop(schedule, region, shapes)
    return _emit_contraction_k_loop(schedule, region, shapes)


def _emit_input_loads(
    region: KernelOp,
    shapes: dict[str, tuple],
    idx_var: str,
    prefix: str,
    body: list,
    var_map: dict[str, LoopExpr],
    out_size: int = 0,
) -> None:
    """Emit a Load for each external input; register the loaded Var in var_map.

    Used by pointwise, per-reduce, and per-element-epilogue passes — all
    three previously duplicated this loop with only prefix / idx_var differing.
    """
    for inp in [p.buffer_id for p in region.inputs]:
        indices = _input_indices(inp, region, shapes, idx_var, out_size)
        load_name = f"{prefix}{_safe(inp)}"
        body.append(Load(load_name, _safe(inp), indices, "global"))
        var_map[inp] = Var(load_name)


def _recompute_ops_into(
    ops: list,
    var_map: dict[str, LoopExpr],
    prefix: str,
    body: list,
    coord_mapping,
    shapes,
    include_ids,
    needed: set | None,
) -> None:
    """Emit IndexMap/Elementwise ops into body+var_map with shared filter logic.

    Used by per-reduce prologue recompute and per-element-epilogue recompute.
    When ``needed`` is None, all ops in ``ops`` are emitted (no filtering);
    otherwise only ops whose id is in ``needed`` are.
    """
    from deplodock.compiler.ops import IndexMapOp as _IndexMapOp

    for _node in ops:
        if needed is not None and _node.id not in needed:
            continue
        if isinstance(_node.op, _IndexMapOp):
            _emit_indexmap(_node.id, _node.op, _node.inputs, var_map, prefix, body, coord_mapping, shapes, include_ids)
        elif isinstance(_node.op, ElementwiseOp):
            a = var_map.get(_node.inputs[0], Literal(0.0)) if _node.inputs else Literal(0.0)
            b = var_map.get(_node.inputs[1], Literal(0.0)) if len(_node.inputs) > 1 else Literal(0.0)
            var_name = f"{prefix}{_safe(_node.id)}"
            body.append(Let(var_name, OpCall(_node.op.fn, [a] if len(_node.inputs) == 1 else [a, b])))
            var_map[_node.id] = Var(var_name)


def _emit_pointwise_body(
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Pointwise: load inputs, emit all ops, store outputs inside a guard."""
    out_shape = shapes.get([p.buffer_id for p in region.outputs][0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    guard_body: list = []
    var_map: dict[str, LoopExpr] = {}
    _emit_input_loads(region, shapes, "i", "v_", guard_body, var_map, out_size)

    coord_mapping = _build_pointwise_coord_mapping(out_shape)
    _emit_loop_ops(region.body_ops(), var_map, "v_", guard_body, coord_mapping=coord_mapping, shapes=shapes)

    for out_id in [p.buffer_id for p in region.outputs]:
        val = var_map.get(out_id, Literal(0.0))
        guard_body.append(Store(_safe(out_id), [Var("i")], val, "global"))

    return [Guard(Var("i").lt(Var("n")), guard_body)]


def _emit_scalar_reductions(
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit scalar reduction loops (single or multi-reduce)."""
    phases = _phases_view(region)
    out_shape = shapes.get([p.buffer_id for p in region.outputs][0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))
    guarded: list = []

    # Pre-reduction shape (input to the first reduce) — used to build the
    # IndexMap coord substitution for any IndexMaps inside the region.
    first_reduce_input = phases.reduces[0].inputs[0] if phases.reduces else None
    pre_shape = shapes.get(first_reduce_input, out_shape) if first_reduce_input else out_shape
    coord_mapping = _build_reduce_coord_mapping(pre_shape, Var("row"), Var("j"))

    # Accumulator allocs were already emitted in phase 2; build var tracking.
    reduce_vars: dict[str, tuple[str, str]] = {}
    for _node in phases.reduces:
        acc_name = f"acc_{_safe(_node.id)}"
        reduce_vars[_node.id] = (acc_name, _node.op.fn)

    var_map: dict[str, LoopExpr] = {}

    # One tile loop per reduce pass. For single-reduce this is one pass;
    # for multi-reduce (softmax) it's one per reduce with inter-reduce ops between.
    for ri, _node in enumerate(phases.reduces):
        node_id = _node.id
        input_ids = _node.inputs
        acc_name, fn = reduce_vars[node_id]
        pass_body: list = []

        pass_var_map: dict[str, LoopExpr] = {}
        _emit_input_loads(region, shapes, "j", f"r{ri}ld_", pass_body, pass_var_map, out_size)

        for prev_nid, (prev_acc, _prev_fn) in reduce_vars.items():
            pass_var_map[prev_nid] = Var(prev_acc)

        all_ops_this_pass = list(phases.inter_reduce[ri - 1]) if ri > 0 else []
        needed = _needed_by(all_ops_this_pass + [_node])
        prologue_ids = {pn.id for pn in phases.prologue}
        _recompute_ops_into(
            phases.prologue,
            pass_var_map,
            f"r{ri}p_",
            pass_body,
            coord_mapping,
            shapes,
            prologue_ids,
            needed,
        )

        if ri > 0 and phases.inter_reduce:
            _emit_loop_ops(phases.inter_reduce[ri - 1], pass_var_map, f"r{ri}_", pass_body, coord_mapping=coord_mapping, shapes=shapes)

        val = pass_var_map.get(input_ids[0], Literal(0.0))
        pass_body.append(Accum(acc_name, fn, val))

        guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), pass_body))
        guarded.append(ShuffleReduce(acc_name, fn))
        var_map[node_id] = Var(acc_name)

    return [Guard(Var("row").lt(Var("rows")), guarded)]


def _emit_contraction_k_loop(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit the K-loop for contraction (naive global-load strategy)."""
    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    cinfo = region.contraction_info(shapes)
    is_batched = _is_batched(region, shapes)
    A = Var(_safe(cinfo.a_id))  # noqa: N806
    B = Var(_safe(cinfo.b_id))  # noqa: N806
    a_src = Var("Ab") if is_batched else A
    b_src = Var("Bb") if is_batched else B

    bn, tc, bm, tr = Var("bn"), Var("tc"), Var("bm"), Var("tr")
    k, K, N, M = Var("k"), Var("K"), Var("N"), Var("M")  # noqa: N806

    port_imaps = region.port_indexmaps()
    a_imap = port_imaps.get(cinfo.a_id)
    b_imap = port_imaps.get(cinfo.b_id)

    def _apply_indexmap(indices: list, imap) -> list:
        if imap is None or not imap.sources:
            return indices
        from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

        src = imap.sources[0]
        mapping = {f"{PLACEHOLDER_PREFIX}{i}": ix for i, ix in enumerate(indices)}
        return [substitute(c, mapping) for c in src.coord_map]

    k_body: list = []
    for c in range(thread_n):
        col = bn + tc + c
        b_idx = _apply_indexmap([k, col], b_imap)
        k_body.append(Load(f"b{c}", b_src, b_idx, "global", guard=col.lt(N)))

    for r in range(thread_m):
        row = bm + tr + r
        a_idx = _apply_indexmap([row, k], a_imap)
        k_body.append(Load(f"a{r}", a_src, a_idx, "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accum(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))

    return [LoopNest("k", Literal(0, "int"), K, None, k_body)]


def _emit_tma_k_loop(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list:
    """Emit TMA double-buffered K-loop via SmemPipelineKLoop."""
    from deplodock.compiler.backend.ir.loop_ir import SmemPipelineKLoop

    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    tile_m = schedule.tile_m or 64
    tile_n = schedule.tile_n or 128
    bk = schedule.block_k
    a_size = tile_m * bk
    stage = a_size + bk * tile_n

    return [
        SmemPipelineKLoop(
            stages=2,
            tile_m=tile_m,
            tile_n=tile_n,
            block_k=bk,
            a_size=a_size,
            stage_size=stage,
            thread_m=thread_m,
            thread_n=thread_n,
            tx=schedule.grid.block_size[0],
            k_splits=schedule.k_splits,
            is_batched=_is_batched(region, shapes),
        )
    ]


def _emit_smem_k_loop(schedule: Schedule, region: KernelOp, shapes: dict[str, tuple]) -> list:
    """Emit smem K-tile loop via SmemPipelineKLoop.expand()."""
    from deplodock.compiler.backend.ir.loop_ir import SmemPipelineKLoop

    cinfo = region.contraction_info(shapes)
    is_batched = _is_batched(region, shapes)
    A = Var(_safe(cinfo.a_id))  # noqa: N806
    B = Var(_safe(cinfo.b_id))  # noqa: N806
    a_buf = "Ab" if is_batched else A.name
    b_buf = "Bb" if is_batched else B.name

    pipeline = SmemPipelineKLoop(
        stages=2,
        tile_m=schedule.tile_m or 16,
        tile_n=schedule.tile_n or 128,
        block_k=schedule.block_k,
        a_size=(schedule.tile_m or 16) * schedule.block_k,
        stage_size=(schedule.tile_m or 16) * schedule.block_k + schedule.block_k * (schedule.tile_n or 128),
        thread_m=schedule.thread_m or 4,
        thread_n=schedule.thread_n or 4,
        tx=schedule.grid.block_size[0],
        k_splits=schedule.k_splits,
        is_batched=is_batched,
        a_buf=a_buf,
        b_buf=b_buf,
    )
    return pipeline.expand()


# ---------------------------------------------------------------------------
# Online contraction + multi-reduce (handles phases 3-5 together)
# ---------------------------------------------------------------------------


def _emit_online_contraction_reduce(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit N-tiled contraction with generic online reduction.

    When a contraction (matmul) is followed by multi-reduce (e.g. softmax),
    we tile over N sequentially and process each reduce in a separate pass
    over the output buffer.  This works for any multi-reduce pattern without
    hardcoded correction factors.

    Structure:
      Loop 1: K-loop + prologue + head reduce + write raw scores
      Loop 2..len(reduces): apply inter_reduce ops, compute next reduce
      Final loop: apply all inter_reduce + epilogue ops, write result

    TODO: When N <= tile_n, a KernelIR optimizer pass could detect
    single-iteration N-loops and fuse them into one in-register pass,
    eliminating the global memory round-trip for raw scores.
    """
    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    tile_n = schedule.tile_n or 128
    phases = _phases_view(region)
    output = _safe([p.buffer_id for p in region.outputs][0])
    input_set = set([p.buffer_id for p in region.inputs])

    I = "int"  # noqa: E741
    M, N = Var("M"), Var("N")  # noqa: N806

    # The first reduce (reduces[0]) is the contraction reduce over K —
    # already handled by the K-loop.  The N-axis reduces start at index 1.
    # inter_reduce[0] contains ops between the K-reduce and the first N-reduce
    # (e.g. scale), which we apply on the register tile after the K-loop.
    n_reduces = phases.reduces[1:]  # reduces over N (skip contraction reduce)
    n_inter = phases.inter_reduce  # inter_reduce[i] is between reduces[i] and reduces[i+1]
    is_batched = _is_batched(region, shapes)

    def _out_indices(row: LoopExpr, col: LoopExpr) -> list[LoopExpr]:
        """Build output buffer indices, adding batch dim if needed."""
        if is_batched:
            return [Var("batch"), row, col]
        return [row, col]

    def _running_var(node_id: str, r: int) -> Var:
        """Get the per-row running accumulator variable for a reduce node."""
        return Var(f"{reduce_running[node_id]}{r}")

    def _apply_ops_with_running(
        val_expr: LoopExpr,
        op_entries: list,
        r: int,
    ) -> LoopExpr:
        """Apply a chain of Elementwise ops, threading per-row running
        reduction vars as the binary-op partner when the op references them.
        """
        for entry in op_entries:
            op_obj = entry.op
            op_inputs = entry.inputs
            if not isinstance(op_obj, ElementwiseOp):
                continue
            if op_obj.info.arity == 1:
                val_expr = OpCall(op_obj.fn, [val_expr])
                continue
            other_input = next((inp for inp in op_inputs if inp in reduce_running), None)
            if other_input is not None:
                other_var = _running_var(other_input, r)
                if op_inputs[0] == other_input:
                    val_expr = OpCall(op_obj.fn, [other_var, val_expr])
                else:
                    val_expr = OpCall(op_obj.fn, [val_expr, other_var])
            else:
                val_expr = OpCall(op_obj.fn, [val_expr, Literal(0.0)])
        return val_expr

    def _combine_running(rvar_r: str, tvar: str, fn: str) -> LoopExpr:
        """Fold a tile-reduced value into the per-row running accumulator."""
        if fn == "sum":
            return SetVar(rvar_r, Var(rvar_r) + Var(tvar))
        fc = "fmaxf" if fn == "max" else fn
        return SetVar(rvar_r, FuncCall(fc, [Var(rvar_r), Var(tvar)]))

    ops: list = []

    # Running accumulators for each N-axis reduce, one per row of the thread
    # tile.  Each thread handles thread_m rows, so we need thread_m variables
    # per reduce.  reduce_running maps node_id → base name (append {r} for row).
    reduce_running: dict[str, str] = {}  # node_id → base var name (e.g. "running_mx")
    for rn in n_reduces:
        base = f"running_{_safe(rn.id)}"
        reduce_running[rn.id] = base
        for r in range(thread_m):
            ops.append(AccumInit(f"{base}{r}", rn.op.fn))

    # ---------------------------------------------------------------
    # Loop 1: K-loop + post-contraction ops + head N-reduce + write raw scores
    # ---------------------------------------------------------------
    loop1_body: list = []
    loop1_body.append(Let("bn", Var("n_tile"), dtype=I))

    # Reset contraction accumulators
    for r in range(thread_m):
        for c in range(thread_n):
            loop1_body.append(SetVar(f"c{r}{c}", Literal(0.0)))

    loop1_body.extend(_dispatch_k_loop(schedule, region, shapes))

    # Apply inter_reduce[0] ops on register tile (ops between contraction
    # reduce and first N-reduce, e.g. scale multiplication).
    contraction_id = phases.reduces[0].id
    if n_inter:
        loop1_body.extend(
            _apply_ops_on_register_tile(
                n_inter[0],
                contraction_id,
                shapes,
                input_set,
                thread_m,
                thread_n,
                "ir0",
            )
        )

    # Head N-reduce (n_reduces[0], e.g. max) via warp_xor per row
    head_node = n_reduces[0]
    head_id = head_node.id
    head_op = head_node.op
    head_running = reduce_running[head_id]
    for r in range(thread_m):
        tvar = f"tile_{_safe(head_id)}_{r}"
        loop1_body.append(AccumInit(tvar, head_op.fn))
        for c in range(thread_n):
            col_c = Var("bn") + Var("tc") + c
            loop1_body.append(
                Guard(
                    col_c.lt(N),
                    [
                        Accum(tvar, head_op.fn, RegAccess("c", [r, c])),
                    ],
                )
            )
        loop1_body.append(ShuffleReduce(tvar, head_op.fn, kind="warp_xor"))
        loop1_body.append(_combine_running(f"{head_running}{r}", tvar, head_op.fn))

    # Write raw scores (contraction output after inter_reduce[0] ops, before N-reduces)
    loop1_body.extend(
        _contraction_write_ops(
            Var(output),
            thread_m,
            thread_n,
            is_batched,
            k_splits=1,
        )
    )

    ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), loop1_body))

    # ---------------------------------------------------------------
    # Loop 2..len(n_reduces): compute subsequent N-axis reduces
    # ---------------------------------------------------------------
    for ri in range(1, len(n_reduces)):
        red_node = n_reduces[ri]
        red_id = red_node.id
        red_op = red_node.op
        red_running = reduce_running[red_id]
        # inter_reduce between n_reduces[ri-1] and n_reduces[ri] is
        # phases.inter_reduce[ri] (offset by 1 because inter_reduce[0]
        # is between the contraction reduce and n_reduces[0])
        inter_idx = ri  # inter_reduce[ri] = ops between reduces[ri] and reduces[ri+1]
        inter_ops = n_inter[inter_idx] if inter_idx < len(n_inter) else []

        loop_body: list = []
        loop_body.append(Let("bn", Var("n_tile"), dtype=I))

        for r in range(thread_m):
            row = Var("bm") + Var("tr") + r
            row_body: list = []

            for c in range(thread_n):
                col = Var("bn") + Var("tc") + c
                ld_name = f"rv{ri}_{r}_{c}"
                row_body.append(Load(ld_name, output, _out_indices(row, col), "global", guard=col.lt(N)))

                # Apply inter_reduce ops (e.g. sub(v, running_max{r}), exp(...))
                val_expr = _apply_ops_with_running(Var(ld_name), inter_ops, r)
                vname = f"tv{ri}_{r}_{c}"
                row_body.append(Let(vname, val_expr))

            # Reduce across columns via warp_xor
            tvar = f"tile_{_safe(red_id)}_{r}"
            row_body.append(AccumInit(tvar, red_op.fn))
            for c in range(thread_n):
                col = Var("bn") + Var("tc") + c
                row_body.append(
                    Guard(
                        col.lt(N),
                        [
                            Accum(tvar, red_op.fn, Var(f"tv{ri}_{r}_{c}")),
                        ],
                    )
                )
            row_body.append(ShuffleReduce(tvar, red_op.fn, kind="warp_xor"))
            row_body.append(_combine_running(f"{red_running}{r}", tvar, red_op.fn))

            loop_body.append(Guard(row.lt(M), row_body))

        ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), loop_body))

    # ---------------------------------------------------------------
    # Final loop: apply all inter_reduce + epilogue ops, write result
    # ---------------------------------------------------------------
    final_body: list = []
    final_body.append(Let("bn", Var("n_tile"), dtype=I))

    for r in range(thread_m):
        row = Var("bm") + Var("tr") + r
        row_body: list = []

        for c in range(thread_n):
            col = Var("bn") + Var("tc") + c
            ld_name = f"fv_{r}_{c}"
            row_body.append(Load(ld_name, output, _out_indices(row, col), "global", guard=col.lt(N)))

            # Apply N-axis inter_reduce ops (skip inter_reduce[0] — already
            # applied on the register tile in loop 1) then epilogue ops.
            all_inter_ops = [op for group in n_inter[1:] for op in group]
            val_expr = _apply_ops_with_running(Var(ld_name), all_inter_ops, r)
            val_expr = _apply_ops_with_running(val_expr, list(phases.epilogue), r)

            result_name = f"fr_{r}_{c}"
            row_body.append(Let(result_name, val_expr))
            row_body.append(Store(output, _out_indices(row, col), Var(result_name), "global", guard=col.lt(N)))

        row_body_wrapped: list = []
        row_body_wrapped.append(Guard(row.lt(M), row_body))
        final_body.extend(row_body_wrapped)

    ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), final_body))

    return ops


# ---------------------------------------------------------------------------
# Phase 4: Epilogue
# ---------------------------------------------------------------------------


def _emit_epilogue(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    phases = _phases_view(region)
    pattern = _pattern(region, shapes)
    if pattern == "pointwise":
        # Pointwise: epilogue was inlined in the pointwise body
        return []

    if _grid_type(schedule, region, shapes) == "1d_contraction":
        # Online reduction handles epilogue internally
        return []

    if pattern == "contraction":
        # Contraction register tile epilogue
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        return _contraction_epilogue_ops(phases, shapes, [p.buffer_id for p in region.inputs], thread_m, thread_n)

    # Scalar reduction epilogue
    return _emit_scalar_epilogue(region, shapes)


def _emit_scalar_epilogue(
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit epilogue for scalar reductions (inside the row guard)."""
    phases = _phases_view(region)
    out_shape = shapes.get([p.buffer_id for p in region.outputs][0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))
    epi_per_elem = _epilogue_per_element(region, shapes)

    if not phases.epilogue and not epi_per_elem:
        return []

    reduce_vars: dict[str, tuple[str, str]] = {}
    for _node in phases.reduces:
        reduce_vars[_node.id] = (f"acc_{_safe(_node.id)}", _node.op.fn)

    var_map: dict[str, LoopExpr] = {}
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = Var(acc_name)

    has_multi_reduce = len(phases.reduces) > 1

    if has_multi_reduce or epi_per_elem:
        # Per-element epilogue loop
        epi_body: list = []
        epi_var_map: dict[str, LoopExpr] = dict(var_map)

        _emit_input_loads(region, shapes, "j", "epld_", epi_body, epi_var_map, out_size)

        # Coord mapping for IndexMaps: epilogue iterates per-element over output
        # at (row, j); pre-reduction shape governs how row decomposes for >2D.
        first_reduce_input = phases.reduces[0].inputs[0] if phases.reduces else None
        pre_shape = shapes.get(first_reduce_input, out_shape) if first_reduce_input else out_shape
        coord_mapping = _build_reduce_coord_mapping(pre_shape, Var("row"), Var("j"))

        # Re-compute prologue + inter_reduce ops
        all_inter_ops = [op for group in phases.inter_reduce for op in group] if has_multi_reduce else []
        recompute_ops = phases.prologue + all_inter_ops if has_multi_reduce else phases.prologue
        # Compute transitive closure of "needed" over recompute_ops so a
        # downstream epilogue op pulls in its prologue producers' producers.
        needed = set(_needed_by(phases.epilogue))
        for _node in reversed(recompute_ops):
            if _node.id in needed:
                needed.update(_node.inputs)
        recompute_ids = {_node.id for _node in recompute_ops}
        _recompute_ops_into(
            recompute_ops,
            epi_var_map,
            "ep_",
            epi_body,
            coord_mapping,
            shapes,
            recompute_ids,
            needed=None if has_multi_reduce else needed,
        )

        _emit_loop_ops(phases.epilogue, epi_var_map, "e_", epi_body, coord_mapping=coord_mapping, shapes=shapes)

        for out_id in [p.buffer_id for p in region.outputs]:
            val = epi_var_map.get(out_id, Literal(0.0))
            epi_body.append(Store(_safe(out_id), [Var("row"), Var("j")], val, "global"))

        return [LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), epi_body)]

    # Scalar epilogue (no per-element loop needed)
    ops: list = []
    _emit_loop_ops(phases.epilogue, var_map, "e_", ops)
    return ops


# ---------------------------------------------------------------------------
# Phase 5: Write outputs
# ---------------------------------------------------------------------------


def _emit_write(
    schedule: Schedule,
    region: KernelOp,
    shapes: dict[str, tuple],
) -> list:
    pattern = _pattern(region, shapes)
    if pattern == "pointwise":
        # Pointwise: writes were already emitted in the guard body
        return []

    if _grid_type(schedule, region, shapes) == "1d_contraction":
        # Online reduction handles writes internally
        return []

    if pattern == "contraction":
        # Contraction register tile write
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        out_id = [p.buffer_id for p in region.outputs][0]
        rb = Var(schedule.row_base_var) if schedule.row_base_var else None
        cb = Var(schedule.col_base_var) if schedule.col_base_var else None
        return _contraction_write_ops(
            Var(_safe(out_id)), thread_m, thread_n, _is_batched(region, shapes), schedule.k_splits, row_base=rb, col_base=cb
        )

    # Scalar reduction write
    phases = _phases_view(region)
    reduce_vars: dict[str, tuple[str, str]] = {}
    for _node in phases.reduces:
        reduce_vars[_node.id] = (f"acc_{_safe(_node.id)}", _node.op.fn)

    var_map: dict[str, LoopExpr] = {}
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = Var(acc_name)

    epi_per_elem = _epilogue_per_element(region, shapes)
    if epi_per_elem or len(phases.reduces) > 1:
        # Per-element writes were already emitted in the epilogue loop
        if phases.epilogue or epi_per_elem:
            return []
        # No epilogue — write last reduce result (thread 0 only)
        ops: list = []
        for out_id in [p.buffer_id for p in region.outputs]:
            val = var_map.get(out_id, Literal(0.0))
            ops.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), [Var("row")], val, "global")],
                )
            )
        return ops

    # Single reduce without per-element epilogue
    if phases.epilogue:
        # Scalar epilogue was emitted in phase 4; thread 0 writes
        ops = []
        # Re-derive var_map with epilogue results
        _emit_loop_ops(phases.epilogue, var_map, "e_", [])  # just update var_map
        for out_id in [p.buffer_id for p in region.outputs]:
            val = var_map.get(out_id, Literal(0.0))
            ops.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), [Var("row")], val, "global")],
                )
            )
        return ops

    # No epilogue — write last reduce result (thread 0)
    ops = []
    for out_id in [p.buffer_id for p in region.outputs]:
        val = var_map.get(out_id, Literal(0.0))
        ops.append(
            Guard(
                Builtin("threadIdx.x").eq(Literal(0, "int")),
                [Store(_safe(out_id), [Var("row")], val, "global")],
            )
        )
    return ops


# ---------------------------------------------------------------------------
# Convenience entry point: build_schedule + lower_generic in one call
# ---------------------------------------------------------------------------


def lower_to_loop_ir(
    region: KernelOp,
    name: str,
    shapes: dict[str, tuple],
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> tuple[LoopProgram, object]:
    """Lower a KernelOp to LoopIR via ``build_schedule()`` + ``lower_generic()``.

    Returns ``(loop_program, schedule)`` so callers can pass the Schedule
    through to ``loop_ir_to_kernel()``.
    """
    from deplodock.compiler.backend.cuda.schedule import build_schedule

    schedule = build_schedule(region, shapes, strategy, hints or {})
    return lower_generic(region, name, shapes, schedule), schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _build_dim_strides(
    region: KernelOp,
    shapes: dict[str, tuple],
) -> dict[str, list[str]]:
    """Build per-buffer stride variable names for multi-dim index flattening.

    Contraction buffers use M/N/K strides.  Reduction buffers use "cols".
    Pointwise buffers use a single flat index (no strides needed).
    """
    strides: dict[str, list[str]] = {}
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    pattern = region.tile_pattern(shapes, out_shape)
    input_access = region.input_accesses(shapes, out_shape)
    cinfo = region.contraction_info(shapes)

    if pattern == "contraction" and cinfo is not None and cinfo.a_id:
        a = _safe(cinfo.a_id)
        b = _safe(cinfo.b_id)
        strides[a] = ["K"]
        strides[b] = ["N"]
        # Batched pointer aliases
        strides["Ab"] = ["K"]
        strides["Bb"] = ["N"]
        batched = cinfo.batch_size > 1
        for out_id in [p.buffer_id for p in region.outputs]:
            if batched:
                strides[_safe(out_id)] = ["M * N", "N"]
            else:
                strides[_safe(out_id)] = ["N"]
        # Epilogue external inputs: use N for 2D, nothing for 1D/scalar
        for inp in [p.buffer_id for p in region.inputs]:
            safe = _safe(inp)
            if safe not in strides:
                acc = input_access.get(inp)
                if acc and acc.is_2d:
                    strides[safe] = ["N"]
    elif pattern in ("row_reduce", "reduce_broadcast", "multi_reduce"):
        # All 2D buffers use "cols" stride
        for inp in [p.buffer_id for p in region.inputs]:
            acc = input_access.get(inp)
            if acc and acc.is_2d:
                strides[_safe(inp)] = ["cols"]
        for out_id in [p.buffer_id for p in region.outputs]:
            strides[_safe(out_id)] = ["cols"]

    return strides


def _broadcast_index_expr(
    flat_idx: LoopExpr,
    out_shape: tuple[int, ...],
    inp_shape: tuple[int, ...],
) -> LoopExpr:
    """Compute input flat index from output flat index via broadcast mapping.

    Decomposes the output flat index into multi-dim coordinates, maps each
    coordinate through the input shape (clamping broadcast dims to 0), and
    recomposes into the input flat index using input strides.

    All shapes must be concrete ints.  The result is a single arithmetic
    expression (no loops) that the CUDA compiler can optimize.
    """
    ndim = len(out_shape)
    padded = (1,) * (ndim - len(inp_shape)) + inp_shape

    # Compute row-major strides.
    out_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        out_strides[d] = out_strides[d + 1] * (out_shape[d + 1] if isinstance(out_shape[d + 1], int) else 1)

    inp_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        inp_strides[d] = inp_strides[d + 1] * (padded[d + 1] if isinstance(padded[d + 1], int) else 1)

    result: LoopExpr = Literal(0, "int")
    for d in range(ndim):
        pd = padded[d]
        if not isinstance(pd, int) or pd == 1:
            continue  # broadcast dim — always index 0
        # coord_d = (flat_idx / out_stride_d) % out_dim_d
        coord: LoopExpr = flat_idx / out_strides[d] % out_shape[d]
        result = result + coord * inp_strides[d]

    return result


def _input_indices(
    inp: str,
    region: KernelOp,
    shapes: dict[str, tuple],
    idx_var: str,
    out_size: int = 0,
) -> list[LoopExpr]:
    """Build per-dimension index expressions for reading an input tensor.

    Returns a list of index expressions, one per logical dimension:
    - scalar → []
    - 1D (pointwise) → [i] or broadcast index expr
    - row-vector → [j]
    - per-row → [row]
    - broadcast → [broadcast_index_expr]
    - 2D → [row, j]

    If the input's Port carries an ``indexmap`` (set by
    070_absorb_indexmap_into_port), the natural indices are first
    substituted as placeholder coord values into the IndexMap's coord_map
    to produce the actual input-space indices — this is how
    transpose-into-matmul / slice-into-matmul load directly.
    """
    natural = _natural_input_indices(inp, region, shapes, idx_var, out_size)
    indexmap = region.port_indexmaps().get(inp)
    if indexmap is None or not indexmap.sources:
        return natural
    from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

    src = indexmap.sources[0]
    mapping = {f"{PLACEHOLDER_PREFIX}{i}": ix for i, ix in enumerate(natural)}
    return [substitute(c, mapping) for c in src.coord_map]


def _natural_input_indices(
    inp: str,
    region: KernelOp,
    shapes: dict[str, tuple],
    idx_var: str,
    out_size: int = 0,
) -> list[LoopExpr]:
    """Per-dim index expressions assuming no Port.indexmap substitution."""
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    pattern = region.tile_pattern(shapes, out_shape)
    acc = region.input_accesses(shapes, out_shape)[inp]
    j = Var(idx_var)

    if pattern == "pointwise":
        if acc.is_scalar:
            return []
        if acc.is_broadcast or acc.size < out_size:
            # Use stride-based multi-dim broadcast indexing.
            return [_broadcast_index_expr(Var("i"), out_shape, acc.shape)]
        return [Var("i")]

    if acc.is_broadcast:
        # Reduce pattern with broadcast input: build flat index from (row, j)
        # and apply broadcast mapping.
        flat_idx = Var("row") * Var("cols") + j
        return [_broadcast_index_expr(flat_idx, out_shape, acc.shape)]
    if acc.is_2d:
        return [Var("row"), j]
    if acc.is_per_row:
        return [Var("row")]
    if acc.is_row_vector:
        return [j]
    return []


def _emit_indexmap(
    node_id: str,
    op,
    input_ids: list[str],
    var_map: dict[str, LoopExpr],
    prefix: str,
    body: list,
    coord_mapping: dict[str, LoopExpr],
    shapes: dict[str, tuple],
    in_region_ids: set[str] | None = None,
) -> None:
    """Emit code for a single IndexMapOp inside a fused-region kernel.

    Substitutes the IndexMap's coord_map placeholders with the kernel's
    actual per-axis coord LoopExprs, flattens the per-axis indices using the
    input's known shape (so the codegen doesn't need stride scalars for these
    auxiliary buffers), emits one Load per source, and binds ``var_map[node_id]``
    to either the single Load var or a Ternary chain over multi-source loads.

    When ``in_region_ids`` is provided and the source's input is in that set,
    the source's value is taken directly from ``var_map`` (the in-region producer
    already computed it at the current output position; fusion ensured the
    coord_map is identity for in-region sources). External inputs always emit
    a fresh Load at the coord-substituted position.
    """
    from deplodock.compiler.backend.ir.expr import Ternary
    from deplodock.compiler.coord_expr import substitute

    var_name = f"{prefix}{_safe(node_id)}"
    src_values: list[tuple[LoopExpr, LoopExpr | None]] = []
    for i, src in enumerate(op.sources):
        src_input = input_ids[src.input_idx]
        is_in_region = in_region_ids is not None and src_input in in_region_ids
        if is_in_region and src_input in var_map:
            # Reuse the in-region producer's value at the current position.
            value: LoopExpr = var_map[src_input]
        else:
            external = _safe(src_input)
            in_shape = shapes.get(src_input, ())
            in_strides = _row_major_strides(in_shape) if in_shape else []
            substituted = [substitute(c, coord_mapping) for c in src.coord_map]
            # Pre-flatten the multi-dim indices into a single linear address using
            # the input's row-major strides — the codegen would otherwise emit
            # stride_X_d kernel-param references that we haven't declared.
            if len(substituted) > 1 and len(in_strides) == len(substituted):
                flat: LoopExpr = Literal(0, "int")
                for axis, expr in enumerate(substituted):
                    flat = flat + expr * Literal(in_strides[axis], "int")
                substituted = [flat]
            load_name = f"{var_name}_s{i}" if len(op.sources) > 1 else var_name
            body.append(Load(load_name, external, substituted, "global"))
            value = Var(load_name)
        sel = substitute(src.select, coord_mapping) if src.select is not None else None
        src_values.append((value, sel))
    if len(src_values) == 1:
        var_map[node_id] = src_values[0][0]
    else:
        # Build a Ternary chain ending in the last (default) source.
        result: LoopExpr = src_values[-1][0]
        for value, sel in reversed(src_values[:-1]):
            if sel is None:
                raise RuntimeError(f"IndexMapOp {node_id} multi-source needs select on every source except the last")
            result = Ternary(sel, value, result)
        var_map[node_id] = result


def _emit_loop_ops(
    ops: list,
    var_map: dict[str, LoopExpr],
    prefix: str,
    body: list,
    coord_mapping: dict[str, LoopExpr] | None = None,
    shapes: dict[str, tuple] | None = None,
) -> None:
    """Walk ops and emit Let(name, OpCall(...)) nodes, updating var_map.

    ``coord_mapping`` maps IndexMap placeholder var names (``out_coord_d``)
    to the kernel's per-axis coord LoopExprs in the current scope. ``shapes``
    maps buffer names to their full shape tuples (needed to flatten IndexMap
    multi-dim loads). Both are required when ``ops`` contains any
    ``IndexMapOp``; pointwise / reduce / epilogue callers build them per pattern.
    """
    from deplodock.compiler.ops import IndexMapOp

    in_region_ids = {n.id for n in ops}

    for _node in ops:
        if isinstance(_node.op, (ReshapeOp, TransposeOp)):
            if _node.inputs and _node.inputs[0] in var_map:
                var_map[_node.id] = var_map[_node.inputs[0]]
            continue

        if isinstance(_node.op, IndexMapOp):
            if coord_mapping is None or shapes is None:
                raise RuntimeError(
                    f"IndexMapOp {_node.id} reached _emit_loop_ops without coord_mapping/shapes; "
                    "the calling pattern lowering must build and pass them."
                )
            _emit_indexmap(_node.id, _node.op, _node.inputs, var_map, prefix, body, coord_mapping, shapes, in_region_ids)
            continue

        if isinstance(_node.op, ElementwiseOp):
            a = var_map.get(_node.inputs[0], Literal(0.0)) if _node.inputs else Literal(0.0)
            b = var_map.get(_node.inputs[1], Literal(0.0)) if len(_node.inputs) > 1 else Literal(0.0)
            var_name = f"{prefix}{_safe(_node.id)}"
            args = [a, b] if _node.op.info.arity == 2 and len(_node.inputs) > 1 else [a]
            body.append(Let(var_name, OpCall(_node.op.fn, args)))
            var_map[_node.id] = Var(var_name)


def _row_major_strides(shape: tuple) -> list[int]:
    """Row-major strides (last dim has stride 1). Symbolic dims are treated as 1."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        d = shape[i + 1] if isinstance(shape[i + 1], int) else 1
        strides[i] = strides[i + 1] * d
    return strides


def _build_pointwise_coord_mapping(out_shape: tuple) -> dict[str, LoopExpr]:
    """Decompose flat ``Var("i")`` into per-axis output coords for IndexMap substitution."""
    from deplodock.compiler.coord_expr import placeholder

    strides = _row_major_strides(out_shape)
    mapping: dict[str, LoopExpr] = {}
    for d in range(len(out_shape)):
        size = out_shape[d] if isinstance(out_shape[d], int) else 1
        if d == len(out_shape) - 1:
            mapping[placeholder(d).name] = Var("i") % Literal(size, "int")
        else:
            mapping[placeholder(d).name] = (Var("i") / Literal(strides[d], "int")) % Literal(size, "int")
    return mapping


def _build_reduce_coord_mapping(pre_shape: tuple, row_var: LoopExpr, j_var: LoopExpr) -> dict[str, LoopExpr]:
    """For row-reduce kernels: map placeholders to (row, j).

    For rank-2 ``pre_shape``: placeholder(0) → row, placeholder(1) → j.
    For higher ranks, decompose ``row`` across the leading dims using strides.
    """
    from deplodock.compiler.coord_expr import placeholder

    ndim = len(pre_shape)
    mapping: dict[str, LoopExpr] = {}
    if ndim == 0:
        return mapping
    mapping[placeholder(ndim - 1).name] = j_var
    if ndim == 1:
        return mapping
    leading = pre_shape[:-1]
    leading_strides = _row_major_strides(leading) if len(leading) > 1 else [1]
    for d in range(len(leading)):
        size = leading[d] if isinstance(leading[d], int) else 1
        if d == len(leading) - 1:
            mapping[placeholder(d).name] = row_var % Literal(size, "int")
        else:
            mapping[placeholder(d).name] = (row_var / Literal(leading_strides[d], "int")) % Literal(size, "int")
    return mapping


def _build_params(region: KernelOp) -> list[tuple[str, str]]:
    """Build kernel params, deduplicating buffers that are both input and output.

    When a buffer appears in both input_names and output_names (in-place op),
    it is emitted once as ``float*`` (read-write), not duplicated as both
    ``const float*`` and ``float*``.
    """
    params: list[tuple[str, str]] = []
    output_set = set([p.buffer_id for p in region.outputs])
    for inp in [p.buffer_id for p in region.inputs]:
        if inp in output_set:
            continue  # will be added as read-write output
        params.append(("const float* __restrict__", _safe(inp)))
    for out in [p.buffer_id for p in region.outputs]:
        params.append(("float* __restrict__", _safe(out)))
    return params


# ---------------------------------------------------------------------------
# Contraction helpers
# ---------------------------------------------------------------------------


def _cta_swizzle_grid(
    thread_m: int,
    thread_n: int,
    tile_m: int,
    tile_n: int,
    is_batched: bool,
) -> list:
    """Generate LoopIR ops for CTA-swizzle grid setup.

    Maps a linearized 1D grid (blockIdx.x) to 2D tile coordinates (bm, bn)
    using an 8-way swizzle for improved memory access patterns.
    Produces named variables: tr, tc, bm, bn (and batch if batched).
    """
    ops: list = []
    I = "int"  # noqa: E741
    N, M = Var("N"), Var("M")  # noqa: N806
    bidy, bidx = Builtin("blockIdx.y"), Builtin("blockIdx.x")
    tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
    gdimx = Builtin("gridDim.x")
    SWIZ = 8

    if is_batched:
        ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))

    ops.append(Let("tr", tidy * thread_m, dtype=I))
    ops.append(Let("tc", tidx * thread_n, dtype=I))
    ops.append(Let("ntx", (N + (tile_n - 1)) / tile_n, dtype=I))
    ops.append(Let("nty", (M + (tile_m - 1)) / tile_m, dtype=I))

    ntx, nty = Var("ntx"), Var("nty")
    pid, grp, rem = Var("pid"), Var("grp"), Var("rem")
    by_s, bx_s = Var("by_s"), Var("bx_s")

    ops.append(Let("pid", bidx + bidy * gdimx, dtype=I))
    ops.append(Let("grp", pid / (ntx * SWIZ), dtype=I))
    ops.append(Let("rem", pid % (ntx * SWIZ), dtype=I))
    ops.append(Let("by_s", grp * SWIZ + rem % SWIZ, dtype=I))
    ops.append(Let("bx_s", rem / SWIZ, dtype=I))

    ops.append(Guard(by_s.ge(nty).or_(bx_s.ge(ntx)), []))

    ops.append(Let("bm", by_s * tile_m, dtype=I))
    ops.append(Let("bn", bx_s * tile_n, dtype=I))

    return ops


def _contraction_epilogue_ops(
    phases,
    shapes: dict[str, tuple],
    input_names: list[str],
    thread_m: int,
    thread_n: int,
) -> list:
    """Generate LoopIR ops for contraction epilogue on the register tile.

    Handles both simple epilogues (bias, activation) and multi-reduce
    epilogues (softmax: inter_reduce ops + WarpShuffleXor between reduces).
    Walks phases generically — no pattern-specific checks.

    Structure:
    1. For each inter_reduce group: apply ops on tile, then per-row
       WarpShuffleXor for the corresponding reduce
    2. Apply epilogue ops on tile (referencing reduce results as per-row vars)
    """
    ops: list = []
    input_set = set(input_names)
    has_inter = bool(phases.inter_reduce)

    if not has_inter and not phases.epilogue:
        return []

    # Track the "previous" node ID through the op chain.
    prev_id = phases.reduces[0].id

    # Per-row reduce result variables: reduce_node_id → "rXXX{r}" template
    reduce_row_vars: dict[str, str] = {}

    # Process inter-reduce groups: ops between consecutive reduces.
    # inter_reduce[i] holds ops between reduces[i] and reduces[i+1].
    for ri, inter_ops in enumerate(phases.inter_reduce):
        # The reduce that follows this inter-group (reduces[ri+1])
        reduce_node_id = phases.reduces[ri + 1][0]
        reduce_fn = phases.reduces[ri + 1][1].fn

        # Apply inter-reduce ops on the register tile
        ops.extend(
            _apply_ops_on_register_tile(inter_ops, prev_id, shapes, input_set, thread_m, thread_n, f"ir{ri}", extra_vars=reduce_row_vars)
        )
        prev_id = inter_ops[-1][0] if inter_ops else prev_id

        # Per-row reduce via WarpShuffleXor
        for r in range(thread_m):
            rvar = f"r{reduce_fn}{r}"
            ops.append(AccumInit(rvar, reduce_fn))
            for c in range(thread_n):
                col_c = Var("bn") + Var("tc") + c
                ops.append(Guard(col_c.lt(Var("N")), [Accum(rvar, reduce_fn, RegAccess("c", [r, c]))]))
            ops.append(ShuffleReduce(rvar, reduce_fn, kind="warp_xor"))

        reduce_row_vars[reduce_node_id] = f"r{reduce_fn}{{r}}"

    # Apply epilogue ops on the register tile.
    if phases.epilogue:
        if reduce_row_vars:
            # Epilogue references reduce results as per-row variables.
            # Apply per-element with column guard.
            epi_prev = prev_id
            for r in range(thread_m):
                for c in range(thread_n):
                    col_c = Var("bn") + Var("tc") + c
                    # Resolve per-row vars for this specific row
                    row_extra = {k: v.format(r=r) for k, v in reduce_row_vars.items()}
                    epi_ops = _apply_ops_on_register_tile(
                        phases.epilogue, epi_prev, shapes, input_set, 1, 1, f"ep_{r}_{c}", extra_vars=row_extra
                    )
                    fixed: list = []
                    for tile_op in epi_ops:
                        if isinstance(tile_op, SetVar) and tile_op.name == "c00":
                            tile_op = SetVar(f"c{r}{c}", _fixup_reg_refs_expr(tile_op.expr, 0, 0, r, c))
                        fixed.append(tile_op)
                    ops.append(Guard(col_c.lt(Var("N")), fixed))
        else:
            # Simple epilogue (no inter-reduces): apply directly on full tile
            ops.extend(_apply_ops_on_register_tile(phases.epilogue, prev_id, shapes, input_set, thread_m, thread_n, "ep"))

    return ops


def _apply_ops_on_register_tile(
    op_list: list,
    prev_id: str,
    shapes: dict[str, tuple],
    input_set: set[str],
    thread_m: int,
    thread_n: int,
    prefix: str,
    extra_vars: dict[str, str] | None = None,
) -> list:
    """Apply a list of elementwise ops to the register tile c[r][c].

    Walks the op chain generically — any elementwise op is supported.
    Binary ops where the "other" input is an external buffer get a Load;
    binary ops where the other is a per-row variable (e.g. rmax, rsum)
    use ``extra_vars`` mapping.  Unary ops apply directly.
    """
    ops: list = []
    extra = extra_vars or {}

    for _node in op_list:
        if not isinstance(_node.op, ElementwiseOp):
            continue
        fn = _node.op.fn

        # Find the "other" input (not the accumulator chain) for binary ops.
        other = None
        if _node.op.info.arity == 2 and len(_node.inputs) == 2:
            for inp in _node.inputs:
                if inp != prev_id:
                    other = inp
                    break
        if other is not None and other not in input_set and other not in extra:
            other = None

        for r in range(thread_m):
            for c in range(thread_n):
                acc = RegAccess("c", [r, c])
                dst = f"c{r}{c}"

                if other is not None and other in extra:
                    # Per-row variable (rmax, rsum)
                    other_val = Var(extra[other].format(r=r))
                    if _node.inputs[0] == prev_id:
                        ops.append(SetVar(dst, OpCall(fn, [acc, other_val])))
                    else:
                        ops.append(SetVar(dst, OpCall(fn, [other_val, acc])))
                elif other is not None:
                    # External buffer input
                    safe = _safe(other)
                    other_shape = shapes.get(other, ())
                    other_size = math.prod(d for d in other_shape if isinstance(d, int))
                    if other_size <= 1:
                        indices: list = []
                    elif len(other_shape) <= 1:
                        indices = [Var("bn") + Var("tc") + c]
                    else:
                        row_e = Var("bm") + Var("tr") + r
                        col_e = Var("bn") + Var("tc") + c
                        indices = [row_e, col_e]
                    ld_name = f"_{prefix}_{safe}_{r}_{c}"
                    ops.append(Load(ld_name, safe, indices, "global"))
                    if _node.inputs[0] == prev_id:
                        ops.append(SetVar(dst, OpCall(fn, [acc, Var(ld_name)])))
                    else:
                        ops.append(SetVar(dst, OpCall(fn, [Var(ld_name), acc])))
                else:
                    # Unary op on accumulator
                    ops.append(SetVar(dst, OpCall(fn, [acc])))

        prev_id = _node.id

    return ops


def _fixup_reg_refs_expr(expr: LoopExpr, from_r: int, from_c: int, to_r: int, to_c: int) -> LoopExpr:
    """Replace RegAccess("c", [from_r, from_c]) with RegAccess("c", [to_r, to_c]) in an expression tree."""
    if isinstance(expr, RegAccess) and expr.name == "c" and expr.indices == [from_r, from_c]:
        return RegAccess("c", [to_r, to_c])
    if isinstance(expr, OpCall):
        return OpCall(expr.op, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    if isinstance(expr, FuncCall):
        return FuncCall(expr.name, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    return expr


def _contraction_write_ops(
    output: Var,
    thread_m: int,
    thread_n: int,
    is_batched: bool,
    k_splits: int,
    row_base: LoopExpr | None = None,
    col_base: LoopExpr | None = None,
) -> list:
    """Generate LoopIR ops for the contraction write phase.

    ``row_base`` and ``col_base`` are the base expressions for computing
    global row/col indices.  Defaults: ``bm + tr`` and ``bn + tc``
    (CTA-swizzle layout).  Smem uses ``row_base`` and ``col_base`` vars.
    """
    ops: list = []
    M, N = Var("M"), Var("N")  # noqa: N806
    rb = row_base if row_base is not None else Var("bm") + Var("tr")
    cb = col_base if col_base is not None else Var("bn") + Var("tc")

    for r in range(thread_m):
        row = rb + r
        row_body: list = []
        for c_idx in range(thread_n):
            col = cb + c_idx
            if is_batched:
                indices = [Var("batch"), row, col]
            else:
                indices = [row, col]
            row_body.append(Store(output, indices, RegAccess("c", [r, c_idx]), "global", guard=col.lt(N), atomic=k_splits > 1))
        ops.append(Guard(row.lt(M), row_body))

    return ops
