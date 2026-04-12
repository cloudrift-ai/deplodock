"""Unified tiled kernel generator.

Produces a KernelDef from a TileAnalysis for any fused region pattern:
pointwise, row_reduce, reduce_broadcast, or contraction.

A single code path handles all patterns through composable phases:
  1. Grid setup — thread-to-element mapping
  2. Accumulator declarations (if reduction)
  3. Tile loop — iterate over reduced dimension
  4. Prologue — elementwise ops before the reduce
  5. Accumulation — reduce ops
  6. Cross-thread reduce — warp shuffle + smem (row reductions only)
  7. Epilogue — elementwise ops after the reduce
  8. Write — output to global memory
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, _needed_by
from deplodock.compiler.backend.cuda.ir import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinOp,
    CudaBuiltin,
    Expr,
    ForLoop,
    FuncCall,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    RawCode,
    SyncThreads,
    Var,
    VarDecl,
)
from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReshapeOp, TransposeOp

# ---------------------------------------------------------------------------
# Convenience entry points (moved from fused.py and matmul.py shims)
# ---------------------------------------------------------------------------


def generate_kernel(region: FusedRegionOp, name: str, shapes: dict[str, tuple]) -> KernelDef:
    """Generate a CUDA kernel from a FusedRegionOp.

    Analyzes the region and produces a KernelDef via the unified tiled generator.
    """
    from deplodock.compiler.backend.cuda.generators.analysis import analyze

    analysis = analyze(region, shapes)
    return lower_tiled(region, name, shapes, analysis)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _build_expr(fn: str, a: Expr, b: Expr | None = None) -> Expr | None:
    """Map an ElementwiseOp function name to an AST expression."""
    if fn == "mul":
        return BinOp("*", a, b)
    if fn == "add":
        return BinOp("+", a, b)
    if fn == "sub":
        return BinOp("-", a, b)
    if fn == "div":
        return BinOp("/", a, b)
    if fn == "neg":
        return BinOp("-", Literal(0.0), a)
    if fn == "exp":
        return FuncCall("expf", [a])
    if fn == "rsqrt":
        return FuncCall("rsqrtf", [a])
    if fn == "recip":
        return BinOp("/", Literal(1.0), a)
    return None


def _reduce_op_expr(fn: str, acc: Expr, val: Expr) -> Expr:
    """Build the accumulation expression for a ReduceOp."""
    if fn == "max":
        return FuncCall("fmaxf", [acc, val])
    # sum (and anything else) → acc + val
    return BinOp("+", acc, val)


def _reduce_init(fn: str) -> Literal:
    """Initial value for a reduce accumulator."""
    if fn == "max":
        return Literal(-1e30)
    return Literal(0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lower_tiled(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> KernelDef:
    """Generate a KernelDef for a fused region based on its TileAnalysis.

    A single generator handles all pattern classes (pointwise, row_reduce,
    reduce_broadcast, contraction) through composable phases parameterized
    by the TileAnalysis.

    For contraction strategies (naive or tma_db), the same code path handles
    grid setup, accumulators, and write.  Only the K-loop body differs.
    """
    return _lower_naive(region, name, shapes, analysis, strategy=strategy, hints=hints or {})


# ---------------------------------------------------------------------------
# Input indexing — how each input is accessed per pattern
# ---------------------------------------------------------------------------


def _input_expr(inp: str, analysis: TileAnalysis, idx_var: str, out_size: int = 0) -> Expr:
    """Build the array access expression for an input tensor.

    Args:
        inp: Input name.
        analysis: TileAnalysis for AccessPattern lookup.
        idx_var: Index variable name — "i" for pointwise, "j" for reductions, "k" for contractions.
        out_size: Total output elements (for pointwise broadcast).
    """
    acc = analysis.input_access[inp]
    safe = _safe(inp)

    if analysis.pattern == "pointwise":
        if acc.is_scalar:
            return ArrayAccess(safe, Literal(0, "int"))
        if acc.size < out_size:
            return ArrayAccess(safe, BinOp("%", Var("i"), Literal(acc.size, "int")))
        return ArrayAccess(safe, Var("i"))

    # row_reduce / reduce_broadcast: index by [row * cols + j] or [j] or [0]
    if acc.is_2d:
        return ArrayAccess(safe, BinOp("+", BinOp("*", Var("row"), Var("cols")), Var("j")))
    if acc.is_row_vector:
        return ArrayAccess(safe, Var("j"))
    if acc.is_scalar:
        return ArrayAccess(safe, Literal(0, "int"))
    return ArrayAccess(safe, Literal(0, "int"))


def _contraction_input_expr(inp: str, analysis: TileAnalysis) -> Expr:
    """Build input access for the contraction epilogue: index by [row*N+col]."""
    acc = analysis.input_access[inp]
    safe = _safe(inp)
    if acc.is_scalar:
        return ArrayAccess(safe, Literal(0, "int"))
    if acc.is_row_vector:
        return ArrayAccess(safe, Var("col"))
    # Default: 2D indexing
    return ArrayAccess(safe, BinOp("+", BinOp("*", Var("row"), Var("N")), Var("col")))


# ---------------------------------------------------------------------------
# Op emission — walk ops and build AST expressions
# ---------------------------------------------------------------------------


def _emit_ops(
    ops: list,
    var_map: dict[str, Expr],
    prefix: str,
) -> list:
    """Walk ops and emit VarDecl statements, updating var_map.

    Args:
        ops: List of (node_id, op, input_ids) tuples.
        var_map: Mutable map of node_id → Expr, updated in place.
        prefix: Variable name prefix ("p_" for prologue, "e_" for epilogue, "v_" for pointwise).

    Returns:
        List of VarDecl statements.
    """
    stmts = []
    for node_id, op, input_ids in ops:
        if isinstance(op, (ReshapeOp, TransposeOp)):
            if input_ids and input_ids[0] in var_map:
                var_map[node_id] = var_map[input_ids[0]]
            continue

        if isinstance(op, ElementwiseOp):
            a = var_map.get(input_ids[0], Literal(0.0)) if input_ids else Literal(0.0)
            b = var_map.get(input_ids[1], Literal(0.0)) if len(input_ids) > 1 else Literal(0.0)
            expr = _build_expr(op.fn, a, b)
            if expr is None:
                expr = Literal(0.0)
            var_name = f"{prefix}{_safe(node_id)}"
            stmts.append(VarDecl("float", var_name, expr))
            var_map[node_id] = Var(var_name)

    return stmts


# ---------------------------------------------------------------------------
# Warp-shuffle cross-thread reduction (AST-based)
# ---------------------------------------------------------------------------


def _emit_warp_reduce(node_id: str, acc_var: str, fn: str) -> list:
    """Emit warp-shuffle + cross-warp shared memory reduction for one accumulator.

    Pattern:
      1. Intra-warp: __shfl_down_sync in a loop (warpSize/2 → 1)
      2. Lane 0 of each warp writes to __shared__ warp_X[warp_id]
      3. First warp loads from warp_X[], repeats shuffle
      4. Thread 0 writes to __shared__ s_X, all threads read it back
    """
    safe = _safe(node_id)
    acc = Var(acc_var)
    init_val = _reduce_init(fn)

    # The warp shuffle uses `offset >>= 1` which isn't a standard counted loop,
    # so we use RawCode for the shuffle loops and AST for the shared memory ops.
    stmts: list = []

    # Intra-warp shuffle
    if fn == "sum":
        shfl_body = f"{acc_var} += __shfl_down_sync(0xffffffff, {acc_var}, offset);"
    else:
        shfl_body = f"{acc_var} = fmaxf({acc_var}, __shfl_down_sync(0xffffffff, {acc_var}, offset));"
    stmts.append(RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {shfl_body}"))

    # Cross-warp via shared memory
    warp_arr = f"warp_{safe}"
    s_var = f"s_{safe}"
    stmts.append(ArrayDecl("__shared__ float", warp_arr, [8]))
    stmts.append(
        IfStmt(
            BinOp("==", BinOp("%", CudaBuiltin("threadIdx.x"), CudaBuiltin("warpSize")), Literal(0, "int")),
            [Assign(ArrayAccess(warp_arr, BinOp("/", CudaBuiltin("threadIdx.x"), CudaBuiltin("warpSize"))), acc)],
        )
    )
    stmts.append(SyncThreads())

    # First warp loads partial results
    stmts.append(
        IfStmt(
            BinOp("<", CudaBuiltin("threadIdx.x"), BinOp("/", CudaBuiltin("blockDim.x"), CudaBuiltin("warpSize"))),
            [VarDecl("__placeholder", "", None)],  # placeholder — need conditional assign
        )
    )
    # The conditional load + fallback init is easier as RawCode
    stmts.pop()  # remove placeholder
    stmts.append(
        RawCode(
            f"if (threadIdx.x < blockDim.x / warpSize) {acc_var} = {warp_arr}[threadIdx.x];\n"
            f"else {acc_var} = {init_val.value:.1f}f;"
        )
    )

    # Second shuffle pass (across warps)
    stmts.append(RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {shfl_body}"))

    # Broadcast final result to all threads via shared scalar
    stmts.append(VarDecl("__shared__ float", s_var))
    stmts.append(
        IfStmt(
            BinOp("==", CudaBuiltin("threadIdx.x"), Literal(0, "int")),
            [RawCode(f"{s_var} = {acc_var};")],
        )
    )
    stmts.append(SyncThreads())
    stmts.append(RawCode(f"{acc_var} = {s_var};"))

    return stmts


# ---------------------------------------------------------------------------
# Unified naive lowering — single path for all patterns
# ---------------------------------------------------------------------------


def _lower_naive(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> KernelDef:
    """Single generator for all patterns.  For contractions, `strategy`
    selects the K-loop implementation: 'naive' (global loads) or 'tma_db'
    (TMA double-buffered shared memory pipeline).  All other phases
    (grid, accumulators, write) are shared.

    Hints (cuda.matmul.* namespace, without prefix):
        block_k: K-tile size (default 16 for naive, 32 for tma_db)
    """
    is_contraction = analysis.pattern == "contraction"
    is_pointwise = analysis.pattern == "pointwise"
    has_reduce = bool(analysis.op_phases.reduces)
    phases = analysis.op_phases

    # --- Params ---
    params = []
    for inp in region.input_names:
        params.append(KernelParam("const float* __restrict__", _safe(inp)))
    for out in region.output_names:
        params.append(KernelParam("float* __restrict__", _safe(out)))

    if is_contraction:
        params.extend([KernelParam("int", "M"), KernelParam("int", "N"), KernelParam("int", "K")])
        # Block (32, 8) = 256 threads.  Each thread computes 8×4 = 32 outputs.
        # Output tile per block: 64 rows × 128 cols (same as TMA strategy).
        tx, ty = 32, 8
        thread_m, thread_n = 8, 4  # coarsening per thread
        tile_m = ty * thread_m  # 64
        tile_n = tx * thread_n  # 128
        block_size = (tx, ty, 1)
    elif is_pointwise:
        params.append(KernelParam("int", "n"))
        block_size = (256, 1, 1)
    else:
        params.extend([KernelParam("int", "rows"), KernelParam("int", "cols")])
        block_size = (256, 1, 1)

    body: list = []

    # --- Phase 1: Grid setup ---
    def _thread_idx(block_dim: str, block_idx: str, thread_idx: str) -> Expr:
        return BinOp("+", BinOp("*", CudaBuiltin(block_idx), CudaBuiltin(block_dim)), CudaBuiltin(thread_idx))

    if is_contraction:
        # CTA-swizzle grid + coarsened 8×4 thread mapping.
        # Shared structure with TMA strategy — only the load phase differs.
        body.append(RawCode(
            f"int tr=threadIdx.y*{thread_m},tc=threadIdx.x*{thread_n};\n"
            f"const int SWIZ=8;\n"
            f"int ntx=(N+{tile_n - 1})/{tile_n};\n"
            f"int nty=(M+{tile_m - 1})/{tile_m};\n"
            f"int pid=blockIdx.x+blockIdx.y*gridDim.x;\n"
            f"int grp=pid/(ntx*SWIZ);\n"
            f"int rem=pid%(ntx*SWIZ);\n"
            f"int by_s=grp*SWIZ+rem%SWIZ;\n"
            f"int bx_s=rem/SWIZ;\n"
            f"if(by_s>=nty||bx_s>=ntx)return;\n"
            f"int bm=by_s*{tile_m},bn=bx_s*{tile_n};"
        ))
    elif is_pointwise:
        body.append(VarDecl("int", "i", _thread_idx("blockDim.x", "blockIdx.x", "threadIdx.x")))
        body.append(IfStmt(BinOp(">=", Var("i"), Var("n")), [RawCode("return;")]))
    else:
        body.append(VarDecl("int", "row", BinOp("*", Literal(1, "int"), CudaBuiltin("blockIdx.x"))))
        body.append(IfStmt(BinOp(">=", Var("row"), Var("rows")), [RawCode("return;")]))

    # --- Build var_map for input access ---
    var_map: dict[str, Expr] = {}
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    if is_contraction:
        # Contraction inputs are indexed inside the k-loop; set up later.
        pass
    else:
        for inp in region.input_names:
            var_map[inp] = _input_expr(inp, analysis, "j" if has_reduce else "i", out_size)

    # --- Pointwise: emit all ops inline, write, return ---
    if is_pointwise:
        body.extend(_emit_ops(region.region_ops, var_map, "v_"))
        for out_id in region.output_names:
            val = var_map.get(out_id, Literal(0.0))
            body.append(Assign(ArrayAccess(_safe(out_id), Var("i")), val))
        return KernelDef(name=name, params=params, body=body, block_size=block_size)

    # --- Phase 2: Accumulator declarations ---
    reduce_vars: dict[str, tuple[str, str]] = {}  # node_id → (acc_var_name, fn)
    for node_id, op, _ in phases.reduces:
        acc_name = f"acc_{_safe(node_id)}"
        body.append(VarDecl("float", acc_name, _reduce_init(op.fn)))
        reduce_vars[node_id] = (acc_name, op.fn)

    # --- Phase 3+4+5: Tile loop with prologue + accumulation ---
    if is_contraction:
        a_name = _safe(analysis.contraction_a)
        b_name = _safe(analysis.contraction_b)
        out_id = region.output_names[0]
        c_name = _safe(out_id)
        _hints = hints or {}
        default_bk = 32 if strategy == "tma_db" else 16
        bk = int(_hints.get("block_k", default_bk))
        k_splits = int(_hints.get("k_splits", 1))
        a_size = tile_m * bk
        b_size = bk * tile_n
        stage = a_size + b_size

        # --- Shared: accumulator declarations ---
        acc_decls = ",".join(
            f"c{r}{c}=0" for r in range(thread_m) for c in range(thread_n)
        )

        # --- Shared: FMA block (reads from `cas`/`cbs` for TMA, from global for naive) ---
        fma_lines = []
        for r in range(thread_m):
            fma_lines.append(
                f"float a{r}=cas[(tr+{r})*{bk}+kk];"
                + "".join(f"c{r}{c}+=a{r}*b{c};" for c in range(thread_n))
            )
        fma_block = "\n            ".join(fma_lines)

        # --- Shared: W() macro + write calls ---
        w_calls = "\n    ".join(
            f"W({r},{','.join(f'c{r}{c}' for c in range(thread_n))})"
            for r in range(thread_m)
        )
        # W() macro: bounds-checked writes. Uses atomicAdd when k_splits > 1
        # (multiple blocks accumulate partial results into the same output).
        if k_splits > 1:
            w_body = (
                "if(gc<N)atomicAdd(&Cout[gr*N+gc],v0);"
                "if(gc+1<N)atomicAdd(&Cout[gr*N+gc+1],v1);"
                "if(gc+2<N)atomicAdd(&Cout[gr*N+gc+2],v2);"
                "if(gc+3<N)atomicAdd(&Cout[gr*N+gc+3],v3);"
            )
        else:
            w_body = (
                "if(gc<N)Cout[gr*N+gc]=v0;"
                "if(gc+1<N)Cout[gr*N+gc+1]=v1;"
                "if(gc+2<N)Cout[gr*N+gc+2]=v2;"
                "if(gc+3<N)Cout[gr*N+gc+3]=v3;"
            )
        write_macro = (
            f"#define W(r,v0,v1,v2,v3) {{int gr=bm+tr+(r);"
            f"if(gr<M){{int gc=bn+tc;float*Cout={c_name}; "
            f"{w_body}}}}}\n"
            f"    {w_calls}"
        )

        # --- Strategy-specific: K-loop body ---
        if strategy == "tma_db":
            # TMA double-buffered K-loop: smem setup + mbarrier pipeline.
            tma_a_ref = f"&{a_name}_tma"
            tma_b_ref = f"&{b_name}_tma"
            smem_bytes = stage * 4
            first_k = "k_start" if k_splits > 1 else "0"
            next_k_prefix = "k_start+" if k_splits > 1 else ""

            if k_splits > 1:
                k_range_code = (
                    f"int k_per_split=(K/{bk}/k_splits)*{bk};\n"
                    f"int k_start=blockIdx.z*k_per_split;\n"
                    f"int k_end=(blockIdx.z==k_splits-1)?K:k_start+k_per_split;\n"
                    f"int p0=0,p1=0,nt=(k_end-k_start)/{bk};\n"
                )
            else:
                k_range_code = f"int p0=0,p1=0,nt=K/{bk};\n"

            tma_code = (
                f"extern __shared__ __align__(128) char dsmem[];\n"
                f"float*smem=(float*)dsmem;\n"
                f"uint64_t*mbar=(uint64_t*)(dsmem+2*{stage}*4);\n"
                f"const int as0=(int)__cvta_generic_to_shared(&smem[0]);\n"
                f"const int bs0=(int)__cvta_generic_to_shared(&smem[{a_size}]);\n"
                f"const int as1=(int)__cvta_generic_to_shared(&smem[{stage}]);\n"
                f"const int bs1=(int)__cvta_generic_to_shared(&smem[{stage}+{a_size}]);\n"
                f"const int mb0=(int)__cvta_generic_to_shared(&mbar[0]);\n"
                f"const int mb1=(int)__cvta_generic_to_shared(&mbar[1]);\n"
                f"int tid=threadIdx.y*{tx}+threadIdx.x;\n"
                f'if(tid==0){{asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb0),"r"(1));'
                f'asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(mb1),"r"(1));'
                f'asm volatile("fence.mbarrier_init.release.cluster;");}}\n'
                f"__syncthreads();\n"
                f"float {acc_decls};\n"
                f"const int bytes={smem_bytes};\n"
                f"{k_range_code}"
                f"if(nt>0&&tid==0){{"
                f'asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 '
                f'_,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");'
                f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
                f"::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];\""
                f'::"r"(as0),"l"({tma_a_ref}),"r"({first_k}),"r"(bm),"r"(mb0):"memory");'
                f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
                f"::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];\""
                f'::"r"(bs0),"l"({tma_b_ref}),"r"(bn),"r"({first_k}),"r"(mb0):"memory");}}\n'
                f"for(int t=0;t<nt;t++){{\n"
                f"    int s=t%2;int cm=s==0?mb0:mb1;int cp=s==0?p0:p1;\n"
                f"    int nm=s==0?mb1:mb0;int na=s==0?as1:as0;int nb=s==0?bs1:bs0;\n"
                f'    asm volatile("{{\\n\\t.reg .pred P1;\\n\\tLW:\\n\\t'
                f"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0],%1,%2;\\n\\t"
                f'@P1 bra.uni LD;\\n\\tbra.uni LW;\\n\\tLD:\\n\\t}}"::"r"(cm),"r"(cp),"r"(0xffffffff));\n'
                f"    if(s==0)p0^=1;else p1^=1;\n"
                f"    if(tid==0&&t+1<nt){{int nk={next_k_prefix}(t+1)*{bk};"
                f'asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 '
                f'_,[%0],%1;"::"r"(nm),"r"(bytes):"memory");'
                f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
                f"::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];\""
                f'::"r"(na),"l"({tma_a_ref}),"r"(nk),"r"(bm),"r"(nm):"memory");'
                f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
                f"::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];\""
                f'::"r"(nb),"l"({tma_b_ref}),"r"(bn),"r"(nk),"r"(nm):"memory");}}\n'
                f"    float*cas=&smem[s*{stage}];float*cbs=&smem[s*{stage}+{a_size}];\n"
                f"    #pragma unroll\n"
                f"    for(int kk=0;kk<{bk};kk++){{\n"
                f"            float b0=cbs[kk*{tile_n}+tc],b1=cbs[kk*{tile_n}+tc+1],"
                f"b2=cbs[kk*{tile_n}+tc+2],b3=cbs[kk*{tile_n}+tc+3];\n"
                f"            {fma_block}\n"
                f"    }}\n"
                f"    __syncthreads();\n"
                f"}}\n"
                f"{write_macro}"
            )
            body.append(RawCode(tma_code))

            # TMA kernel: only C (output) as regular param.
            # A/B come via TMA descriptors; M/N/K come via #define from backend.
            tma_exclude = {a_name, b_name, "M", "N", "K"}
            params = [p for p in params if p.name not in tma_exclude]
            if k_splits > 1:
                params.append(KernelParam("int", "k_splits"))

            return KernelDef(
                name=name, params=params, body=body, block_size=block_size,
                includes=["cuda.h"],
                tile_m=tile_m, tile_n=tile_n,
                tma_params=[f"{a_name}_tma", f"{b_name}_tma"],
            )
        else:
            # Naive K-loop: load directly from global memory.
            # Bounds-safe B loads: zero for out-of-bounds columns.
            # A loads: zero for out-of-bounds rows.
            naive_a_lines = []
            for r in range(thread_m):
                naive_a_lines.append(
                    f"float a{r}=(bm+tr+{r}<M)?{a_name}[(bm+tr+{r})*K+k]:0.0f;"
                    + "".join(f"c{r}{c}+=a{r}*b{c};" for c in range(thread_n))
                )
            naive_fma_safe = "\n        ".join(naive_a_lines)

            body.append(RawCode(
                f"float {acc_decls};\n"
                f"for(int k=0;k<K;k++){{\n"
                f"    float b0=(bn+tc<N)?{b_name}[k*N+bn+tc]:0.0f,"
                f"b1=(bn+tc+1<N)?{b_name}[k*N+bn+tc+1]:0.0f,"
                f"b2=(bn+tc+2<N)?{b_name}[k*N+bn+tc+2]:0.0f,"
                f"b3=(bn+tc+3<N)?{b_name}[k*N+bn+tc+3]:0.0f;\n"
                f"        {naive_fma_safe}\n"
                f"}}\n"
                f"{write_macro}"
            ))
            return KernelDef(
                name=name, params=params, body=body, block_size=block_size,
                tile_m=tile_m, tile_n=tile_n,
            )

    # --- Row reduction / reduce+broadcast ---
    # Tile loop over columns
    loop_body: list = []
    loop_body.extend(_emit_ops(phases.prologue, var_map, "p_"))

    # Accumulation
    for node_id, _op, input_ids in phases.reduces:
        acc_name, fn = reduce_vars[node_id]
        val = var_map.get(input_ids[0], Literal(0.0))
        if fn == "sum":
            loop_body.append(AugAssign(acc_name, "+=", val))
        else:
            # max/min: acc = fmaxf(acc, val) — no AugAssign for this
            val_str = _expr_to_str(val)
            loop_body.append(RawCode(
                f"{acc_name} = fmaxf({acc_name}, {val_str});"
            ))

    body.append(ForLoop("j", CudaBuiltin("threadIdx.x"), Var("cols"), loop_body, step=CudaBuiltin("blockDim.x")))

    # --- Phase 6: Cross-thread warp shuffle ---
    for node_id, (acc_name, fn) in reduce_vars.items():
        body.extend(_emit_warp_reduce(node_id, acc_name, fn))
        var_map[node_id] = Var(acc_name)

    # --- Phase 7: Epilogue ---
    epilogue_ops = phases.epilogue
    if epilogue_ops:
        if analysis.epilogue_needs_per_element:
            # Second pass over columns: re-read inputs, re-compute needed prologue, emit epilogue
            epi_body: list = []

            # Re-map inputs for the epilogue loop
            epi_var_map: dict[str, Expr] = dict(var_map)
            for inp in region.input_names:
                epi_var_map[inp] = _input_expr(inp, analysis, "j", out_size)

            # Re-compute prologue ops needed by epilogue
            needed = _needed_by(epilogue_ops)
            for node_id, op, input_ids in phases.prologue:
                if isinstance(op, ElementwiseOp) and node_id in needed:
                    a = epi_var_map.get(input_ids[0], Literal(0.0)) if input_ids else Literal(0.0)
                    b = epi_var_map.get(input_ids[1], Literal(0.0)) if len(input_ids) > 1 else Literal(0.0)
                    expr = _build_expr(op.fn, a, b)
                    if expr is None:
                        expr = Literal(0.0)
                    vn = f"p_{_safe(node_id)}"
                    epi_body.append(VarDecl("float", vn, expr))
                    epi_var_map[node_id] = Var(vn)

            epi_body.extend(_emit_ops(epilogue_ops, epi_var_map, "e_"))

            for out_id in region.output_names:
                val = epi_var_map.get(out_id, Literal(0.0))
                idx = BinOp("+", BinOp("*", Var("row"), Var("cols")), Var("j"))
                epi_body.append(Assign(ArrayAccess(_safe(out_id), idx), val))

            body.append(ForLoop("j", CudaBuiltin("threadIdx.x"), Var("cols"), epi_body, step=CudaBuiltin("blockDim.x")))
        else:
            # Per-row epilogue: only thread 0 writes scalar output
            body.extend(_emit_ops(epilogue_ops, var_map, "e_"))
            write_body = []
            for out_id in region.output_names:
                val = var_map.get(out_id, Literal(0.0))
                write_body.append(Assign(ArrayAccess(_safe(out_id), Var("row")), val))
            body.append(IfStmt(BinOp("==", CudaBuiltin("threadIdx.x"), Literal(0, "int")), write_body))
    else:
        # No epilogue — write reduce result (thread 0 only)
        write_body = []
        for out_id in region.output_names:
            val = var_map.get(out_id, Literal(0.0))
            write_body.append(Assign(ArrayAccess(_safe(out_id), Var("row")), val))
        body.append(IfStmt(BinOp("==", CudaBuiltin("threadIdx.x"), Literal(0, "int")), write_body))

    return KernelDef(name=name, params=params, body=body, block_size=block_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_epilogue_inline(
    region: FusedRegionOp,
    analysis: TileAnalysis,
    phases,
    var_map: dict[str, Expr],
    reduce_vars: dict[str, tuple[str, str]],
) -> list:
    """Build epilogue statements for contraction (inline after k-loop)."""
    epilogue_ops = phases.epilogue
    if not epilogue_ops:
        return []

    stmts = []

    # Map reduce results to acc
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = Var(acc_name)

    # Map external inputs for epilogue
    for inp in region.input_names:
        var_map[inp] = _contraction_input_expr(inp, analysis)

    stmts.extend(_emit_ops(epilogue_ops, var_map, "e_"))
    return stmts


def _expr_to_str(expr: Expr) -> str:
    """Quick string conversion for an Expr (used in RawCode fallbacks)."""
    from deplodock.compiler.backend.cuda.codegen import _emit_expr

    return _emit_expr(expr)


