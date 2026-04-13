"""LoopIR to KernelDef codegen.

Mechanically translates a LoopProgram into a KernelDef AST.  Each LoopOp
maps to a small number of KernelDef statements.  No pattern-matching or
analysis — all structural decisions were made during lowering.
"""

from __future__ import annotations

from deplodock.compiler.backend.kernel_ir import (
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
    Stmt,
    SyncThreads,
    Ternary,
    Var,
    VarAssign,
    VarDecl,
)
from deplodock.compiler.backend.loop_ir import (
    Accumulate,
    Alloc,
    Barrier,
    Compute,
    Guard,
    Load,
    LoopBinOp,
    LoopBuiltin,
    LoopExpr,
    LoopFuncCall,
    LoopLiteral,
    LoopNest,
    LoopOp,
    LoopProgram,
    LoopTernary,
    LoopVar,
    ParallelAxis,
    RawLoopOp,
    RegAccess,
    Store,
    WarpReduce,
    WarpShuffleXor,
)


def loop_ir_to_kernel(program: LoopProgram) -> KernelDef:
    """Translate a LoopProgram into a KernelDef."""
    params = [KernelParam(dtype, name) for dtype, name in program.params]
    body = _lower_ops(program.body)

    return KernelDef(
        name=program.name,
        params=params,
        body=body,
        block_size=program.block_size,
        includes=program.includes,
        tile_m=program.tile_m,
        tile_n=program.tile_n,
        grid_2d=program.grid_2d,
        tma_params=program.tma_params,
        batched=program.batched,
        extra_smem_bytes=program.extra_smem_bytes,
        min_blocks_per_sm=program.min_blocks_per_sm,
    )


# ---------------------------------------------------------------------------
# Expression lowering
# ---------------------------------------------------------------------------


def _lower_expr(expr: LoopExpr) -> Expr:
    """Convert a LoopExpr to a KernelDef Expr."""
    if isinstance(expr, LoopVar):
        return Var(expr.name)
    if isinstance(expr, LoopLiteral):
        return Literal(expr.value, expr.dtype)
    if isinstance(expr, LoopBinOp):
        return BinOp(expr.op, _lower_expr(expr.left), _lower_expr(expr.right))
    if isinstance(expr, LoopBuiltin):
        return CudaBuiltin(expr.name)
    if isinstance(expr, LoopFuncCall):
        return FuncCall(expr.name, [_lower_expr(a) for a in expr.args])
    if isinstance(expr, LoopTernary):
        return Ternary(_lower_expr(expr.cond), _lower_expr(expr.if_true), _lower_expr(expr.if_false))
    if isinstance(expr, RegAccess):
        # Expand to scalar variable name: c[3][2] → "c32"
        return Var(expr.name + "".join(str(i) for i in expr.indices))
    msg = f"Unknown LoopExpr type: {type(expr)}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Op lowering
# ---------------------------------------------------------------------------

# CUDA C expression templates for elementwise ops.
# {a} and {b} are placeholders for lowered operand expressions.
_C_EXPR: dict[str, str] = {
    "add": "{a} + {b}",
    "sub": "{a} - {b}",
    "mul": "{a} * {b}",
    "div": "{a} / {b}",
    "mod": "{a} % {b}",
    "neg": "-{a}",
    "exp": "expf({a})",
    "rsqrt": "rsqrtf({a})",
    "recip": "1.0f / {a}",
    "relu": "fmaxf(0.0f, {a})",
    "tanh": "tanhf({a})",
    "sigmoid": "1.0f / (1.0f + expf(-{a}))",
    # Identity-like: used for builtin aliases (e.g. batch = blockIdx.z)
    "builtin": "{a}",
}


def _render_c_expr(op_name: str, a: str, b: str = "0.0f") -> str:
    """Render a C expression from the template for an elementwise op.

    Binary op arguments are parenthesized to preserve precedence.
    """
    from deplodock.compiler.ops import _DEFAULT_OP_INFO, OP_REGISTRY

    tmpl = _C_EXPR.get(op_name)
    if tmpl is None:
        return f"/* unknown op: {op_name} */ 0.0f"
    info = OP_REGISTRY.get(op_name, _DEFAULT_OP_INFO)
    if info.arity == 2:
        return tmpl.format(a=f"({a})", b=f"({b})")
    return tmpl.format(a=a, b=b)


def _lower_ops(ops: list[LoopOp]) -> list[Stmt]:
    """Lower a list of LoopOps to KernelDef statements."""
    stmts: list[Stmt] = []
    for op in ops:
        stmts.extend(_lower_op(op))
    return stmts


def _lower_op(op: LoopOp) -> list[Stmt]:
    """Lower a single LoopOp to KernelDef statements."""

    if isinstance(op, ParallelAxis):
        # ParallelAxis("i", "blockIdx.x", "n") →
        #   int i = blockIdx.x * blockDim.x + threadIdx.x;
        # ParallelAxis("row", "blockIdx.x", "rows") →
        #   int row = 1 * blockIdx.x;
        #
        # The exact mapping depends on the axis name convention.
        # For 1D pointwise: i = blockIdx.x * blockDim.x + threadIdx.x
        # For row-parallel: row = blockIdx.x (one block per row)
        if op.name == "i":
            # 1D thread-to-element mapping
            return [
                VarDecl(
                    "int",
                    op.name,
                    BinOp(
                        "+",
                        BinOp("*", CudaBuiltin("blockIdx.x"), CudaBuiltin("blockDim.x")),
                        CudaBuiltin("threadIdx.x"),
                    ),
                )
            ]
        # Row-parallel: one block per row
        return [VarDecl("int", op.name, BinOp("*", Literal(1, "int"), CudaBuiltin(op.dim)))]

    if isinstance(op, LoopNest):
        body = _lower_ops(op.body)
        return [
            ForLoop(
                op.var,
                _lower_expr(op.start),
                _lower_expr(op.end),
                body,
                step=_lower_expr(op.step) if op.step else None,
            )
        ]

    if isinstance(op, Alloc):
        if op.space == "smem":
            dims = list(op.shape) if op.shape else [1]
            return [ArrayDecl(f"__shared__ {op.dtype}", op.name, dims)]
        # Register allocation
        init = _lower_expr(op.init) if op.init else None
        if op.shape:
            # Register array: Alloc("c", "float", (8, 4), "reg", init=0.0)
            # → "float c00=0.0f,c01=0.0f,...,c73=0.0f;"
            import itertools

            indices = list(itertools.product(*(range(d) for d in op.shape)))
            names = [op.name + "".join(str(i) for i in idx) for idx in indices]
            init_str = _expr_to_c(init) if init else "0.0f"
            decl = ",".join(f"{n}={init_str}" for n in names)
            return [RawCode(f"{op.dtype} {decl};")]
        return [VarDecl(op.dtype, op.name, init)]

    if isinstance(op, Load):
        idx = _lower_expr(op.indices)
        if op.guard:
            guard = _lower_expr(op.guard)
            # Guarded load: type dst = guard ? src[idx] : 0.0f;
            return [
                VarDecl(
                    "float",
                    op.dst,
                    Ternary(guard, ArrayAccess(op.src, idx), Literal(0.0)),
                )
            ]
        return [VarDecl("float", op.dst, ArrayAccess(op.src, idx))]

    if isinstance(op, Store):
        idx = _lower_expr(op.indices)
        val = _lower_expr(op.value)
        if op.guard:
            guard = _lower_expr(op.guard)
            if op.atomic:
                val_str = _expr_to_c(val)
                return [
                    IfStmt(
                        guard,
                        [RawCode(f"atomicAdd(&{op.dst}[{_expr_to_c(idx)}], {val_str});")],
                    )
                ]
            return [IfStmt(guard, [Assign(ArrayAccess(op.dst, idx), val)])]
        if op.atomic:
            val_str = _expr_to_c(val)
            return [RawCode(f"atomicAdd(&{op.dst}[{_expr_to_c(idx)}], {val_str});")]
        return [Assign(ArrayAccess(op.dst, idx), val)]

    if isinstance(op, Compute):
        # Check if this is an in-place update: if any arg (recursively)
        # references dst, emit assignment instead of declaration.
        is_inplace = _refs_name(op.args, op.dst)
        dtype = op.dtype
        args_lowered = [_lower_expr(a) for a in op.args]
        a_str = _expr_to_c(args_lowered[0]) if args_lowered else "0.0f"
        b_str = _expr_to_c(args_lowered[1]) if len(args_lowered) > 1 else "0.0f"
        c_code = _render_c_expr(op.op, a_str, b_str)
        if is_inplace:
            return [RawCode(f"{op.dst} = {c_code};")]
        return [RawCode(f"{dtype} {op.dst} = {c_code};")]

    if isinstance(op, Accumulate):
        val = _lower_expr(op.value)
        if op.op == "sum":
            return [AugAssign(op.dst, "+=", val)]
        # max
        val_str = _expr_to_c(val)
        return [RawCode(f"{op.dst} = fmaxf({op.dst}, {val_str});")]

    if isinstance(op, WarpReduce):
        return _emit_warp_reduce(op.var, op.op)

    if isinstance(op, WarpShuffleXor):
        return _emit_warp_shuffle_xor(op.var, op.op)

    if isinstance(op, Barrier):
        return [SyncThreads()]

    if isinstance(op, Guard):
        if not op.body:
            # Empty body = early return (used for grid bounds checks)
            return [IfStmt(_lower_expr(op.cond), [RawCode("return;")])]
        body = _lower_ops(op.body)
        return [IfStmt(_lower_expr(op.cond), body)]

    if isinstance(op, RawLoopOp):
        return [RawCode(op.code)]

    # CUDA-specific extension: TMA ops
    from deplodock.compiler.backend.cuda.tma_ops import TMAKLoop

    if isinstance(op, TMAKLoop):
        return [RawCode(_emit_tma_k_loop(op))]

    msg = f"Unknown LoopOp type: {type(op)}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Warp-shuffle reduction (produces KernelDef statements)
# ---------------------------------------------------------------------------


def _emit_warp_reduce(acc_var: str, fn: str) -> list[Stmt]:
    """Emit warp-shuffle + cross-warp shared memory reduction.

    Uses KernelIR AST nodes wherever possible.  The only remaining RawCode
    is the ``for (offset >>= 1)`` loop which has a non-standard step that
    KernelIR's ForLoop can't express.
    """
    acc = Var(acc_var)
    init_val = Literal(0.0) if fn == "sum" else Literal(-1e30)
    shfl = FuncCall("__shfl_down_sync", [Literal(0xFFFFFFFF, "int"), acc, Var("offset")])
    warp_arr = f"warp_{acc_var}"
    s_var = f"s_{acc_var}"
    tid = CudaBuiltin("threadIdx.x")
    warp_size = CudaBuiltin("warpSize")
    block_dim = CudaBuiltin("blockDim.x")

    # Shuffle body: acc += shfl (sum) or acc = fmaxf(acc, shfl) (max)
    if fn == "sum":
        shfl_stmt: Stmt = AugAssign(acc_var, "+=", shfl)
    else:
        shfl_stmt = VarAssign(acc_var, FuncCall("fmaxf", [acc, shfl]))

    # Shuffle loop: for (int offset = warpSize/2; offset > 0; offset >>= 1)
    # KernelIR ForLoop can't express >>= step, so this stays as RawCode.
    def _shfl_loop() -> RawCode:
        body_c = _expr_to_c(shfl_stmt.value) if isinstance(shfl_stmt, VarAssign) else _expr_to_c(shfl)
        if fn == "sum":
            return RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {acc_var} += {body_c};")
        return RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {acc_var} = {body_c};")

    stmts: list[Stmt] = []

    # 1. Intra-warp shuffle
    stmts.append(_shfl_loop())

    # 2. Cross-warp: lane 0 of each warp writes to shared array
    stmts.append(ArrayDecl("__shared__ float", warp_arr, [8]))
    stmts.append(
        IfStmt(
            BinOp("==", BinOp("%", tid, warp_size), Literal(0, "int")),
            [Assign(ArrayAccess(warp_arr, BinOp("/", tid, warp_size)), acc)],
        )
    )
    stmts.append(SyncThreads())

    # 3. First warp loads partial results; others get init value
    stmts.append(
        IfStmt(
            BinOp("<", tid, BinOp("/", block_dim, warp_size)),
            [VarAssign(acc_var, ArrayAccess(warp_arr, tid))],
            else_body=[VarAssign(acc_var, init_val)],
        )
    )

    # 4. Second shuffle pass (across warps)
    stmts.append(_shfl_loop())

    # 5. Broadcast final result to all threads via shared scalar
    stmts.append(VarDecl("__shared__ float", s_var))
    stmts.append(
        IfStmt(
            BinOp("==", tid, Literal(0, "int")),
            [VarAssign(s_var, acc)],
        )
    )
    stmts.append(SyncThreads())
    stmts.append(VarAssign(acc_var, Var(s_var)))

    return stmts


def _emit_tma_k_loop(op) -> str:
    """Generate the TMA double-buffered K-loop C code from structured TMAKLoop parameters."""
    bk = op.block_k
    a_size = op.a_size
    stage = op.stage
    tile_n = op.tile_n
    tx = op.tx
    thread_m = op.thread_m
    thread_n = op.thread_n
    k_splits = op.k_splits
    smem_bytes = stage * 4

    first_k = "k_start" if k_splits > 1 else "0"
    next_k_prefix = "k_start+" if k_splits > 1 else ""

    if k_splits > 1:
        k_range = (
            f"int k_per_split=(K/{bk}/k_splits)*{bk};\n"
            f"int k_start=blockIdx.z*k_per_split;\n"
            f"int k_end=(blockIdx.z==k_splits-1)?K:k_start+k_per_split;\n"
            f"int p0=0,p1=0,nt=(k_end-k_start)/{bk};"
        )
    else:
        k_range = f"int p0=0,p1=0,nt=K/{bk};"

    # FMA block: unrolled over thread_m x thread_n
    fma_lines = []
    for r in range(thread_m):
        fma_lines.append(f"float a{r}=cas[(tr+{r})*{bk}+kk];" + "".join(f"c{r}{c}+=a{r}*b{c};" for c in range(thread_n)))
    fma_block = "\n            ".join(fma_lines)

    # Accumulator declarations (TMA code references them inline)
    acc_decls = ",".join(f"c{r}{c}=0" for r in range(thread_m) for c in range(thread_n))

    a_ref = op.a_tma_ref
    b_ref = op.b_tma_ref

    return (
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
        f"{k_range}\n"
        f"if(nt>0&&tid==0){{"
        f'asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 '
        f'_,[%0],%1;"::"r"(mb0),"r"(bytes):"memory");'
        f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
        f'::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"'
        f'::"r"(as0),"l"({a_ref}),"r"({first_k}),"r"(bm),"r"(mb0):"memory");'
        f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
        f'::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"'
        f'::"r"(bs0),"l"({b_ref}),"r"(bn),"r"({first_k}),"r"(mb0):"memory");}}\n'
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
        f'::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"'
        f'::"r"(na),"l"({a_ref}),"r"(nk),"r"(bm),"r"(nm):"memory");'
        f'asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier'
        f'::complete_tx::bytes [%0],[%1,{{%2,%3}}],[%4];"'
        f'::"r"(nb),"l"({b_ref}),"r"(bn),"r"(nk),"r"(nm):"memory");}}\n'
        f"    float*cas=&smem[s*{stage}];float*cbs=&smem[s*{stage}+{a_size}];\n"
        f"    #pragma unroll\n"
        f"    for(int kk=0;kk<{bk};kk++){{\n"
        f"            float b0=cbs[kk*{tile_n}+tc],b1=cbs[kk*{tile_n}+tc+1],"
        f"b2=cbs[kk*{tile_n}+tc+2],b3=cbs[kk*{tile_n}+tc+3];\n"
        f"            {fma_block}\n"
        f"    }}\n"
        f"    __syncthreads();\n"
        f"}}"
    )


def _refs_name(exprs: list, name: str) -> bool:
    """Check if any LoopExpr in exprs (recursively) references the given name."""
    for e in exprs:
        if isinstance(e, LoopVar) and e.name == name:
            return True
        if isinstance(e, RegAccess) and e.name + "".join(str(i) for i in e.indices) == name:
            return True
        if isinstance(e, LoopFuncCall) and _refs_name(e.args, name):
            return True
        if isinstance(e, LoopBinOp) and _refs_name([e.left, e.right], name):
            return True
        if isinstance(e, LoopTernary) and _refs_name([e.cond, e.if_true, e.if_false], name):
            return True
    return False


def _emit_warp_shuffle_xor(acc_var: str, fn: str) -> list[Stmt]:
    """Emit horizontal warp shuffle reduction using __shfl_xor_sync.

    Unlike WarpReduce (vertical __shfl_down_sync across all threads in a
    block), this reduces within a warp using XOR lane masks.  Used for
    in-register softmax where each thread holds different columns.
    """
    if fn == "sum":
        body = f"{acc_var} += __shfl_xor_sync(0xffffffff, {acc_var}, o);"
    else:
        body = f"{acc_var} = fmaxf({acc_var}, __shfl_xor_sync(0xffffffff, {acc_var}, o));"
    return [RawCode(f"for(int o=16;o>0;o>>=1){body}")]


# ---------------------------------------------------------------------------
# C expression helper (for RawCode emission)
# ---------------------------------------------------------------------------


def _expr_to_c(expr: Expr) -> str:
    """Quick-and-dirty Expr to C string for use in RawCode."""
    from deplodock.compiler.backend.codegen import _emit_expr

    return _emit_expr(expr)
