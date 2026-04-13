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

# Map elementwise op names to C/CUDA expression builders.
_ELEM_OPS: dict[str, object] = {
    "mul": lambda a, b: BinOp("*", a, b),
    "add": lambda a, b: BinOp("+", a, b),
    "sub": lambda a, b: BinOp("-", a, b),
    "div": lambda a, b: BinOp("/", a, b),
    "mod": lambda a, b: BinOp("%", a, b),
    "neg": lambda a, _b: BinOp("-", Literal(0.0), a),
    "exp": lambda a, _b: FuncCall("expf", [a]),
    "rsqrt": lambda a, _b: FuncCall("rsqrtf", [a]),
    "recip": lambda a, _b: BinOp("/", Literal(1.0), a),
    "relu": lambda a, _b: FuncCall("fmaxf", [Literal(0.0), a]),
    # Identity-like: just returns the first arg (used for builtin aliases)
    "builtin": lambda a, _b: a,
}


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
        # Check if this is an in-place update on a register array element:
        # if any arg is a RegAccess whose expanded name matches dst, emit
        # assignment instead of declaration.
        is_inplace = any(isinstance(a, RegAccess) and a.name + "".join(str(i) for i in a.indices) == op.dst for a in op.args)
        dtype = op.dtype
        builder = _ELEM_OPS.get(op.op)
        if builder:
            args_lowered = [_lower_expr(a) for a in op.args]
            a = args_lowered[0]
            b = args_lowered[1] if len(args_lowered) > 1 else Literal(0.0)
            expr = builder(a, b)
            if is_inplace:
                return [VarAssign(op.dst, expr)]
            return [VarDecl(dtype, op.dst, expr)]
        if is_inplace:
            return [VarAssign(op.dst, Literal(0.0))]
        return [VarDecl(dtype, op.dst, Literal(0.0))]

    if isinstance(op, Accumulate):
        val = _lower_expr(op.value)
        if op.op == "sum":
            return [AugAssign(op.dst, "+=", val)]
        # max
        val_str = _expr_to_c(val)
        return [RawCode(f"{op.dst} = fmaxf({op.dst}, {val_str});")]

    if isinstance(op, WarpReduce):
        return _emit_warp_reduce(op.var, op.op)

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


# ---------------------------------------------------------------------------
# C expression helper (for RawCode emission)
# ---------------------------------------------------------------------------


def _expr_to_c(expr: Expr) -> str:
    """Quick-and-dirty Expr to C string for use in RawCode."""
    from deplodock.compiler.backend.codegen import _emit_expr

    return _emit_expr(expr)
