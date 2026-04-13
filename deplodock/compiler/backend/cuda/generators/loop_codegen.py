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
    "neg": lambda a, _b: BinOp("-", Literal(0.0), a),
    "exp": lambda a, _b: FuncCall("expf", [a]),
    "rsqrt": lambda a, _b: FuncCall("rsqrtf", [a]),
    "recip": lambda a, _b: BinOp("/", Literal(1.0), a),
    "relu": lambda a, _b: FuncCall("fmaxf", [Literal(0.0), a]),
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
        builder = _ELEM_OPS.get(op.op)
        if builder:
            args_lowered = [_lower_expr(a) for a in op.args]
            a = args_lowered[0]
            b = args_lowered[1] if len(args_lowered) > 1 else Literal(0.0)
            expr = builder(a, b)
            if is_inplace:
                return [RawCode(f"{op.dst} = {_expr_to_c(expr)};")]
            return [VarDecl("float", op.dst, expr)]
        if is_inplace:
            return [RawCode(f"{op.dst} = 0.0f;")]
        return [VarDecl("float", op.dst, Literal(0.0))]

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

    Identical logic to tiled.py::_emit_warp_reduce, producing the same
    KernelDef AST.
    """
    init_val = 0.0 if fn == "sum" else -1e30

    stmts: list[Stmt] = []

    # Intra-warp shuffle
    if fn == "sum":
        shfl_body = f"{acc_var} += __shfl_down_sync(0xffffffff, {acc_var}, offset);"
    else:
        shfl_body = f"{acc_var} = fmaxf({acc_var}, __shfl_down_sync(0xffffffff, {acc_var}, offset));"
    stmts.append(RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {shfl_body}"))

    # Cross-warp via shared memory
    warp_arr = f"warp_{acc_var}"
    s_var = f"s_{acc_var}"
    stmts.append(ArrayDecl("__shared__ float", warp_arr, [8]))
    stmts.append(
        IfStmt(
            BinOp("==", BinOp("%", CudaBuiltin("threadIdx.x"), CudaBuiltin("warpSize")), Literal(0, "int")),
            [Assign(ArrayAccess(warp_arr, BinOp("/", CudaBuiltin("threadIdx.x"), CudaBuiltin("warpSize"))), Var(acc_var))],
        )
    )
    stmts.append(SyncThreads())

    # First warp loads partial results
    stmts.append(
        RawCode(f"if (threadIdx.x < blockDim.x / warpSize) {acc_var} = {warp_arr}[threadIdx.x];\nelse {acc_var} = {init_val:.1f}f;")
    )

    # Second shuffle pass
    stmts.append(RawCode(f"for (int offset = warpSize / 2; offset > 0; offset >>= 1)\n    {shfl_body}"))

    # Broadcast final result via shared scalar
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
# C expression helper (for RawCode emission)
# ---------------------------------------------------------------------------


def _expr_to_c(expr: Expr) -> str:
    """Quick-and-dirty Expr to C string for use in RawCode."""
    from deplodock.compiler.backend.codegen import _emit_expr

    return _emit_expr(expr)
