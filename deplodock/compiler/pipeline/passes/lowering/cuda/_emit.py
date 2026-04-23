"""Kernel IR to C source code generation.

Backend-agnostic printer: walks the AST from ``compiler/ir/kernel/ir.py`` and
emits C/C++ text. Usable for both CUDA and HIP targets.
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinOp,
    Builtin,
    Cast,
    FieldAccess,
    ForLoop,
    FuncCall,
    GpuExpr,
    GpuKernel,
    IfStmt,
    Literal,
    PragmaUnroll,
    RawCode,
    Stmt,
    SyncThreads,
    Ternary,
    Var,
    VarAssign,
    VarDecl,
    VectorLoad,
)

# ---------------------------------------------------------------------------
# Backend-neutral → CUDA spelling translation tables.
#
# Kernel IR is backend-neutral: ``Builtin("thread_idx.x")``, ``FuncCall("rsqrt", ...)``,
# ``ArrayDecl(storage="shared", ...)``. The CUDA emitter is the only layer that
# knows CUDA's spellings — every other pass operates on neutral names.
# ---------------------------------------------------------------------------

_BUILTIN_TO_CUDA: dict[str, str] = {
    "thread_idx.x": "threadIdx.x",
    "thread_idx.y": "threadIdx.y",
    "thread_idx.z": "threadIdx.z",
    "block_idx.x": "blockIdx.x",
    "block_idx.y": "blockIdx.y",
    "block_idx.z": "blockIdx.z",
    "block_dim.x": "blockDim.x",
    "block_dim.y": "blockDim.y",
    "block_dim.z": "blockDim.z",
    "grid_dim.x": "gridDim.x",
    "grid_dim.y": "gridDim.y",
    "grid_dim.z": "gridDim.z",
    "warp_size": "warpSize",
}

_INTRINSIC_TO_CUDA: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "abs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
}


def _translate_builtin(name: str) -> str:
    """Neutral builtin name → CUDA spelling. Passthrough for names already in CUDA form."""
    return _BUILTIN_TO_CUDA.get(name, name)


def _translate_intrinsic(name: str) -> str:
    """Neutral intrinsic name → CUDA libm spelling. Passthrough for names already in CUDA form."""
    return _INTRINSIC_TO_CUDA.get(name, name)


# Operator precedence for minimal parenthesization.
_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    ">": 4,
    "<=": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


def emit_kernel(kernel: GpuKernel) -> str:
    """Emit complete CUDA C source for a __global__ kernel."""
    preamble = ""
    if kernel.includes:
        preamble = "\n".join(f"#include <{h}>" for h in kernel.includes) + "\n"
        if "mma.h" in kernel.includes:
            preamble += "using namespace nvcuda;\n"
        preamble += "\n"
    params = ", ".join(f"{p.dtype} {p.name}" for p in kernel.params)
    body = "\n".join(_emit_stmt(s, indent=1) for s in kernel.body)
    bx, by, bz = kernel.block_size
    max_threads = bx * by * bz
    min_blocks = getattr(kernel, "min_blocks_per_sm", 0)
    if max_threads <= 1024:
        if min_blocks > 0:
            launch_bounds = f"\n__launch_bounds__({max_threads}, {min_blocks})"
        else:
            launch_bounds = f"\n__launch_bounds__({max_threads})"
    else:
        launch_bounds = ""
    return f'{preamble}extern "C" __global__{launch_bounds} void {kernel.name}({params}) {{\n{body}\n}}\n'


def _emit_expr(expr: GpuExpr, parent_prec: int = 0) -> str:
    """Emit an expression as C code."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        if isinstance(expr.value, float) or expr.dtype == "float":
            return f"{float(expr.value):.1f}f"
        v = int(expr.value)
        # Use LL suffix for large constants to prevent 32-bit overflow
        # in index arithmetic (e.g., stride * idx for large tensors).
        return f"{v}LL" if abs(v) > 32768 else str(v)
    if isinstance(expr, BinOp):
        prec = _PRECEDENCE.get(expr.op, 10)
        left = _emit_expr(expr.left, prec)
        right = _emit_expr(expr.right, prec + 1)
        result = f"{left} {expr.op} {right}"
        if prec < parent_prec:
            return f"({result})"
        return result
    if isinstance(expr, ArrayAccess):
        idx = _emit_expr(expr.index)
        return f"{expr.array}[{idx}]"
    if isinstance(expr, Builtin):
        return _translate_builtin(expr.name)
    if isinstance(expr, FuncCall):
        args = ", ".join(_emit_expr(a) for a in expr.args)
        return f"{_translate_intrinsic(expr.name)}({args})"
    if isinstance(expr, Cast):
        inner = _emit_expr(expr.expr)
        return f"(({expr.dtype})({inner}))"
    if isinstance(expr, FieldAccess):
        inner = _emit_expr(expr.expr)
        return f"{inner}.{expr.field}"
    if isinstance(expr, Ternary):
        cond = _emit_expr(expr.cond)
        t = _emit_expr(expr.if_true)
        f = _emit_expr(expr.if_false)
        return f"(({cond}) ? ({t}) : ({f}))"
    if isinstance(expr, VectorLoad):
        idx = _emit_expr(expr.index)
        return f"*reinterpret_cast<const float{expr.width}*>(&{expr.array}[{idx}])"
    raise TypeError(f"Unknown expression type: {type(expr)}")


def _emit_stmt(stmt: Stmt, indent: int) -> str:
    """Emit a statement with indentation."""
    pad = "    " * indent

    if isinstance(stmt, VarDecl):
        if stmt.init is not None:
            init = _emit_expr(stmt.init)
            return f"{pad}{stmt.dtype} {stmt.name} = {init};"
        return f"{pad}{stmt.dtype} {stmt.name};"

    if isinstance(stmt, Assign):
        target = _emit_expr(stmt.target)
        value = _emit_expr(stmt.value)
        return f"{pad}{target} = {value};"

    if isinstance(stmt, VarAssign):
        value = _emit_expr(stmt.value)
        return f"{pad}{stmt.name} = {value};"

    if isinstance(stmt, AugAssign):
        value = _emit_expr(stmt.value)
        return f"{pad}{stmt.target} {stmt.op} {value};"

    if isinstance(stmt, ForLoop):
        start = _emit_expr(stmt.start)
        end = _emit_expr(stmt.end)
        body = "\n".join(_emit_stmt(s, indent + 1) for s in stmt.body)
        if stmt.step is not None:
            step = _emit_expr(stmt.step)
            return f"{pad}for (int {stmt.var} = {start}; {stmt.var} < {end}; {stmt.var} += {step}) {{\n{body}\n{pad}}}"
        return f"{pad}for (int {stmt.var} = {start}; {stmt.var} < {end}; {stmt.var}++) {{\n{body}\n{pad}}}"

    if isinstance(stmt, IfStmt):
        cond = _emit_expr(stmt.cond)
        body = "\n".join(_emit_stmt(s, indent + 1) for s in stmt.body)
        if stmt.else_body:
            else_body = "\n".join(_emit_stmt(s, indent + 1) for s in stmt.else_body)
            return f"{pad}if ({cond}) {{\n{body}\n{pad}}} else {{\n{else_body}\n{pad}}}"
        return f"{pad}if ({cond}) {{\n{body}\n{pad}}}"

    if isinstance(stmt, SyncThreads):
        return f"{pad}__syncthreads();"

    if isinstance(stmt, ArrayDecl):
        dims = "".join(f"[{d}]" for d in stmt.dimensions)
        storage = "__shared__ " if getattr(stmt, "storage", "local") == "shared" else ""
        if stmt.init is not None:
            init = _emit_expr(stmt.init)
            return f"{pad}{storage}{stmt.dtype} {stmt.name}{dims} = {init};"
        return f"{pad}{storage}{stmt.dtype} {stmt.name}{dims};"

    if isinstance(stmt, PragmaUnroll):
        if stmt.factor is not None:
            return f"{pad}#pragma unroll {stmt.factor}"
        return f"{pad}#pragma unroll"

    if isinstance(stmt, RawCode):
        # Indent each line of the raw code.
        lines = stmt.code.splitlines()
        return "\n".join(pad + line for line in lines)

    raise TypeError(f"Unknown statement type: {type(stmt)}")
