"""CUDA IR → CUDA C source code generation."""

from __future__ import annotations

from deplodock.compiler.cuda.ir import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinOp,
    Cast,
    CudaBuiltin,
    Expr,
    FieldAccess,
    ForLoop,
    FuncCall,
    IfStmt,
    KernelDef,
    Literal,
    PragmaUnroll,
    Stmt,
    SyncThreads,
    Ternary,
    Var,
    VarDecl,
    VectorLoad,
)

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


def emit_kernel(kernel: KernelDef) -> str:
    """Emit complete CUDA C source for a __global__ kernel."""
    params = ", ".join(f"{p.dtype} {p.name}" for p in kernel.params)
    body = "\n".join(_emit_stmt(s, indent=1) for s in kernel.body)
    return f"__global__ void {kernel.name}({params}) {{\n{body}\n}}\n"


def _emit_expr(expr: Expr, parent_prec: int = 0) -> str:
    """Emit an expression as C code."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        if isinstance(expr.value, float) or expr.dtype == "float":
            return f"{float(expr.value):.1f}f"
        return str(expr.value)
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
    if isinstance(expr, CudaBuiltin):
        return expr.name
    if isinstance(expr, FuncCall):
        args = ", ".join(_emit_expr(a) for a in expr.args)
        return f"{expr.name}({args})"
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
        return f"*reinterpret_cast<float{expr.width}*>(&{expr.array}[{idx}])"
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
        if stmt.init is not None:
            init = _emit_expr(stmt.init)
            return f"{pad}{stmt.dtype} {stmt.name}{dims} = {init};"
        return f"{pad}{stmt.dtype} {stmt.name}{dims};"

    if isinstance(stmt, PragmaUnroll):
        if stmt.factor is not None:
            return f"{pad}#pragma unroll {stmt.factor}"
        return f"{pad}#pragma unroll"

    raise TypeError(f"Unknown statement type: {type(stmt)}")
