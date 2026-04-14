"""Kernel IR: imperative AST for GPU kernel code generation.

Backend-agnostic: the AST nodes represent generic C/C++ constructs
(variables, loops, array accesses, function calls) that can be emitted
as CUDA, HIP, or any C-like GPU language by a matching codegen printer.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.backend.ir.expr import (
    BinOp,
    Builtin,
    Expr,
    FuncCall,
    Literal,
    Ternary,
    Var,
    _ExprOps,
)

# Re-export shared types so existing ``from kernel_ir import Var`` still works.
__all__ = [
    # Shared expression types (re-exported)
    "Var",
    "Literal",
    "BinOp",
    "Builtin",
    "FuncCall",
    "Ternary",
    "Expr",
    "_ExprOps",
    # Kernel-specific expression types
    "ArrayAccess",
    "Cast",
    "FieldAccess",
    "VectorLoad",
    "KernelExpr",
    # Statements
    "VarDecl",
    "Assign",
    "VarAssign",
    "AugAssign",
    "ForLoop",
    "IfStmt",
    "SyncThreads",
    "ArrayDecl",
    "PragmaUnroll",
    "RawCode",
    "Stmt",
    # Kernel definition
    "KernelParam",
    "KernelDef",
    # Utilities
    "pretty_print",
]

# ---------------------------------------------------------------------------
# Kernel-specific expression types
# ---------------------------------------------------------------------------


@dataclass
class ArrayAccess(_ExprOps):
    """Array element access: array[index]."""

    array: str
    index: Expr


@dataclass
class Cast(_ExprOps):
    """Type cast: (dtype)(expr)."""

    dtype: str  # "float4", "int", etc.
    expr: Expr


@dataclass
class FieldAccess(_ExprOps):
    """Struct field access: expr.field (for float4.x, .y, .z, .w)."""

    expr: Expr
    field: str  # "x", "y", "z", "w"


@dataclass
class VectorLoad(_ExprOps):
    """Load N contiguous floats as a vector: *(floatN*)(&array[index]).

    Used for coalesced float2/float4 loads from global memory.
    """

    array: str
    index: Expr
    width: int = 4  # 2 or 4


KernelExpr = Expr | ArrayAccess | Cast | FieldAccess | VectorLoad


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass
class VarDecl:
    """Variable declaration with optional initializer."""

    dtype: str
    name: str
    init: KernelExpr | None = None


@dataclass
class Assign:
    """Array element assignment: target[index] = value."""

    target: ArrayAccess
    value: KernelExpr


@dataclass
class VarAssign:
    """Plain variable reassignment: name = value."""

    name: str
    value: KernelExpr


@dataclass
class AugAssign:
    """Augmented assignment: target op= value."""

    target: str
    op: str  # "+=", "-=", "*="
    value: KernelExpr


@dataclass
class ForLoop:
    """C-style for loop: for (int var = start; var < end; var += step)."""

    var: str
    start: KernelExpr
    end: KernelExpr
    body: list[Stmt]
    step: KernelExpr | None = None  # None = var++ (increment by 1)


@dataclass
class IfStmt:
    """If statement with optional else."""

    cond: KernelExpr
    body: list[Stmt]
    else_body: list[Stmt] | None = None


@dataclass
class SyncThreads:
    """__syncthreads() barrier."""


@dataclass
class ArrayDecl:
    """Fixed-size array declaration, e.g. __shared__ float tile[BK][BM]."""

    dtype: str  # "__shared__ float", "float", "float4"
    name: str
    dimensions: list[int]  # [64, 64] for 2D, [256] for 1D
    init: KernelExpr | None = None


@dataclass
class PragmaUnroll:
    """#pragma unroll [factor] before a for loop."""

    factor: int | None = None  # None = fully unroll


@dataclass
class RawCode:
    """Raw C/CUDA code injected verbatim."""

    code: str


Stmt = VarDecl | Assign | VarAssign | AugAssign | ForLoop | IfStmt | SyncThreads | ArrayDecl | PragmaUnroll | RawCode


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------


@dataclass
class KernelParam:
    """Kernel function parameter."""

    dtype: str  # "float*", "int", etc.
    name: str


@dataclass
class KernelDef:
    """__global__ kernel function."""

    name: str
    params: list[KernelParam]
    body: list[Stmt]
    block_size: tuple[int, int, int] = (16, 16, 1)
    includes: list[str] | None = None  # Extra #include headers
    tile_m: int | None = None  # Output tile rows per block (overrides grid computation)
    tile_n: int | None = None  # Output tile cols per block
    grid_2d: bool = False  # Use standard 2D grid (blockIdx.x=cols, blockIdx.y=rows) instead of CTA swizzle
    tma_params: list[str] | None = None  # TMA descriptor param names (e.g. ["A_tma", "B_tma"])
    batched: bool = False  # Batched GEMM: TMA descriptors are per-batch arrays
    extra_smem_bytes: int = 0  # Extra dynamic smem beyond standard double-buffer (e.g. for hybrid TF32 split scratch)
    min_blocks_per_sm: int = 0  # If >0, emit __launch_bounds__(threads, min_blocks_per_sm) to force occupancy


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_INDENT = "  "


def pretty_print(kernel: KernelDef) -> str:
    """Human-readable text dump of a KernelDef AST."""
    lines: list[str] = []
    params = ", ".join(f"{p.dtype} {p.name}" for p in kernel.params)
    bx, by, bz = kernel.block_size
    lines.append(f"kernel {kernel.name}({params}) {{")
    lines.append(f"  block_size: ({bx}, {by}, {bz})")
    if kernel.tile_m is not None:
        lines.append(f"  tile: ({kernel.tile_m}, {kernel.tile_n})")
    lines.append("")
    for stmt in kernel.body:
        lines.extend(_pp_stmt(stmt, depth=1))
    lines.append("}")
    return "\n".join(lines)


def _pp_expr(expr: KernelExpr) -> str:
    """Pretty-print a KernelExpr as a compact string."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        if isinstance(expr.value, float):
            return f"{expr.value:g}"
        return str(expr.value)
    if isinstance(expr, BinOp):
        return f"{_pp_expr(expr.left)} {expr.op} {_pp_expr(expr.right)}"
    if isinstance(expr, Builtin):
        return expr.name
    if isinstance(expr, FuncCall):
        args = ", ".join(_pp_expr(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, Ternary):
        return f"{_pp_expr(expr.cond)} ? {_pp_expr(expr.if_true)} : {_pp_expr(expr.if_false)}"
    if isinstance(expr, ArrayAccess):
        return f"{expr.array}[{_pp_expr(expr.index)}]"
    if isinstance(expr, Cast):
        return f"({expr.dtype})({_pp_expr(expr.expr)})"
    if isinstance(expr, FieldAccess):
        return f"{_pp_expr(expr.expr)}.{expr.field}"
    if isinstance(expr, VectorLoad):
        return f"*(float{expr.width}*)(&{expr.array}[{_pp_expr(expr.index)}])"
    return "??"


def _pp_stmt(stmt: Stmt, depth: int) -> list[str]:
    """Pretty-print a single Stmt at the given indentation depth."""
    pad = _INDENT * depth

    if isinstance(stmt, VarDecl):
        init = f" = {_pp_expr(stmt.init)}" if stmt.init is not None else ""
        return [f"{pad}{stmt.dtype} {stmt.name}{init}"]

    if isinstance(stmt, VarAssign):
        return [f"{pad}{stmt.name} = {_pp_expr(stmt.value)}"]

    if isinstance(stmt, AugAssign):
        return [f"{pad}{stmt.target} {stmt.op} {_pp_expr(stmt.value)}"]

    if isinstance(stmt, Assign):
        return [f"{pad}{_pp_expr(stmt.target)} = {_pp_expr(stmt.value)}"]

    if isinstance(stmt, ArrayDecl):
        dims = "".join(f"[{d}]" for d in stmt.dimensions)
        init = f" = {_pp_expr(stmt.init)}" if stmt.init is not None else ""
        return [f"{pad}{stmt.dtype} {stmt.name}{dims}{init}"]

    if isinstance(stmt, ForLoop):
        step = f", step={_pp_expr(stmt.step)}" if stmt.step else ""
        lines = [f"{pad}for {stmt.var} in [{_pp_expr(stmt.start)}, {_pp_expr(stmt.end)}){step} {{"]
        for child in stmt.body:
            lines.extend(_pp_stmt(child, depth + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, IfStmt):
        lines = [f"{pad}if ({_pp_expr(stmt.cond)}) {{"]
        for child in stmt.body:
            lines.extend(_pp_stmt(child, depth + 1))
        if stmt.else_body:
            lines.append(f"{pad}}} else {{")
            for child in stmt.else_body:
                lines.extend(_pp_stmt(child, depth + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(stmt, SyncThreads):
        return [f"{pad}__syncthreads()"]

    if isinstance(stmt, PragmaUnroll):
        factor = f" {stmt.factor}" if stmt.factor is not None else ""
        return [f"{pad}#pragma unroll{factor}"]

    if isinstance(stmt, RawCode):
        code_lines = stmt.code.split("\n")
        return [f"{pad}{line}" for line in code_lines]

    return [f"{pad}{type(stmt).__name__}(...)"]
