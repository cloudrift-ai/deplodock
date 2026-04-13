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
