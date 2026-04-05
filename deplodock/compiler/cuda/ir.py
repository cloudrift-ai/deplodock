"""CUDA IR — imperative AST for kernel code generation."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass
class Var:
    """Variable reference."""

    name: str


@dataclass
class Literal:
    """Numeric constant."""

    value: int | float
    dtype: str = "float"


@dataclass
class BinOp:
    """Binary operation."""

    op: str  # "+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "&&", "||"
    left: Expr
    right: Expr


@dataclass
class ArrayAccess:
    """Array element access: array[index]."""

    array: str
    index: Expr


@dataclass
class CudaBuiltin:
    """CUDA built-in variable (threadIdx.x, blockIdx.y, blockDim.x, etc.)."""

    name: str


Expr = Var | Literal | BinOp | ArrayAccess | CudaBuiltin


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass
class VarDecl:
    """Variable declaration with optional initializer."""

    dtype: str
    name: str
    init: Expr | None = None


@dataclass
class Assign:
    """Assignment: target = value."""

    target: ArrayAccess
    value: Expr


@dataclass
class AugAssign:
    """Augmented assignment: target op= value."""

    target: str
    op: str  # "+=", "-=", "*="
    value: Expr


@dataclass
class ForLoop:
    """C-style for loop: for (int var = start; var < end; var++)."""

    var: str
    start: Expr
    end: Expr
    body: list[Stmt]


@dataclass
class IfStmt:
    """If statement (no else)."""

    cond: Expr
    body: list[Stmt]


@dataclass
class SyncThreads:
    """__syncthreads() barrier."""


Stmt = VarDecl | Assign | AugAssign | ForLoop | IfStmt | SyncThreads


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
