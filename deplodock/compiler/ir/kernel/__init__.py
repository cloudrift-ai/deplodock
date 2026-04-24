"""Kernel IR — C-like AST types.

- :mod:`.ir` — ``GpuKernel`` and its statement/expression node types
  plus the ``KernelOp`` graph-op wrapper.
- :mod:`.normalize` — structural normalization applied to every
  ``GpuKernel`` via ``KernelOp.__post_init__``.

Codegen that produces / consumes these types lives in
``pipeline/passes/lowering/{kernel,cuda}/_emit.py`` — this module holds
only the op definitions.
"""

from deplodock.compiler.ir.kernel.ir import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FieldAccess,
    ForLoop,
    FuncCallExpr,
    GpuExpr,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    KernelOp,
    Literal,
    PragmaUnroll,
    RawCode,
    Stmt,
    SyncThreads,
    TernaryExpr,
    Var,
    VarAssign,
    VarDecl,
    VectorLoad,
    pretty_print,
)

__all__ = [
    # Expression types
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "Expr",
    "ArrayAccess",
    "CastExpr",
    "FieldAccess",
    "VectorLoad",
    "GpuExpr",
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
    "GpuKernelParam",
    "GpuKernel",
    "KernelOp",
    # Pretty printer
    "pretty_print",
]
