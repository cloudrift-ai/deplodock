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
    BinOp,
    Builtin,
    Cast,
    Expr,
    FieldAccess,
    ForLoop,
    FuncCall,
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
    Ternary,
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
    "BinOp",
    "Builtin",
    "FuncCall",
    "Ternary",
    "Expr",
    "ArrayAccess",
    "Cast",
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
