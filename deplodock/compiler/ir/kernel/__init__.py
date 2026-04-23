"""Kernel IR — C-like AST + per-kernel codegen.

Submodules:
- :mod:`.ir` — ``GpuKernel`` and its statement/expression node types
  (``VarDecl``, ``Assign``, ``ForLoop``, ``IfStmt``, ``ArrayAccess``, …),
  plus the ``KernelOp`` graph-op wrapper that carries a ``GpuKernel`` as
  a graph node payload with launch metadata.
- :mod:`.emit` — per-node codegen: ``LoopOp`` node → ``GpuKernel``
  (``emit_kernel`` / ``launch_config`` / ``kernel_name_for``) plus the
  ``emit_kernel_source`` helper that renders a ``GpuKernel`` to a C
  source string via ``ir/cuda/emit``.

The ``passes/lowering/kernel`` pass uses :mod:`.emit` to turn each
``LoopOp`` graph node into a ``KernelOp``; ``passes/lowering/cuda``
consumes ``KernelOp.kernel`` via ``emit_kernel_source``.
"""

from deplodock.compiler.ir.kernel.emit import (
    emit_kernel,
    emit_kernel_source,
    kernel_name_for,
    launch_config,
)
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
    # Codegen
    "emit_kernel",
    "emit_kernel_source",
    "kernel_name_for",
    "launch_config",
]
