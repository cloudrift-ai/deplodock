"""CUDA backend: codegen, lowering, and kernel execution."""

from deplodock.compiler.backend.cuda.generators import analyze, generate_kernel, lower_tiled
from deplodock.compiler.backend.cuda.runner import KernelResult, has_cuda_gpu, has_nvcc, run_kernel
from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel
from deplodock.compiler.backend.ir.kernel_ir import (
    ArrayAccess,
    Assign,
    AugAssign,
    BinOp,
    Builtin,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    Stmt,
    SyncThreads,
    Var,
    VarDecl,
)

__all__ = [
    "ArrayAccess",
    "Assign",
    "AugAssign",
    "BinOp",
    "Builtin",
    "ForLoop",
    "IfStmt",
    "KernelDef",
    "KernelParam",
    "Literal",
    "Stmt",
    "SyncThreads",
    "Var",
    "VarDecl",
    "emit_kernel",
    "KernelResult",
    "has_cuda_gpu",
    "has_nvcc",
    "analyze",
    "generate_kernel",
    "lower_tiled",
    "run_kernel",
]
