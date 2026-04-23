"""CUDA backend: codegen, lowering, and kernel execution."""

from deplodock.compiler.backend.cuda.runner import KernelResult, has_cuda_gpu, has_nvcc, run_kernel
from deplodock.compiler.ir.kernel import (
    ArrayAccess,
    Assign,
    AugAssign,
    BinOp,
    Builtin,
    ForLoop,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    Literal,
    Stmt,
    SyncThreads,
    Var,
    VarDecl,
    emit_kernel,
    emit_kernel_source,
)

__all__ = [
    "ArrayAccess",
    "Assign",
    "AugAssign",
    "BinOp",
    "Builtin",
    "ForLoop",
    "GpuKernel",
    "GpuKernelParam",
    "IfStmt",
    "KernelResult",
    "Literal",
    "Stmt",
    "SyncThreads",
    "Var",
    "VarDecl",
    "emit_kernel",
    "emit_kernel_source",
    "has_cuda_gpu",
    "has_nvcc",
    "run_kernel",
]
