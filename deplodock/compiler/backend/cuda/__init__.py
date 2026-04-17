"""CUDA backend: codegen, lowering, and kernel execution."""

from deplodock.compiler.backend.cuda.emit import compile_kernels, emit_kernel
from deplodock.compiler.backend.cuda.runner import KernelResult, has_cuda_gpu, has_nvcc, run_kernel
from deplodock.compiler.backend.kernel_codegen import emit_kernel as emit_kernel_source
from deplodock.compiler.ir.gpu import (
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
    "compile_kernels",
    "emit_kernel",
    "emit_kernel_source",
    "has_cuda_gpu",
    "has_nvcc",
    "run_kernel",
]
