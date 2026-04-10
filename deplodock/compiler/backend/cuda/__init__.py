"""CUDA backend: codegen, lowering, and kernel execution."""

from deplodock.compiler.backend.cuda.codegen import emit_kernel
from deplodock.compiler.backend.cuda.ir import (
    ArrayAccess,
    Assign,
    AugAssign,
    BinOp,
    CudaBuiltin,
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
from deplodock.compiler.backend.cuda.lower import lower_graph
from deplodock.compiler.backend.cuda.runner import KernelResult, has_cuda_gpu, has_nvcc, run_kernel

__all__ = [
    "ArrayAccess",
    "Assign",
    "AugAssign",
    "BinOp",
    "CudaBuiltin",
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
    "lower_graph",
    "run_kernel",
]
