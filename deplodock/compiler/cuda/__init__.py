"""CUDA backend: codegen, lowering, and kernel execution."""

from deplodock.compiler.cuda.codegen import emit_kernel
from deplodock.compiler.cuda.ir import (
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
from deplodock.compiler.cuda.lower import lower_graph
from deplodock.compiler.cuda.runner import KernelResult, has_cuda_gpu, has_nvcc, run_kernel

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
