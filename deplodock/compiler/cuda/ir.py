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


@dataclass
class FuncCall:
    """Function call expression: name(args)."""

    name: str  # "min", "max", "__ldg", etc.
    args: list[Expr]


@dataclass
class Cast:
    """Type cast: (dtype)(expr)."""

    dtype: str  # "float4", "int", etc.
    expr: Expr


@dataclass
class FieldAccess:
    """Struct field access: expr.field (for float4.x, .y, .z, .w)."""

    expr: Expr
    field: str  # "x", "y", "z", "w"


@dataclass
class Ternary:
    """Ternary expression: cond ? if_true : if_false."""

    cond: Expr
    if_true: Expr
    if_false: Expr


@dataclass
class VectorLoad:
    """Load N contiguous floats as a vector: *(floatN*)(&array[index]).

    Used for coalesced float2/float4 loads from global memory.
    """

    array: str
    index: Expr
    width: int = 4  # 2 or 4


Expr = Var | Literal | BinOp | ArrayAccess | CudaBuiltin | FuncCall | Cast | FieldAccess | Ternary | VectorLoad


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
    """C-style for loop: for (int var = start; var < end; var += step)."""

    var: str
    start: Expr
    end: Expr
    body: list[Stmt]
    step: Expr | None = None  # None = var++ (increment by 1)


@dataclass
class IfStmt:
    """If statement with optional else."""

    cond: Expr
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
    init: Expr | None = None


@dataclass
class PragmaUnroll:
    """#pragma unroll [factor] before a for loop."""

    factor: int | None = None  # None = fully unroll


@dataclass
class RawCode:
    """Raw C/CUDA code injected verbatim."""

    code: str


Stmt = VarDecl | Assign | AugAssign | ForLoop | IfStmt | SyncThreads | ArrayDecl | PragmaUnroll | RawCode


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
    tma_params: list[str] | None = None  # TMA descriptor param names (e.g. ["A_tma", "B_tma"])
    batched: bool = False  # Batched GEMM: TMA descriptors are per-batch arrays
    extra_smem_bytes: int = 0  # Extra dynamic smem beyond standard double-buffer (e.g. for hybrid TF32 split scratch)
    min_blocks_per_sm: int = 0  # If >0, emit __launch_bounds__(threads, min_blocks_per_sm) to force occupancy


# ---------------------------------------------------------------------------
# Execution plan — multi-kernel block execution
# ---------------------------------------------------------------------------


@dataclass
class BufferAlloc:
    """GPU buffer allocation for an intermediate tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"


@dataclass
class KernelLaunch:
    """One kernel invocation in the execution plan."""

    kernel_name: str  # references a KernelDef.name
    input_buffers: list[str]
    output_buffers: list[str]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem_bytes: int = 0


@dataclass
class ExecutionPlan:
    """Full block execution: kernels + buffers + ordered launches."""

    kernels: list[KernelDef]
    buffers: list[BufferAlloc]
    launches: list[KernelLaunch]
    input_names: list[str]
    output_names: list[str]
    constant_names: list[str]
