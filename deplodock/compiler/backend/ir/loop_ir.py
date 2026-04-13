"""LoopIR: loop-nest intermediate representation for GPU kernel generation.

Backend-agnostic: describes a kernel as explicit parallel axes, sequential
loops, memory operations, and reductions.  Sits between TileAnalysis
(pattern classification) and KernelDef (imperative C AST), separating
the "what loops do we need" decision from "how to emit C code".

Backend-specific constructs (e.g. CUDA TMA inline asm) use the RawLoopOp
escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.backend.ir.expr import (
    BinOp,
    Builtin,
    Expr,
    FuncCall,
    Literal,
    Ternary,
    Var,
    _coerce,
    _ExprOps,
)

# Re-export shared types so existing ``from loop_ir import Var`` still works.
__all__ = [
    # Shared expression types (re-exported)
    "Var",
    "Literal",
    "BinOp",
    "Builtin",
    "FuncCall",
    "Ternary",
    "Expr",
    "_ExprOps",
    "_coerce",
    # Loop-specific expression types
    "OpCall",
    "RegAccess",
    "LoopExpr",
    # Operations
    "Let",
    "SetVar",
    "ParallelAxis",
    "LoopNest",
    "Alloc",
    "Load",
    "Store",
    "AccumInit",
    "Accum",
    "ShuffleReduce",
    "Barrier",
    "Guard",
    "RawLoopOp",
    "LoopOp",
    # Program
    "LoopProgram",
    # Utilities
    "pretty_print",
    "to_dict",
]

# ---------------------------------------------------------------------------
# Loop-specific expression types
# ---------------------------------------------------------------------------


@dataclass
class OpCall(_ExprOps):
    """Apply a registered elementwise op by name.

    Unlike ``FuncCall`` (which maps to a literal C function call),
    ``OpCall`` is rendered by the codegen via the ``_C_EXPR`` template
    registry.  Example: ``OpCall("relu", [x])`` → ``fmaxf(0.0f, x)``.
    """

    op: str  # op registry name: "add", "relu", "exp", etc.
    args: list[LoopExpr]


@dataclass
class RegAccess(_ExprOps):
    """Compile-time indexed access into a register array.

    The array must have been declared via ``Alloc(name, ..., space="reg",
    shape=(M, N))``.  Indices are compile-time integer constants —
    the codegen expands ``RegAccess("c", [3, 2])`` to the scalar name ``c32``.
    """

    name: str
    indices: list[int]


LoopExpr = Expr | RegAccess | OpCall


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@dataclass
class Let:
    """Bind a named variable to an expression: ``dtype name = expr;``."""

    name: str
    expr: LoopExpr
    dtype: str = "float"


@dataclass
class SetVar:
    """Reassign an existing variable: ``name = expr;``."""

    name: str
    expr: LoopExpr


@dataclass
class ParallelAxis:
    """Maps to a grid dimension.  One block per value in [0, bound).

    The codegen decides how to implement this (e.g. VarDecl + bounds guard).
    """

    name: str  # "row", "blk", "cta"
    dim: str  # variable name bound to this axis ("blockIdx.x")
    bound: str  # parameter name for the upper bound ("rows", "M", "n")


@dataclass
class LoopNest:
    """Sequential loop over a dimension (K-loop, column scan, epilogue pass)."""

    var: str
    start: LoopExpr
    end: LoopExpr
    step: LoopExpr | None  # None = increment by 1
    body: list[LoopOp]


@dataclass
class Alloc:
    """Register or shared-memory allocation.

    For accumulators: shape=None (scalar), space="reg", init=Literal(0.0).
    For smem buffers: shape=(64, 64), space="smem".
    """

    name: str
    dtype: str  # "float"
    shape: tuple[int, ...] | None  # None = scalar
    space: str  # "reg" | "smem"
    init: LoopExpr | None = None


def _to_name(v: str | Var) -> str:
    """Extract buffer name from a string or Var."""
    return v.name if isinstance(v, Var) else v


@dataclass
class Load:
    """Load from global or shared memory into a register."""

    dst: str
    src: str | Var  # buffer name (Var auto-extracted)
    indices: LoopExpr
    space: str  # "global" | "smem"
    guard: LoopExpr | None = None  # bounds check; zero if false

    def __post_init__(self) -> None:
        self.src = _to_name(self.src)


@dataclass
class Store:
    """Write to global or shared memory."""

    dst: str | Var  # buffer name (Var auto-extracted)
    indices: LoopExpr
    value: LoopExpr
    space: str  # "global" | "smem"
    guard: LoopExpr | None = None
    atomic: bool = False  # atomicAdd for split-K

    def __post_init__(self) -> None:
        self.dst = _to_name(self.dst)


@dataclass
class AccumInit:
    """Declare a reduction accumulator with the identity value for ``op``.

    The codegen looks up the init value from ``REDUCE_REGISTRY``.
    """

    name: str
    op: str  # "sum" | "max" | "prod"


@dataclass
class Accum:
    """Fold value into accumulator using reduction op."""

    dst: str
    op: str  # "sum" | "max"
    value: LoopExpr


@dataclass
class ShuffleReduce:
    """Warp shuffle reduction.

    ``kind`` selects the variant:
    - ``"block"``: full block-wide reduce using __shfl_down_sync + cross-warp
      shared memory + broadcast.  After this op, all threads hold the result.
    - ``"warp_xor"``: intra-warp reduce using __shfl_xor_sync.  Used for
      in-register softmax where each thread holds different columns.
    """

    var: str
    op: str  # "sum" | "max"
    kind: str = "block"  # "block" | "warp_xor"


@dataclass
class Barrier:
    """Thread synchronization barrier (__syncthreads)."""


@dataclass
class Guard:
    """Conditional execution (bounds check)."""

    cond: LoopExpr
    body: list[LoopOp]


@dataclass
class SmemPipelineKLoop:
    """Double-buffered K-loop through shared memory.

    Backend-agnostic pipeline schedule.  Describes the structure (stages,
    tile geometry, thread tile, k_splits) without specifying HOW tiles are
    loaded or HOW synchronization happens — the backend codegen decides
    (TMA + mbarrier for CUDA sm_90+, explicit loads + __syncthreads otherwise).
    """

    stages: int  # 2 = double-buffer, 3 = triple-buffer (future)
    tile_m: int
    tile_n: int
    block_k: int
    a_size: int  # tile_m * block_k (A tile elements per stage)
    stage_size: int  # a_size + block_k * tile_n (total per stage)
    thread_m: int
    thread_n: int
    tx: int  # blockDim.x
    k_splits: int
    is_batched: bool


@dataclass
class RawLoopOp:
    """Escape hatch for backend-specific inline code (e.g. TMA asm).

    Should be replaced with proper LoopIR ops over time.
    """

    code: str
    comment: str = ""


LoopOp = (
    Let
    | SetVar
    | ParallelAxis
    | LoopNest
    | Alloc
    | Load
    | Store
    | AccumInit
    | Accum
    | ShuffleReduce
    | SmemPipelineKLoop
    | Barrier
    | Guard
    | RawLoopOp
)


# ---------------------------------------------------------------------------
# Top-level program
# ---------------------------------------------------------------------------


@dataclass
class LoopProgram:
    """Complete loop program for one kernel."""

    name: str
    params: list[tuple[str, str]]  # [(dtype, name), ...] e.g. [("const float*", "A"), ("int", "M")]
    body: list[LoopOp]
    block_size: tuple[int, int, int]
    smem_bytes: int = 0
    # Metadata carried forward into KernelDef.
    tile_m: int | None = None
    tile_n: int | None = None
    grid_2d: bool = False
    tma_params: list[str] | None = None
    tma_config: object | None = None  # TMALoadConfig from cuda backend (if TMA strategy)
    batched: bool = False
    includes: list[str] | None = None
    extra_smem_bytes: int = 0
    min_blocks_per_sm: int = 0


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_INDENT = "  "


def pretty_print(program: LoopProgram) -> str:
    """Human-readable text dump of a LoopProgram."""
    lines: list[str] = []
    bx, by, bz = program.block_size
    params_str = ", ".join(f"{dt} {nm}" for dt, nm in program.params)
    lines.append(f"loop_program {program.name}({params_str}) {{")
    lines.append(f"  block_size: ({bx}, {by}, {bz})")
    if program.smem_bytes:
        lines.append(f"  smem_bytes: {program.smem_bytes}")
    if program.tile_m is not None:
        lines.append(f"  tile: ({program.tile_m}, {program.tile_n})")
    lines.append("")
    for op in program.body:
        lines.extend(_pp_op(op, depth=1))
    lines.append("}")
    return "\n".join(lines)


def _pp_op(op: LoopOp, depth: int) -> list[str]:
    """Pretty-print a single LoopOp at the given indentation depth."""
    pad = _INDENT * depth

    if isinstance(op, Let):
        return [f"{pad}let {op.dtype} {op.name} = {_pp_expr(op.expr)}"]

    if isinstance(op, SetVar):
        return [f"{pad}{op.name} = {_pp_expr(op.expr)}"]

    if isinstance(op, ParallelAxis):
        return [f"{pad}parallel {op.name} = {op.dim}  // bound: {op.bound}"]

    if isinstance(op, LoopNest):
        start = _pp_expr(op.start)
        end = _pp_expr(op.end)
        step = _pp_expr(op.step) if op.step else "1"
        lines = [f"{pad}for {op.var} in [{start}, {end}) step {step} {{"]
        for child in op.body:
            lines.extend(_pp_op(child, depth + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(op, Alloc):
        shape_str = f"[{', '.join(str(d) for d in op.shape)}]" if op.shape else "scalar"
        init_str = f" = {_pp_expr(op.init)}" if op.init else ""
        return [f"{pad}alloc {op.space} {op.dtype} {op.name}{shape_str}{init_str}"]

    if isinstance(op, Load):
        guard_str = f"  guard({_pp_expr(op.guard)})" if op.guard else ""
        return [f"{pad}load {op.dst} = {op.src}[{_pp_expr(op.indices)}] ({op.space}){guard_str}"]

    if isinstance(op, Store):
        guard_str = f"  guard({_pp_expr(op.guard)})" if op.guard else ""
        atomic_str = " atomic" if op.atomic else ""
        return [f"{pad}store{atomic_str} {op.dst}[{_pp_expr(op.indices)}] = {_pp_expr(op.value)} ({op.space}){guard_str}"]

    if isinstance(op, AccumInit):
        return [f"{pad}accum_init {op.name} {op.op}"]

    if isinstance(op, Accum):
        return [f"{pad}accum {op.dst} {op.op} {_pp_expr(op.value)}"]

    if isinstance(op, ShuffleReduce):
        return [f"{pad}shuffle_reduce {op.var} {op.op} ({op.kind})"]

    if isinstance(op, Barrier):
        return [f"{pad}barrier"]

    if isinstance(op, SmemPipelineKLoop):
        return [f"{pad}smem_pipeline(stages={op.stages}, tile={op.tile_m}x{op.tile_n}, bk={op.block_k}, ks={op.k_splits})"]

    if isinstance(op, Guard):
        lines = [f"{pad}guard ({_pp_expr(op.cond)}) {{"]
        for child in op.body:
            lines.extend(_pp_op(child, depth + 1))
        lines.append(f"{pad}}}")
        return lines

    if isinstance(op, RawLoopOp):
        comment = f"  // {op.comment}" if op.comment else ""
        # Show first line of code + ellipsis if multi-line.
        first_line = op.code.split("\n")[0]
        n_lines = op.code.count("\n") + 1
        if n_lines > 1:
            return [f"{pad}raw ({n_lines} lines){comment}: {first_line} ..."]
        return [f"{pad}raw{comment}: {first_line}"]

    # Extension ops (e.g. TMAKLoop from cuda backend)
    return [f"{pad}{type(op).__name__}(...)"]


def _pp_expr(expr: LoopExpr) -> str:
    """Pretty-print a LoopExpr as a compact string."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        if isinstance(expr.value, float):
            return f"{expr.value:g}"
        return str(expr.value)
    if isinstance(expr, BinOp):
        return f"({_pp_expr(expr.left)} {expr.op} {_pp_expr(expr.right)})"
    if isinstance(expr, Builtin):
        return expr.name
    if isinstance(expr, FuncCall):
        args = ", ".join(_pp_expr(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, OpCall):
        args = ", ".join(_pp_expr(a) for a in expr.args)
        return f"{expr.op}({args})"
    if isinstance(expr, Ternary):
        return f"({_pp_expr(expr.cond)} ? {_pp_expr(expr.if_true)} : {_pp_expr(expr.if_false)})"
    if isinstance(expr, RegAccess):
        return f"{expr.name}[{','.join(str(i) for i in expr.indices)}]"
    return "??"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize(obj: object) -> object:
    """Recursively serialize LoopIR dataclasses to JSON-compatible dicts.

    Each dataclass gets a ``"type"`` key with the class name.
    All other values pass through as-is.
    """
    import dataclasses

    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # snake_case class name: LoopNest → loop_nest, ParallelAxis → parallel_axis
        cls_name = type(obj).__name__
        tag = "".join(f"_{c.lower()}" if c.isupper() else c for c in cls_name).lstrip("_")
        d: dict = {"type": tag}
        for f in dataclasses.fields(obj):
            d[f.name] = _serialize(getattr(obj, f.name))
        return d
    return obj


def to_dict(program: LoopProgram) -> dict:
    """JSON-serializable dict for dump-dir artifacts."""
    d = _serialize(program)
    # params are tuples serialized as lists; convert to list-of-dicts.
    d["params"] = [{"dtype": p[0], "name": p[1]} for p in program.params]
    return d
