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

# ---------------------------------------------------------------------------
# Expression operator overloading mixin
# ---------------------------------------------------------------------------


class _ExprOps:
    """Mixin that adds arithmetic and comparison operators to LoopExpr nodes.

    Returns LoopBinOp nodes, enabling::

        Var("row") * Var("cols") + Var("j")   # → LoopBinOp("+", LoopBinOp("*", ...), ...)
        Var("i") < Var("n")                    # → LoopBinOp("<", ...)
    """

    def __add__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("+", self, _coerce(other))

    def __radd__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("+", _coerce(other), self)

    def __sub__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("-", self, _coerce(other))

    def __rsub__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("-", _coerce(other), self)

    def __mul__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("*", self, _coerce(other))

    def __rmul__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("*", _coerce(other), self)

    def __truediv__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("/", self, _coerce(other))

    def __mod__(self, other: LoopExpr) -> LoopBinOp:
        return LoopBinOp("%", self, _coerce(other))

    def __neg__(self) -> LoopBinOp:
        return LoopBinOp("-", LoopLiteral(0, "int"), self)

    def lt(self, other: LoopExpr) -> LoopBinOp:
        """Less-than (``<``). Named method to avoid __lt__ conflict with dataclass ordering."""
        return LoopBinOp("<", self, _coerce(other))

    def ge(self, other: LoopExpr) -> LoopBinOp:
        """Greater-or-equal (``>=``)."""
        return LoopBinOp(">=", self, _coerce(other))

    def eq(self, other: LoopExpr) -> LoopBinOp:
        """Equal (``==``). Named method to avoid __eq__ conflict with dataclass equality."""
        return LoopBinOp("==", self, _coerce(other))

    def and_(self, other: LoopExpr) -> LoopBinOp:
        """Logical AND (``&&``)."""
        return LoopBinOp("&&", self, _coerce(other))

    def or_(self, other: LoopExpr) -> LoopBinOp:
        """Logical OR (``||``)."""
        return LoopBinOp("||", self, _coerce(other))


def _coerce(v: LoopExpr | int | float) -> LoopExpr:
    """Coerce Python int/float to LoopLiteral for operator overloading."""
    if isinstance(v, int):
        return LoopLiteral(v, "int")
    if isinstance(v, float):
        return LoopLiteral(v)
    return v


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass
class LoopVar(_ExprOps):
    """Variable reference."""

    name: str


@dataclass
class LoopLiteral(_ExprOps):
    """Numeric constant."""

    value: int | float
    dtype: str = "float"


@dataclass
class LoopBinOp(_ExprOps):
    """Binary operation."""

    op: str  # "+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "&&", "||"
    left: LoopExpr
    right: LoopExpr


@dataclass
class LoopBuiltin(_ExprOps):
    """GPU built-in variable (threadIdx.x, blockIdx.y, blockDim.x, etc.)."""

    name: str


@dataclass
class LoopFuncCall(_ExprOps):
    """Intrinsic / math function call."""

    name: str  # "expf", "rsqrtf", "fmaxf", etc.
    args: list[LoopExpr]


@dataclass
class LoopTernary(_ExprOps):
    """Ternary expression: cond ? if_true : if_false."""

    cond: LoopExpr
    if_true: LoopExpr
    if_false: LoopExpr


@dataclass
class RegAccess(_ExprOps):
    """Compile-time indexed access into a register array.

    The array must have been declared via ``Alloc(name, ..., space="reg",
    shape=(M, N))``.  Indices are compile-time integer constants —
    the codegen expands ``RegAccess("c", [3, 2])`` to the scalar name ``c32``.
    """

    name: str
    indices: list[int]


LoopExpr = LoopVar | LoopLiteral | LoopBinOp | LoopBuiltin | LoopFuncCall | LoopTernary | RegAccess


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


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

    For accumulators: shape=None (scalar), space="reg", init=LoopLiteral(0.0).
    For smem buffers: shape=(64, 64), space="smem".
    """

    name: str
    dtype: str  # "float"
    shape: tuple[int, ...] | None  # None = scalar
    space: str  # "reg" | "smem"
    init: LoopExpr | None = None


@dataclass
class Load:
    """Load from global or shared memory into a register."""

    dst: str
    src: str  # buffer name
    indices: LoopExpr
    space: str  # "global" | "smem"
    guard: LoopExpr | None = None  # bounds check; zero if false


@dataclass
class Store:
    """Write to global or shared memory."""

    dst: str  # buffer name
    indices: LoopExpr
    value: LoopExpr
    space: str  # "global" | "smem"
    guard: LoopExpr | None = None
    atomic: bool = False  # atomicAdd for split-K


@dataclass
class Compute:
    """Elementwise computation producing a named result."""

    dst: str
    op: str  # "mul", "add", "exp", "rsqrt", "relu", "builtin", "ptr_offset", etc.
    args: list[LoopExpr]
    dtype: str = "float"  # "int" for grid index computations


@dataclass
class Accumulate:
    """Reduction body: fold value into accumulator."""

    dst: str
    op: str  # "sum" | "max"
    value: LoopExpr


@dataclass
class WarpReduce:
    """Cross-thread warp shuffle reduction (__shfl_down_sync).

    After this op, `var` holds the block-wide reduced value (thread 0
    for max, all threads for sum via broadcast).
    """

    var: str
    op: str  # "sum" | "max"


@dataclass
class WarpShuffleXor:
    """Horizontal warp shuffle reduction (__shfl_xor_sync).

    Reduces ``var`` across threads using XOR lane masks (offsets 16,8,4,2,1).
    Used for in-register softmax where each thread holds different columns
    of the same output row.
    """

    var: str
    op: str  # "sum" | "max"


@dataclass
class Barrier:
    """Thread synchronization barrier (__syncthreads)."""


@dataclass
class Guard:
    """Conditional execution (bounds check)."""

    cond: LoopExpr
    body: list[LoopOp]


@dataclass
class RawLoopOp:
    """Escape hatch for backend-specific inline code (e.g. TMA asm).

    Should be replaced with proper LoopIR ops over time.
    """

    code: str
    comment: str = ""


LoopOp = ParallelAxis | LoopNest | Alloc | Load | Store | Compute | Accumulate | WarpReduce | WarpShuffleXor | Barrier | Guard | RawLoopOp


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

    if isinstance(op, Compute):
        args_str = ", ".join(_pp_expr(a) for a in op.args)
        return [f"{pad}compute {op.dst} = {op.op}({args_str})"]

    if isinstance(op, Accumulate):
        return [f"{pad}accumulate {op.dst} {op.op} {_pp_expr(op.value)}"]

    if isinstance(op, WarpReduce):
        return [f"{pad}warp_reduce {op.var} {op.op}"]

    if isinstance(op, WarpShuffleXor):
        return [f"{pad}warp_shuffle_xor {op.var} {op.op}"]

    if isinstance(op, Barrier):
        return [f"{pad}barrier"]

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
    if isinstance(expr, LoopVar):
        return expr.name
    if isinstance(expr, LoopLiteral):
        if isinstance(expr.value, float):
            return f"{expr.value:g}"
        return str(expr.value)
    if isinstance(expr, LoopBinOp):
        return f"({_pp_expr(expr.left)} {expr.op} {_pp_expr(expr.right)})"
    if isinstance(expr, LoopBuiltin):
        return expr.name
    if isinstance(expr, LoopFuncCall):
        args = ", ".join(_pp_expr(a) for a in expr.args)
        return f"{expr.name}({args})"
    if isinstance(expr, LoopTernary):
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
