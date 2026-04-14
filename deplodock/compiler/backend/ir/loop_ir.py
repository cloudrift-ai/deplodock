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
    """Load from global or shared memory into a register.

    ``indices`` is a list of per-dimension index expressions.  The codegen
    flattens them using buffer strides (row-major by default).  A single-
    element list ``[expr]`` represents a pre-flattened linear index for
    shared-memory or legacy call sites.
    """

    dst: str
    src: str | Var  # buffer name (Var auto-extracted)
    indices: list[LoopExpr]
    space: str  # "global" | "smem"
    guard: LoopExpr | None = None  # bounds check; zero if false

    def __post_init__(self) -> None:
        self.src = _to_name(self.src)


@dataclass
class Store:
    """Write to global or shared memory.

    ``indices`` is a list of per-dimension index expressions (see Load).
    """

    dst: str | Var  # buffer name (Var auto-extracted)
    indices: list[LoopExpr]
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

    Backend-agnostic pipeline schedule.  For TMA the codegen expands this
    to inline PTX.  For explicit smem, call ``expand()`` to get primitive
    LoopIR ops (Alloc, Load, Store, Barrier, LoopNest).
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
    # Buffer names for explicit smem expansion (not used by TMA codegen)
    a_buf: str = ""
    b_buf: str = ""

    def expand(self) -> list:
        """Expand into primitive LoopIR ops for explicit smem K-loop.

        Produces: smem Alloc + k-range + outer tile loop (load A→smem,
        barrier, inner FMA loop, barrier).  Backend-agnostic.
        """
        bk = self.block_k
        smem_stride = bk + 1
        I = "int"  # noqa: E741
        M, N, K = Var("M"), Var("N"), Var("K")  # noqa: N806
        A, B = Var(self.a_buf), Var(self.b_buf)  # noqa: N806

        ops: list = []
        ops.append(Alloc("As", "float", (self.tile_m * smem_stride,), "smem"))

        if self.k_splits > 1:
            bidz = Builtin("blockIdx.z")
            k_per, k_start = Var("k_per"), Var("k_start")
            ops.append(Let("k_per", K / bk / Var("k_splits") * bk, dtype=I))
            ops.append(Let("k_start", bidz * k_per, dtype=I))
            ops.append(Let("k_end", Ternary(bidz.eq(Literal(self.k_splits - 1, I)), K, k_start + k_per), dtype=I))
        else:
            ops.append(Let("k_start", Literal(0, I), dtype=I))
            ops.append(Let("k_end", K, dtype=I))

        a_src = Var("Ab") if self.is_batched else A
        b_src = Var("Bb") if self.is_batched else B

        row_base, col_base = Var("row_base"), Var("col_base")
        sr, tk, kk = Var("sr"), Var("tk"), Var("kk")
        tidx = Builtin("threadIdx.x")

        tk_body: list = []
        for r in range(self.thread_m):
            row_r = row_base + r
            k_col = tk + tidx
            tk_body.append(Load(f"As_ld_{r}", a_src, [row_r, k_col], "global", guard=row_r.lt(M).and_(k_col.lt(K))))
            tk_body.append(Store("As", [(sr + r) * smem_stride + tidx], Var(f"As_ld_{r}"), "smem"))
        tk_body.append(Barrier())

        kk_body: list = []
        for c in range(self.thread_n):
            col_c = col_base + c
            kk_body.append(Load(f"b{c}", b_src, [tk + kk, col_c], "global", guard=col_c.lt(N)))
        for r in range(self.thread_m):
            kk_body.append(Load(f"a{r}", "As", [(sr + r) * smem_stride + kk], "smem"))
            for c in range(self.thread_n):
                kk_body.append(Accum(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))
        tk_body.append(LoopNest("kk", Literal(0, I), Literal(bk, I), None, kk_body))
        tk_body.append(Barrier())

        ops.append(LoopNest("tk", Var("k_start"), Var("k_end"), Literal(bk, I), tk_body))
        return ops


@dataclass
class RawLoopOp:
    """Escape hatch for backend-specific inline code (e.g. TMA asm)."""

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
    """Complete loop program for one kernel.

    Contains only the loop structure (params, body, strides).  Backend
    metadata (block_size, tile dims, TMA config, etc.) lives on the
    Schedule and flows into KernelDef via ``loop_ir_to_kernel()``.
    """

    name: str
    params: list[tuple[str, str]]  # [(dtype, name), ...] e.g. [("const float*", "A"), ("int", "M")]
    body: list[LoopOp]
    smem_bytes: int = 0
    # Per-buffer stride variable names for multi-dim index flattening.
    # Maps buffer name → list of stride var names (outer-to-inner, excluding
    # the implicit stride-1 innermost dim).  E.g. {"A": ["K"], "B": ["N"]}
    # for a contraction where A is (M, K) and B is (K, N).
    dim_strides: dict[str, list[str]] | None = None


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_INDENT = "  "


def pretty_print(program: LoopProgram, schedule: object | None = None) -> str:
    """Human-readable text dump of a LoopProgram.

    When ``schedule`` is provided (a backend Schedule dataclass), block_size
    and tile dimensions are included in the header.
    """
    lines: list[str] = []
    params_str = ", ".join(f"{dt} {nm}" for dt, nm in program.params)
    lines.append(f"loop_program {program.name}({params_str}) {{")
    if schedule is not None:
        bx, by, bz = schedule.grid.block_size
        lines.append(f"  block_size: ({bx}, {by}, {bz})")
    if program.smem_bytes:
        lines.append(f"  smem_bytes: {program.smem_bytes}")
    if schedule is not None and getattr(schedule, "tile_m", None) is not None:
        lines.append(f"  tile: ({schedule.tile_m}, {schedule.tile_n})")
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
        idx_str = ", ".join(_pp_expr(i) for i in op.indices)
        return [f"{pad}load {op.dst} = {op.src}[{idx_str}] ({op.space}){guard_str}"]

    if isinstance(op, Store):
        guard_str = f"  guard({_pp_expr(op.guard)})" if op.guard else ""
        atomic_str = " atomic" if op.atomic else ""
        idx_str = ", ".join(_pp_expr(i) for i in op.indices)
        return [f"{pad}store{atomic_str} {op.dst}[{idx_str}] = {_pp_expr(op.value)} ({op.space}){guard_str}"]

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


def to_dict(program: LoopProgram, schedule: object | None = None) -> dict:
    """JSON-serializable dict for dump-dir artifacts."""
    d = _serialize(program)
    # params are tuples serialized as lists; convert to list-of-dicts.
    d["params"] = [{"dtype": p[0], "name": p[1]} for p in program.params]
    # Merge schedule metadata when available.
    if schedule is not None:
        d["block_size"] = list(schedule.grid.block_size)
        if getattr(schedule, "tile_m", None) is not None:
            d["tile_m"] = schedule.tile_m
            d["tile_n"] = schedule.tile_n
    return d
