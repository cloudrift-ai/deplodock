"""Tile IR — schedule + leaf compute, the IR that lowers directly to CUDA.

Tile IR sits between Loop IR (math) and CUDA source (text). Its job is to
encode *how* the math runs on the GPU — which axes go to threads / blocks,
where loops are tiled, what lives in shared memory — without touching the
math itself. Every scheduling decision is a Tile-IR-to-Tile-IR rewrite;
codegen consumes whatever final form comes out.

Pipeline shape:

    Loop IR ──lower_naive──▶ Tile IR (single-thread)
                  ──[ExtractGlobalSchedule, TileReduce, SmemStageReduce, …]──▶ Tile IR
                  ──emit_cuda──▶ CUDA source

Lowering is mechanical: each Loop IR ``Loop`` becomes a ``FreeLoop`` (or
``Reduce`` if the body has an ``Accum``); each leaf stmt translates 1:1 to
``Let`` / ``Store`` / ``AccumFold``. Strategies populate ``Kernel.thread_axes``
/ ``block_axes`` and rewrite the body — they don't touch lowering or codegen.

Statement node types:

- **Leaves** — ``Let`` (SSA bind), ``Store`` (memory write), ``AccumFold``
  (reduce step), ``Sync`` (barrier), ``Cond`` (if/else).
- **Loops** — ``FreeLoop`` (per-thread serial walk), ``Reduce`` (register
  accumulator), ``Tile`` (outer slab walk), ``Coop`` (cooperative load).

Expression node types reuse ``ir.expr`` directly (``Var``, ``Literal``,
``BinaryExpr``, ``FuncCallExpr``, ``TernaryExpr``, ``CastExpr``, ``Builtin``)
plus one Tile-IR-specific addition: ``Index(buf, indices)`` for multi-dim
buffer access. The buffer's storage class (global / smem) is determined by
name lookup against ``Kernel.params`` / ``Kernel.smem`` at render time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
    _ExprOps,
)
from deplodock.compiler.ir.expr import (
    Expr as _BaseExpr,
)
from deplodock.compiler.ir.loop import Axis

# ---------------------------------------------------------------------------
# Expressions — Tile-IR addition on top of ir.expr
# ---------------------------------------------------------------------------


@dataclass
class Index(_ExprOps):
    """Memory read: ``buf[i0][i1]...``.

    Works for any storage class — the buffer's storage (global pointer in
    ``Kernel.params``, shared array in ``Kernel.smem``) is determined at
    render time by name lookup. ``indices`` is multi-dim from the start;
    render flattens row-major using the declared shape.
    """

    buf: str
    indices: tuple[Expr, ...]


Expr = _BaseExpr | Index


# ---------------------------------------------------------------------------
# Statements — leaves
# ---------------------------------------------------------------------------


@dataclass
class Let:
    """SSA binding: ``T name = init;``.

    Replaces both Loop IR's ``Load`` (init = ``Index(buf, [...])``) and
    ``Assign`` (init = ``BinaryExpr`` / ``FuncCallExpr`` / ``TernaryExpr``).
    """

    name: str
    init: Expr


@dataclass
class Store:
    """Memory write: ``buf[indices...] = value;``.

    Used both for global writes (Loop IR ``Write`` lowering) and smem writes
    (cooperative-load ``Stage`` strategy output).
    """

    buf: str
    indices: tuple[Expr, ...]
    value: Expr


@dataclass
class AccumFold:
    """Reduce step: ``target op= value;``.

    ``op`` is an ``ElementwiseImpl`` (the same op-vocabulary used by Loop IR's
    ``Accum``), giving us name / commutativity / identity in one object.
    Renders as ``+=`` / ``*=`` for add/multiply (and ``sum``/``prod`` aliases),
    or as a ``target = fmax(target, value)`` / ``fmin(...)`` rebinding for
    maximum / minimum.
    """

    target: str
    op: ElementwiseImpl
    value: Expr

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))


@dataclass
class Sync:
    """``__syncthreads();`` — block-level barrier."""


@dataclass
class Cond:
    """``if (cond) { body } [else { else_body }]``.

    Used for the tid-bounds guard, causal-mask predicates, and any other
    coord-conditional branch. ``else_body`` empty means a bare ``if``.
    """

    cond: Expr
    body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...] = ()


# ---------------------------------------------------------------------------
# Statements — loops
# ---------------------------------------------------------------------------


@dataclass
class FreeLoop:
    """Per-thread serial walk over a free axis.

    Body sees ``Var(axis.name)`` bound to the current iteration value (a
    global coord). Lowering emits one ``FreeLoop`` per Loop IR free Loop;
    ``ExtractGlobalSchedule`` strips the outer FreeLoop chain into
    ``Kernel.thread_axes`` (those bindings then come from the tid decode
    rather than a loop var).
    """

    axis: Axis
    body: tuple[Stmt, ...]


@dataclass
class Acc:
    """One accumulator inside a ``Reduce.accs``.

    ``op`` is an ``ElementwiseImpl`` (Loop IR vocabulary: ``"add"`` /
    ``"multiply"`` / ``"maximum"`` / ``"minimum"`` plus ``"sum"`` / ``"prod"``
    aliases). The accumulator's initial value comes from ``op.identity`` —
    no separate ``init`` field, since the identity is uniquely determined
    by the combine.
    """

    name: str
    op: ElementwiseImpl

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))


@dataclass
class Reduce:
    """Register-accumulated walk over ``axis``.

    Declares each ``Acc`` as a register variable initialized to its
    identity, then walks the axis serially while the body ``AccumFold``\\ s
    into them. After the loop, the accumulator names are visible to
    subsequent stmts in the enclosing scope.

    ``extent`` is ``axis.extent`` for an unmodified Reduce; tiling shrinks
    it to BK while keeping the axis identity intact (the body still uses
    ``Var(axis.name)`` for the global coord).
    """

    axis: Axis
    accs: tuple[Acc, ...]
    body: tuple[Stmt, ...]
    extent: int | None = None


@dataclass
class Tile:
    """Outer slab walk: ``for k_outer = 0; k_outer < axis.extent; k_outer += bk``.

    Body sees ``Var(axis.name)`` bound to the slab origin (the outer-loop
    var). Inner stmts that need a per-element global coord typically wrap
    an inner ``Reduce(axis, extent=bk)`` whose body adds the inner var:
    ``Index("A", [m, axis + k_inner])``.
    """

    axis: Axis
    bk: int
    body: tuple[Stmt, ...]


@dataclass
class Coop:
    """Cooperative loop: ``for v = tid; v < cover; v += blockDim.x { body }``.

    The only thread-collective primitive in Tile IR. Used by smem-staging
    strategies to spread a load across all threads in the block. Body sees
    ``Var(var)`` bound to the per-iteration thread coord.
    """

    cover: int
    var: str
    body: tuple[Stmt, ...]


# Statement union — every node that can appear in a body sequence.
Stmt = Let | Store | AccumFold | Sync | Cond | FreeLoop | Reduce | Tile | Coop


# ---------------------------------------------------------------------------
# Top-level: Param / SmemBuf / Kernel
# ---------------------------------------------------------------------------


@dataclass
class Param:
    """Kernel parameter — global buffer pointer or scalar.

    ``dtype`` is the C declaration text (``"const float*"`` / ``"float*"`` /
    ``"int"`` / etc.); render emits it verbatim in the function signature.

    ``shape`` is the buffer's element shape, used by ``render_kernel`` to
    flatten multi-dim ``Index`` accesses to row-major. ``()`` means scalar
    (the parameter is passed by value, not as a pointer); a ``(d0, d1, ...)``
    tuple means a buffer of that shape behind ``dtype``'s pointer.
    """

    name: str
    dtype: str
    shape: tuple[int, ...] = ()


@dataclass
class SmemBuf:
    """Shared-memory buffer declared at kernel scope.

    Render emits ``__shared__ <dtype> <name>[d0][d1]...;`` at the top of
    the kernel function body. ``Index`` / ``Store`` referencing the buffer
    by ``name`` inside the kernel are resolved to smem via name lookup.
    """

    name: str
    dtype: str
    dims: tuple[int, ...]


@dataclass
class Kernel:
    """One ``__global__`` GPU kernel as a Tile IR program.

    Right after lowering: ``thread_axes == ()`` and ``block_axes == ()`` —
    the program is a single-thread serial walk. ``ExtractGlobalSchedule``
    populates ``thread_axes`` (and computes ``grid`` / ``block``); subsequent
    strategies extend ``smem`` and rewrite ``body``.

    ``prologue`` runs above the tid-bounds guard so all threads see the
    same value (typically scalar Loads of broadcast constants).
    """

    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]
    smem: tuple[SmemBuf, ...] = ()
    thread_axes: tuple[Axis, ...] = ()
    block_axes: tuple[Axis, ...] = ()
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    prologue: tuple[Stmt, ...] = ()


# Re-exports of the shared expression types so callers can do
# ``from deplodock.compiler.ir.tile import Var, Literal, ...``.
__all__ = [
    # Shared expressions
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Tile-IR expressions
    "Index",
    # Statements
    "Let",
    "Store",
    "AccumFold",
    "Sync",
    "Cond",
    "FreeLoop",
    "Reduce",
    "Tile",
    "Coop",
    "Stmt",
    # Top-level
    "Acc",
    "Param",
    "SmemBuf",
    "Kernel",
    # Re-exported from ir.loop / ir.elementwise
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff if not used yet (Kernel uses defaults instead)
