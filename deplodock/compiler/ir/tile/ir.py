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

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` come straight from ``ir.loop`` — buf names are now
strings (``Load.input``, ``Write.output``) so they're directly renderable.
``ElementwiseImpl`` carries op identity / commutativity / numpy callable.

**Schedule structure is Tile-IR-specific.** ``FreeLoop`` / ``Reduce`` /
``Tile`` / ``Coop`` / ``Sync`` / ``Cond`` plus the ``Kernel`` wrapper are
new node types whose body is a broader union (``Stmt``) that admits both
Loop-IR leaves and Tile-IR additions. Strategies match on the schedule
nodes; the leaves they wrap pass through unchanged.

Expression types come from ``ir.expr`` directly (``Var``, ``Literal``,
``BinaryExpr``, ``FuncCallExpr``, ``TernaryExpr``, ``CastExpr``,
``Builtin``). No Tile-IR-specific expression nodes — ``Load`` / ``Write``
carry the buf name + axis-Var indices, and the renderer flattens to
row-major using the buffer's declared shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Cond,
    Load,
    Select,
    SelectBranch,
    Write,
)

# ---------------------------------------------------------------------------
# Tile-IR additions: schedule wrappers + control flow
# ---------------------------------------------------------------------------


@dataclass
class Sync:
    """``__syncthreads();`` — block-level barrier."""


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
class Reduce:
    """Register-accumulated walk over ``axis``.

    The body contains zero or more ``Accum`` stmts (Loop IR's reduce
    primitive) that fold into named accumulators. The renderer scans the
    body for distinct ``Accum.name`` values, emits a ``float <name> =
    <op.identity>;`` declaration *before* the for-loop, then emits the body
    inside the for-loop. After the loop the accumulator names are visible
    to subsequent stmts in the enclosing scope.

    ``extent`` is ``axis.extent`` for an unmodified Reduce; tiling shrinks
    it to BK while keeping the axis identity intact (the body still uses
    ``Var(axis.name)`` for the global coord).
    """

    axis: Axis
    body: tuple[Stmt, ...]
    extent: int | None = None


@dataclass
class Tile:
    """Outer slab walk: ``for k_outer = 0; k_outer < axis.extent; k_outer += bk``.

    Body sees ``Var(axis.name)`` bound to the slab origin (the outer-loop
    var). Inner stmts that need a per-element global coord typically wrap
    an inner ``Reduce(axis, extent=bk)`` whose body adds the inner var:
    ``Load("x", input="A", index=(m, axis + k_inner))``.
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


# Statement union — Loop IR leaves + Tile IR additions. Every node that
# can appear in a body sequence anywhere in Tile IR. ``Cond`` (if/else)
# lives in Loop IR — it's reused here via the re-export.
Stmt = Load | Assign | Select | Write | Accum | Cond | Sync | FreeLoop | Reduce | Tile | Coop


# ---------------------------------------------------------------------------
# Top-level: Param / SmemBuf / Kernel
# ---------------------------------------------------------------------------


@dataclass
class Param:
    """Kernel parameter — global buffer pointer or scalar.

    ``dtype`` is the C declaration text (``"const float*"`` / ``"float*"`` /
    ``"int"`` / etc.); render emits it verbatim in the function signature.

    ``shape`` is the buffer's element shape, used by ``render_kernel`` to
    flatten multi-dim ``Load`` / ``Write`` indices to row-major. ``()``
    means scalar; a ``(d0, d1, ...)`` tuple means a buffer of that shape
    behind ``dtype``'s pointer.
    """

    name: str
    dtype: str
    shape: tuple[int, ...] = ()


@dataclass
class SmemBuf:
    """Shared-memory buffer declared at kernel scope.

    Render emits ``__shared__ <dtype> <name>[d0][d1]...;`` at the top of
    the kernel function body. ``Load`` / ``Write`` referencing the buffer
    by ``name`` inside the kernel resolve to smem via name lookup.
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


__all__ = [
    # Shared expressions (re-exported for convenience)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (re-exported — used as Tile IR statements)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    # Tile-IR statements
    "Sync",
    "FreeLoop",
    "Reduce",
    "Tile",
    "Coop",
    "Stmt",
    # Top-level
    "Param",
    "SmemBuf",
    "Kernel",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
