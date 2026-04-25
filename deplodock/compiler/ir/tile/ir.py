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

**Schedule structure is Tile-IR-specific.** ``Reduce`` / ``Tile`` /
``Coop`` / ``Sync`` plus the ``Kernel`` wrapper are new node types whose
body is a broader union (``Stmt``) that admits both Loop-IR leaves and
Tile-IR additions. Loop IR's ``Loop`` is reused directly for free
iteration; ``Cond`` for if/else. Strategies match on the schedule nodes;
the leaves they wrap pass through unchanged.

Expression types come from ``ir.expr`` directly (``Var``, ``Literal``,
``BinaryExpr``, ``FuncCallExpr``, ``TernaryExpr``, ``CastExpr``,
``Builtin``). No Tile-IR-specific expression nodes — ``Load`` / ``Write``
carry the buf name + axis-Var indices, and the renderer flattens to
row-major using the buffer's declared shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op
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
    Loop,
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


# Free iteration uses Loop IR's ``Loop`` directly — see the ``Loop`` re-export.
# A future ``Enclosure`` will replace ``Loop`` here for axes bound to thread /
# block / cooperative coords.


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


@dataclass
class Enclosure:
    """Bind enclosing axes to thread / block coords for the body.

    ``thread_axes`` are flattened into ``threadIdx.x`` (with a tid bounds
    guard for non-divisible cases). ``block_axes`` would be flattened into
    ``blockIdx.x/y/z`` (deferred until a strategy actually populates them).
    The body executes per-thread under those bindings; downstream stmts use
    ``Var(axis.name)`` and rely on the bindings to resolve at render time.

    Conceptually replaces ``Loop(axis)`` for the chosen axes — they iterate
    via thread/block parallelism instead of a serial for-loop. A single
    Enclosure typically appears as the schedule wrapper inside a ``TileOp``
    body; ``ExtractGlobalSchedule`` builds it from the outer free-Loop chain.
    """

    thread_axes: tuple[Axis, ...]
    block_axes: tuple[Axis, ...]
    body: tuple[Stmt, ...]


# Statement union — Loop IR leaves + Tile IR additions. Every node that
# can appear in a body sequence anywhere in Tile IR. ``Cond`` and ``Loop``
# (free iteration) live in Loop IR — reused here via re-exports.
Stmt = Load | Assign | Select | Write | Accum | Cond | Loop | Sync | Reduce | Tile | Coop | Enclosure


# ---------------------------------------------------------------------------
# Top-level: SmemBuf / TileOp
# ---------------------------------------------------------------------------


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
class TileOp(Op):
    """One ``__global__`` GPU kernel as a Tile IR program.

    Op subclass parallel to ``LoopOp``: lives as a graph node, carries a
    body of Tile IR statements plus kernel-level metadata (``smem``, ``name``).

    Buffer shapes are *not* baked in — the surrounding graph supplies them
    at render time, same as ``LoopOp``. Kernel signature is derived from
    the body: distinct ``Load.input`` names become ``const float*`` params,
    distinct ``Write.output`` names become ``float*`` params, ordered by
    first appearance.

    Right after lowering, ``body`` either holds a fully-serial single-thread
    walk or — when ``lower_naive`` strips the outer free-Loop chain — an
    ``Enclosure(thread_axes=...)`` carrying the schedulable work. Subsequent
    strategies (``TileReduce``, ``SmemStageReduce``, …) rewrite the body
    and extend ``smem``. ``body`` may contain stmts before any ``Enclosure``
    (typically scalar ``Load``s of broadcast constants); render emits them
    at the top of the ``__global__`` function above the tid-bounds guard.
    """

    body: tuple[Stmt, ...] = ()
    smem: tuple[SmemBuf, ...] = ()
    name: str = ""

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order — the
        kernel's input parameters."""
        return _distinct_in_order(s.input for s in _walk_stmts(self.body) if isinstance(s, Load))

    @property
    def output_bufs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order —
        the kernel's writeable output parameters."""
        return _distinct_in_order(s.output for s in _walk_stmts(self.body) if isinstance(s, Write))


def _distinct_in_order(items) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for x in items:
        if x not in seen:
            seen[x] = None
    return tuple(seen)


def _walk_stmts(stmts):
    """Pre-order walk over Tile IR body — yields every nested stmt."""
    for s in stmts:
        yield s
        sub = getattr(s, "body", None)
        if isinstance(sub, tuple):
            yield from _walk_stmts(sub)
        # Cond has both body and else_body
        else_body = getattr(s, "else_body", None)
        if isinstance(else_body, tuple):
            yield from _walk_stmts(else_body)


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
    "Loop",
    # Tile-IR statements
    "Sync",
    "Reduce",
    "Tile",
    "Coop",
    "Enclosure",
    "Stmt",
    # Top-level
    "SmemBuf",
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff for the imported but optionally-used helper
