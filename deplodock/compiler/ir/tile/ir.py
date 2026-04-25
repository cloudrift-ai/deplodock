"""Tile IR — schedule + leaf compute, the IR that lowers directly to CUDA.

Tile IR sits between Loop IR (math) and CUDA source (text). Its job is to
encode *how* the math runs on the GPU — which axes go to threads / blocks —
without touching the math itself. Every scheduling decision is a
Tile-IR-to-Tile-IR rewrite; codegen consumes whatever final form comes out.

Pipeline shape:

    Loop IR ──lower_naive──▶ Tile IR (single-thread)
                  ──[strategy rewrites…]──▶ Tile IR
                  ──emit_cuda──▶ CUDA source

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` come straight from ``ir.loop`` — buf names are
strings (``Load.input``, ``Write.output``) so they're directly renderable.
``ElementwiseImpl`` carries op identity / commutativity / numpy callable.

**Schedule structure is Tile-IR-specific.** ``Enclosure`` plus the
``TileOp`` wrapper are new node types whose body is a broader union
(``Stmt``) that admits both Loop-IR leaves and Tile-IR additions.
Loop IR's ``Loop`` is reused directly for both free iteration and
reductions — a Loop is a reduce-Loop iff its body contains an ``Accum``
(detected structurally by the renderer / passes, never stored).
``Cond`` is reused for if/else. Strategies match on the schedule nodes;
the leaves they wrap pass through unchanged.

Expression types come from ``ir.expr`` directly (``Var``, ``Literal``,
``BinaryExpr``, ``FuncCallExpr``, ``TernaryExpr``, ``CastExpr``,
``Builtin``). No Tile-IR-specific expression nodes — ``Load`` / ``Write``
carry the buf name + axis-Var indices, and the renderer flattens to
row-major using the buffer's declared shape.
"""

from __future__ import annotations

from collections.abc import Iterator
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
    Stmt,
    Write,
)

# ---------------------------------------------------------------------------
# Tile-IR additions: schedule wrappers
# ---------------------------------------------------------------------------


@dataclass
class Enclosure(Stmt):
    """Bind enclosing axes to thread / block coords for the body.

    ``thread_axes`` are flattened into ``threadIdx.x`` (with a tid bounds
    guard for non-divisible cases). ``block_axes`` would be flattened into
    ``blockIdx.x/y/z`` (deferred until a strategy actually populates them).
    The body executes per-thread under those bindings; downstream stmts use
    ``Var(axis.name)`` and rely on the bindings to resolve at render time.

    Conceptually replaces ``Loop(axis)`` for the chosen axes — they iterate
    via thread/block parallelism instead of a serial for-loop. A single
    Enclosure typically appears as the schedule wrapper inside a ``TileOp``
    body; ``lower_naive`` builds it from the outer free-Loop chain.
    """

    thread_axes: tuple[Axis, ...]
    block_axes: tuple[Axis, ...]
    body: tuple[Stmt, ...]


# ---------------------------------------------------------------------------
# Top-level: TileOp
# ---------------------------------------------------------------------------


@dataclass
class TileOp(Op):
    """One ``__global__`` GPU kernel as a Tile IR program.

    Op subclass parallel to ``LoopOp``: lives as a graph node, carries a
    body of Tile IR statements plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies them
    at render time, same as ``LoopOp``. Kernel signature is derived from
    the body: distinct ``Load.input`` names become ``const float*`` params,
    distinct ``Write.output`` names become ``float*`` params, ordered by
    first appearance.

    Right after lowering, ``body`` either holds a fully-serial single-thread
    walk or — when ``lower_naive`` strips the outer free-Loop chain — an
    ``Enclosure(thread_axes=...)`` carrying the schedulable work. ``body``
    may contain stmts before any ``Enclosure`` (typically scalar ``Load``s
    of broadcast constants); render emits them at the top of the
    ``__global__`` function above the tid-bounds guard.
    """

    body: tuple[Stmt, ...] = ()
    name: str = ""

    def __iter__(self) -> Iterator[Stmt]:
        return iter_body(self.body)

    @property
    def loads(self) -> tuple[Load, ...]:
        return tuple(s for s in self if isinstance(s, Load))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order — the
        kernel's input parameters."""
        return tuple(dict.fromkeys(s.input for s in self.loads))

    @property
    def writes(self) -> tuple[Write, ...]:
        return tuple(s for s in self if isinstance(s, Write))

    @property
    def outputs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order —
        the kernel's writeable output parameters."""
        return tuple(dict.fromkeys(s.output for s in self.writes))


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    for s in body:
        yield s
        if isinstance(s, (Loop, Enclosure)):
            yield from iter_body(s.body)
        elif isinstance(s, Cond):
            yield from iter_body(s.body)
            yield from iter_body(s.else_body)


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
    "Enclosure",
    "Stmt",
    # Top-level
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff for the imported but optionally-used helper
