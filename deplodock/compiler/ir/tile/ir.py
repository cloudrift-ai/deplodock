"""Tile IR — schedule decisions as structural Stmts.

Tile IR sits between Loop IR (math) and Kernel IR (fully-scheduled
kernel form). Its job is to encode the *logical* compute plus the
*scheduling decisions* — without committing to hardware primitives.
Materialization (``passes/lowering/kernel``) consumes Tile IR and
produces Kernel IR.

Pipeline shape::

    Loop IR ──lower_naive──▶ Tile IR (logical compute, default bindings)
                     ──[strategy passes]──▶ Tile IR (annotated)
                     ──materialize_block──▶ Kernel IR
                     ──render_kernelop──▶ CUDA source

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` / ``Cond`` come straight from ``ir.loop`` — buf
names are strings so they're directly renderable.

**Scheduling decisions live where they naturally belong**:

- ``Block.thread_axes`` / ``Block.block_axes`` — same shape as
  ``Enclosure``: which output axes are bound to thread coords vs CUDA
  block coords. Pointwise has ``thread_axes`` populated and
  ``block_axes`` empty (one thread per output element). Cooperative
  reductions have ``block_axes`` populated and ``thread_axes`` empty;
  the cooperative thread axis is synthesized at materialization.
- ``BoundLoop.bind`` — how an inner iteration axis is walked
  (``BIND_SERIAL`` = per-thread sequential, ``BIND_STRIDED`` =
  cooperative strided walk across the block's threads).
- ``Combine`` — cross-thread collapse of an Accum target; sibling
  Stmt because it's buffer/accumulator-scoped, not axis-local.

The compute body itself is ``BoundLoop`` / ``Accum`` / ``Load`` /
``Assign`` / ``Write`` — a straight iteration tree. Each ``BoundLoop``
carries its own binding; the body reads linearly top to bottom.
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
    Sigma,
    Stmt,
    Write,
)

# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

# BoundLoop.bind values — how an inner iteration axis is walked.
BIND_SERIAL = "SERIAL"
BIND_STRIDED = "STRIDED"


# ---------------------------------------------------------------------------
# Schedule-bearing Stmts
# ---------------------------------------------------------------------------


@dataclass
class Block(Stmt):
    """Output-region wrapper — Tile-IR mirror of Kernel-IR ``Enclosure``.

    Carries the same ``thread_axes`` / ``block_axes`` structure as
    ``Enclosure``: axes in ``thread_axes`` are bound to threads (one
    thread per output point); axes in ``block_axes`` are bound to CUDA
    blocks (one block per output point, threads inside cooperate).

    Pre-strategy default for any reducing kernel is
    ``thread_axes=output_axes`` / ``block_axes=()`` (one-thread-per-row).
    The cooperative-reduce strategy moves the axes from ``thread_axes``
    to ``block_axes`` to opt into cooperative materialization, which
    will synthesize the cooperative thread axis (``t``) and place it in
    the resulting ``Enclosure.thread_axes``.

    The body holds the logical compute (``BoundLoop``, ``Accum``,
    ``Load``, ``Assign``, ``Write``) plus any ``Combine`` siblings
    placed by strategies.
    """

    thread_axes: tuple[Axis, ...]
    block_axes: tuple[Axis, ...]
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class BoundLoop(Stmt):
    """Iteration over ``axis`` with an explicit binding to GPU coords.

    Tile-IR's variant of Loop-IR's ``Loop``: carries the same compute
    (a nested body) plus a ``bind`` field saying how this axis is
    walked. ``BIND_SERIAL`` is the pre-strategy default (each thread
    walks the axis itself). Strategies flip it to ``BIND_STRIDED`` for
    cooperative walks.

    Reduction detection is structural, same as Loop-IR's ``Loop``: a
    ``BoundLoop`` is a reduce-loop iff its body contains an ``Accum``.

    Disjoint from Loop-IR's ``Loop`` so materialization can convert
    between the two layers without ambiguity — post-materialization the
    Kernel IR body contains ``Loop`` (serial) or ``StridedLoop``
    (strided), never ``BoundLoop``.
    """

    axis: Axis
    body: tuple[Stmt, ...]
    bind: str = BIND_SERIAL

    def rewrite(self, rename_ssa, sigma: Sigma = Sigma.IDENTITY):  # type: ignore[override]
        return BoundLoop(
            axis=self.axis,
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
            bind=self.bind,
        )


# Combine ``via`` values — how an Accum's per-thread partials collapse.
COMBINE_REGISTER = "REGISTER"
COMBINE_SMEM_TREE_HALVE = "SMEM_TREE_HALVE"


@dataclass
class Combine(Stmt):
    """Cross-thread reduction of an ``Accum`` target.

    Placed after the reduce ``BoundLoop`` whose ``Accum`` produced
    ``name``. ``via`` says how the partials collapse:

    ``REGISTER`` — no cross-thread combine (each thread owns its own
    output element; its register is the final value).
    ``SMEM_TREE_HALVE`` — stage partials into a per-block smem buffer,
    tree-halve across threads. Materialization emits ``Smem`` +
    ``Write-to-smem`` + ``Sync`` + ``TreeHalve`` + ``Sync`` + broadcast
    ``Load`` in Kernel IR; subsequent reads of ``name`` resolve to the
    broadcast load.
    """

    name: str
    op: ElementwiseImpl
    via: str  # COMBINE_REGISTER | COMBINE_SMEM_TREE_HALVE


# ---------------------------------------------------------------------------
# Top-level: TileOp
# ---------------------------------------------------------------------------


@dataclass
class TileOp(Op):
    """One GPU kernel as a Tile IR program — pre-materialization.

    Op subclass parallel to ``LoopOp``: lives as a graph node, carries a
    body of Tile IR statements plus a kernel name. Materialization turns
    a ``TileOp`` into a ``KernelOp``.
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
        """Distinct ``Load.input`` buf names in body first-use order."""
        return tuple(dict.fromkeys(s.input for s in self.loads))

    @property
    def writes(self) -> tuple[Write, ...]:
        return tuple(s for s in self if isinstance(s, Write))

    @property
    def outputs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order."""
        return tuple(dict.fromkeys(s.output for s in self.writes))


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    for s in body:
        yield s
        if isinstance(s, (Loop, Block, BoundLoop)):
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
    # Loop-IR leaves + control flow (re-exported)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Tile-IR statements
    "Block",
    "BoundLoop",
    "Combine",
    # Binding + Combine kind constants
    "BIND_SERIAL",
    "BIND_STRIDED",
    "COMBINE_REGISTER",
    "COMBINE_SMEM_TREE_HALVE",
    "Stmt",
    # Top-level
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff
