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
- ``BoundLoop.walk`` — how an inner iteration axis is walked
  (``WALK_SERIAL`` = per-thread sequential, ``WALK_STRIDED`` =
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

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
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

# BoundLoop.walk values — how an inner iteration axis is walked.
WALK_SERIAL = "SERIAL"
WALK_STRIDED = "STRIDED"


# ---------------------------------------------------------------------------
# Schedule-bearing Stmts
# ---------------------------------------------------------------------------


@dataclass
class Block(Stmt):
    """Output-region wrapper — Tile-IR mirror of Kernel-IR ``Enclosure``.

    Carries the same ``axes: tuple[BoundAxis, ...]`` structure as
    ``Enclosure``: each output axis is paired with a binding
    (``BIND_THREAD`` = one thread per axis value; ``BIND_BLOCK`` = one
    CUDA block per axis value, threads inside cooperate).

    Pre-strategy default for any reducing kernel is every output axis
    bound to ``BIND_THREAD`` (one-thread-per-row). The cooperative-
    reduce strategy flips axes to ``BIND_BLOCK`` to opt into
    cooperative materialization, which will synthesize the cooperative
    thread axis (``t``) and prepend it to the resulting
    ``Enclosure.axes`` with binding ``BIND_THREAD``.

    The body holds the logical compute (``BoundLoop``, ``Accum``,
    ``Load``, ``Assign``, ``Write``) plus any ``Combine`` siblings
    placed by strategies.

    ``thread_axes`` / ``block_axes`` are convenience properties that
    project ``axes`` by binding kind — they're what the renderer and
    launch-geometry code consume.
    """

    axes: tuple[BoundAxis, ...]
    body: tuple[Stmt, ...]

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    @property
    def thread_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_THREAD)

    @property
    def block_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_BLOCK)


@dataclass(frozen=True)
class BoundLoop(Stmt):
    """Iteration over ``axis`` with an explicit walk strategy.

    Tile-IR's variant of Loop-IR's ``Loop``: carries the same compute
    (a nested body) plus a ``walk`` field saying how this axis is
    iterated. ``WALK_SERIAL`` is the pre-strategy default (each thread
    walks the axis itself). Strategies flip it to ``WALK_STRIDED`` for
    cooperative strided walks.

    ``walk`` is distinct from ``BoundAxis.bind``: ``bind`` says how an
    axis maps to parallel coords (THREAD/BLOCK); ``walk`` says how a
    thread iterates an axis sequentially (SERIAL/STRIDED). They share
    the "pick a value per axis" surface but encode orthogonal concepts.

    Reduction detection is structural, same as Loop-IR's ``Loop``: a
    ``BoundLoop`` is a reduce-loop iff its body contains an ``Accum``.

    Disjoint from Loop-IR's ``Loop`` so materialization can convert
    between the two layers without ambiguity — post-materialization the
    Kernel IR body contains ``Loop`` (serial) or ``StridedLoop``
    (strided), never ``BoundLoop``.
    """

    axis: Axis
    body: tuple[Stmt, ...]
    walk: str = WALK_SERIAL

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    def rewrite(self, rename_ssa, sigma: Sigma = Sigma.IDENTITY):  # type: ignore[override]
        return BoundLoop(
            axis=self.axis,
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
            walk=self.walk,
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
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------

from deplodock.compiler.ir.loop import iter_body  # noqa: E402, F401

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
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "WALK_SERIAL",
    "WALK_STRIDED",
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
