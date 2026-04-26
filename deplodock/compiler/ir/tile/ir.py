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

- ``Tile.thread_axes`` / ``Tile.block_axes`` — same shape as
  ``Enclosure``: which output axes are bound to thread coords vs CUDA
  block coords. Pointwise has ``thread_axes`` populated and
  ``block_axes`` empty (one thread per output element). Cooperative
  reductions have ``block_axes`` populated and ``thread_axes`` empty;
  the cooperative thread axis is synthesized at materialization.
- ``BoundLoop.axis.bind`` — always ``BIND_SERIAL``; cooperative
  iteration is expressed via axis splits in ``Tile.axes``, not a
  body-loop bind.
- ``Combine`` — cross-thread collapse of an Accum target; sibling
  Stmt because it's buffer/accumulator-scoped, not axis-local.

The compute body itself is ``BoundLoop`` / ``Accum`` / ``Load`` /
``Assign`` / ``Write`` — a straight iteration tree. Each ``BoundLoop``
carries its own binding; the body reads linearly top to bottom.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
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
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import (
    Accum,
    Assign,
    Cond,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    Write,
)

# ---------------------------------------------------------------------------
# Schedule-bearing Stmts
# ---------------------------------------------------------------------------
#
# Scheduling decisions are expressed via ``BoundAxis.bind`` values
# defined in ``ir.axis``. ``BIND_THREAD`` / ``BIND_BLOCK`` are the
# launch-geometry bindings (used in ``Tile.axes`` / ``Enclosure.axes``);
# ``BIND_SERIAL`` is the body-loop binding (used on ``BoundLoop.axis``).
# Cooperative iteration is expressed by splitting an axis into
# ``(chunk, t)`` at strategy time — the inner half ``t`` is a THREAD
# axis on ``Tile.axes`` and the outer half is iterated by a SERIAL
# ``BoundLoop`` whose body uses ``chunk * BLOCK_SIZE + t`` as the
# rewritten axis index.


@dataclass
class Tile(Stmt):
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
    """Iteration over an axis paired with its iteration policy.

    Tile-IR's variant of Loop-IR's ``Loop``: carries the same compute
    (a nested body) plus a ``BoundAxis`` whose ``bind`` is always
    ``BIND_SERIAL`` (each thread iterates the axis privately, renders
    to a plain ``for`` loop). Cooperative iteration is expressed by
    axis splits in ``Tile.axes``, not by a body-loop bind.

    Reduction detection is structural, same as Loop-IR's ``Loop``: a
    ``BoundLoop`` is a reduce-loop iff its body contains an ``Accum``.

    Disjoint from Loop-IR's ``Loop`` so materialization can convert
    between the two layers without ambiguity — post-materialization the
    Kernel IR body contains ``Loop`` (or ``StridedLoop`` for cooperative
    smem loads inside Stage expansion), never ``BoundLoop``.
    """

    axis: BoundAxis
    body: tuple[Stmt, ...]

    @property
    def bind(self) -> str:
        """Convenience accessor — the ``BoundAxis.bind`` value."""
        return self.axis.bind

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    def rewrite(self, rename_ssa, sigma: Sigma = Sigma.IDENTITY):  # type: ignore[override]
        return BoundLoop(
            axis=self.axis,
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
        )


@dataclass
class Combine(Stmt):
    """Cross-thread reduction of an ``Accum`` target.

    Placed immediately after the reduce ``BoundLoop`` whose ``Accum``
    produced ``name``. The *scope* of the combine — across the block,
    across a warp, etc. — is derived from the surrounding BoundLoop's
    ``bind``: the BoundLoop says "threads of this scope cooperatively
    walk the axis," and Combine says "now collapse the per-thread
    partials of that same scope." Materialization picks the mechanism
    (smem tree-halve today; warp-shuffle / atomic in the future) from
    the same surrounding bind.

    ``op`` is a redundant copy of the matching ``Accum.op`` — kept as a
    cross-check; if the strategy constructs a Combine with the wrong op
    relative to the matching Accum, validation surfaces the bug.
    """

    name: str
    op: ElementwiseImpl


@dataclass
class Stage(Stmt):
    """Operand-cache declaration: stage ``buf`` for reuse across the
    surrounding ``Tile`` body.

    Subsequent ``Load(buf, ...)`` reads in the body resolve to the
    staged copy, not the original buffer. ``index`` is the original
    source-buffer index pattern (with axis ``Var``s); ``axes`` lists the
    axes that *vary* within the staged fragment — the smem buffer's
    shape is ``tuple(ax.extent for ax in axes)``. Positions in
    ``index`` whose ``Var`` name appears in ``axes`` are the
    cache-dimension positions; other positions (typically block-bound
    axes from ``Tile.axes``) are fixed per CUDA block and contribute
    size 1 to the staged fragment.

    Doesn't commit to storage class — materialization picks (smem in
    today's path; register file or async-copy paths possible in the
    future). Subsequent body ``Load``s of ``buf`` get rewritten by
    materialization to read from the staged buffer with the
    cache-dimension positions of their original index.

    Inserted by the input-staging strategy when multiple ``BoundLoop``s
    in a cooperative ``Tile`` body Load the same buffer with
    block-bound dimensions in common — typical of softmax / norm-style
    fusions where the input is read three times.
    """

    buf: str
    index: tuple[Expr, ...]
    axes: tuple[Axis, ...]


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

from deplodock.compiler.ir.stmt import iter_body  # noqa: E402, F401

# Cooperative thread-block size — number of threads per CUDA block when a
# Tile uses BIND_THREAD axes from a cooperative strategy. Lives at this
# layer because cooperative strategies (cooperative-reduce, blockify)
# already commit to "this many threads cooperate" when they choose axis
# binds and tile sizes; materialization just consumes the choice.
BLOCK_SIZE = 256


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
    "Tile",
    "BoundLoop",
    "Combine",
    "Stage",
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "BIND_SERIAL",
    "Stmt",
    # Top-level
    "TileOp",
    # Scheduling constants
    "BLOCK_SIZE",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff
