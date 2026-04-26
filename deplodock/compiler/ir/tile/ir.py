"""Tile IR — schedule decisions as structural Stmts.

Tile IR sits between Loop IR (math) and Kernel IR (fully-scheduled
kernel form). Its job is to encode the *logical* compute plus the
*scheduling decisions* — without committing to hardware primitives.
Materialization (``passes/lowering/kernel``) consumes Tile IR and
produces Kernel IR.

Pipeline shape::

    Loop IR ──lower_naive──▶ Tile IR (logical compute, default bindings)
                     ──[strategy passes]──▶ Tile IR (annotated)
                     ──materialize_tile──▶ Kernel IR
                     ──render_kernelop──▶ CUDA source

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` / ``Cond`` come straight from ``ir.loop`` — buf
names are strings so they're directly renderable.

**Scheduling decisions live where they naturally belong**:

- ``Tile.thread_axes`` / ``Tile.block_axes`` — same shape as
  ``Tile``: which output axes are bound to thread coords vs CUDA
  block coords. Pointwise has ``thread_axes`` populated and
  ``block_axes`` empty (one thread per output element). Cooperative
  reductions have ``block_axes`` populated and ``thread_axes`` empty;
  the cooperative thread axis is synthesized at materialization.
- Loop constructs in the Tile-IR body are ``Loop`` (serial) and
  ``StridedLoop`` (cooperative — threads stride through the axis).
  Both are shared with Loop-IR / Kernel-IR via ``ir.stmt``.
- ``Combine`` — cross-thread collapse of an Accum target; sibling
  Stmt because it's buffer/accumulator-scoped, not axis-local.

The compute body is ``Loop`` / ``StridedLoop`` / ``Accum`` / ``Load`` /
``Assign`` / ``Write`` — a straight iteration tree.
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
from deplodock.compiler.ir.stmt import (
    Accum,
    Assign,
    Cond,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Tile,
    Write,
)

# ---------------------------------------------------------------------------
# Schedule-bearing Stmts
# ---------------------------------------------------------------------------
#
# Scheduling decisions are expressed via ``BoundAxis.bind`` values on
# ``Tile.axes`` (``BIND_THREAD`` / ``BIND_BLOCK`` for launch geometry)
# and via the choice of body loop construct (``Loop`` for serial,
# ``StridedLoop`` for cooperative striding).


# ``Tile`` is shared infrastructure — defined in ``ir/stmt.py`` and
# re-exported here. Used at Tile IR (with Stage / Combine in the body)
# and at Kernel IR (with Smem / Sync / TreeHalve after materialization).


# Tile-IR loop constructs are ``Loop`` (serial) and ``StridedLoop``
# (cooperative — threads of the block stride through the axis). Both
# come from ``ir.stmt`` directly; Tile IR doesn't add a wrapper.


@dataclass
class Combine(Stmt):
    """Cross-thread reduction of an ``Accum`` target.

    Placed immediately after a cooperative reduce loop (``StridedLoop``
    whose ``Accum`` produced ``name``). Materialization emits the
    cross-thread combine — smem tree-halve today; warp-shuffle / atomic
    in the future.

    ``op`` is a redundant copy of the matching ``Accum.op`` — kept as a
    cross-check; if the strategy constructs a Combine with the wrong op
    relative to the matching Accum, validation surfaces the bug.
    """

    name: str
    op: ElementwiseImpl


@dataclass
class Stage(Stmt):
    """Operand-cache declaration — stage a contiguous slab of ``buf``
    into a named local buffer for reuse across the surrounding ``Tile``
    body.

    Slab geometry:

    - ``origin`` — per-source-dim block-uniform anchor (length == source
      buffer rank). Each entry is an Expr referencing only block-bound
      Vars (``BIND_BLOCK`` axes) and Literals — no thread / cache vars.
    - ``axes`` — cache axes (smem layout, in this order).
    - ``slab_dims`` — parallel to ``axes``; each entry is the source-
      buffer dim that the slab axis adds to. ``source_index[d] =
      origin[d] + (slab axis at d, if any)``.

    SSA-like: ``name`` is the staged buffer's identifier; subsequent
    ``Load(input=name, index=cache-local)`` reads in the body refer to
    it directly. The strategy that inserts the Stage is also responsible
    for rewriting body Loads to target ``name`` with cache-local Vars
    (matching ``axes`` in order).

    The slab form maps directly to TMA / ``cp.async.bulk`` tensor-
    descriptor copies: ``origin`` is the box-origin and ``axes``
    extents are the box-extents.

    Doesn't commit to storage class — materialization picks (smem in
    today's path; TMA / async-copy paths possible in the future).
    """

    name: str
    buf: str
    origin: tuple[Expr, ...]
    axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]


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

    def pretty_body(self) -> str:
        """Render as an indented structural listing."""
        from deplodock.compiler.ir.expr import render as render_expr

        lines: list[str] = []

        def render_stmt(stmt: Stmt, indent: str, owned_axes: dict[str, str]) -> None:
            if isinstance(stmt, Tile):
                axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in stmt.axes) or "-"
                lines.append(f"{indent}Tile(axes=({axes})):")
                inner = {**owned_axes, **{ba.axis.name: ba.bind for ba in stmt.axes}}
                for s in stmt.body:
                    render_stmt(s, indent + "    ", inner)
                return
            if isinstance(stmt, StridedLoop):
                kind = "reduce" if stmt.is_reduce else "free"
                start = render_expr(stmt.start)
                step = render_expr(stmt.step) if hasattr(stmt.step, "__class__") and not isinstance(stmt.step, int) else stmt.step
                lines.append(f"{indent}StridedLoop({stmt.axis.name} = {start}; < {stmt.axis.extent}; += {step}):  # {kind}")
                for s in stmt.body:
                    render_stmt(s, indent + "    ", owned_axes)
                return
            if isinstance(stmt, Combine):
                lines.append(f"{indent}Combine({stmt.name}, op={stmt.op.name})")
                return
            if isinstance(stmt, Stage):
                origin = ", ".join(render_expr(e) for e in stmt.origin)
                slab = ", ".join(f"{ax.name}:{ax.extent}@{d}" for ax, d in zip(stmt.axes, stmt.slab_dims, strict=True))
                lines.append(f"{indent}{stmt.name} = Stage({stmt.buf}, origin=({origin}), slab=({slab}))")
                return
            if isinstance(stmt, Load):
                idx = ", ".join(render_expr(e) for e in stmt.index)
                lines.append(f"{indent}{stmt.name} = load {stmt.input}[{idx}]")
                return
            if isinstance(stmt, Assign):
                args = ", ".join(stmt.args)
                lines.append(f"{indent}{stmt.name} = {stmt.op.name}({args})")
                return
            if isinstance(stmt, Accum):
                lines.append(f"{indent}{stmt.name} <- {stmt.op.name}({stmt.name}, {stmt.value})")
                return
            if isinstance(stmt, Write):
                idx = ", ".join(render_expr(e) for e in stmt.index)
                lines.append(f"{indent}{stmt.output}[{idx}] = {stmt.value}")
                return
            if isinstance(stmt, Select):
                for bi, br in enumerate(stmt.branches):
                    prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                    lines.append(f"{indent}{prefix} {br.value} when ({render_expr(br.select)})")
                return
            if isinstance(stmt, Loop):
                kind = "reduce" if stmt.is_reduce else "free"
                lines.append(f"{indent}Loop({stmt.axis.name} in 0..{stmt.axis.extent}):  # {kind}")
                for s in stmt.body:
                    render_stmt(s, indent + "    ", owned_axes)
                return
            if isinstance(stmt, Cond):
                lines.append(f"{indent}if ({render_expr(stmt.cond)}):")
                for s in stmt.body:
                    render_stmt(s, indent + "    ", owned_axes)
                if stmt.else_body:
                    lines.append(f"{indent}else:")
                    for s in stmt.else_body:
                        render_stmt(s, indent + "    ", owned_axes)
                return
            lines.append(f"{indent}<unrecognized {type(stmt).__name__}>")

        sig_in = ", ".join(self.inputs) or "-"
        sig_out = ", ".join(self.outputs) or "-"
        lines.append(f"kernel {self.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}")
        for s in self.body:
            render_stmt(s, "    ", {})
        return "\n".join(lines)

    @property
    def loads(self) -> tuple[Load, ...]:
        return tuple(s for s in self if isinstance(s, Load))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct external-buffer names in body first-use order.

        A buffer is external if it's loaded from but not produced by a
        ``Stage`` in this TileOp. Loads of staged names are skipped
        (those bufs are smem-local at materialization). Stage source
        bufs (``Stage.buf``) are included — they're the actual external
        reads, performed by the cooperative load."""
        stage_names = {s.name for s in self if isinstance(s, Stage)}
        bufs: dict[str, None] = {}
        for s in self:
            if isinstance(s, Stage):
                bufs.setdefault(s.buf, None)
            elif isinstance(s, Load) and s.input not in stage_names:
                bufs.setdefault(s.input, None)
        return tuple(bufs)

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
    "StridedLoop",
    # Tile-IR statements
    "Tile",
    "Combine",
    "Stage",
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
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
