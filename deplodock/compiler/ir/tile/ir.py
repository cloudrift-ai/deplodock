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

from collections.abc import Callable, Iterator
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
    StridedLoop,
    Tile,
    Write,
    _axis_identity,
    pretty_body,
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
class AsyncWait(Stmt):
    """Wait for outstanding ``cp.async`` groups to drain past
    ``remaining`` (i.e. block until at most ``remaining`` async groups
    are still in flight) and barrier the CTA.

    Materialization emits ``CpAsyncWait(remaining)`` followed by
    ``Sync()``. The wait is per-thread (drains *this* thread's queue);
    the Sync makes the freshly-loaded smem visible to all threads in
    the CTA before any of them reads it.

    Used by the software-pipeline pass to express "compute on chunk N
    can begin once chunk N's load completes, but chunk N+1's load is
    allowed to remain outstanding"."""

    remaining: int = 0

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return AsyncWait(remaining=self.remaining)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}AsyncWait(remaining={self.remaining})"]


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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Combine(name=rename_ssa(self.name), op=self.op)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Combine({self.name}, op={self.op.name})"]


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
    # Optional: full reference Load index expressed in cache-axis Vars.
    # When present, materialization fetches source values by substituting
    # cache-axis decoded coords into this template — handles non-affine
    # layouts (``/``, ``%`` from collapsed-reshape views) that can't be
    # expressed as ``origin[d] + decoded[d]``. None falls back to the
    # additive ``origin + decoded_per_dim`` path.
    source_index_template: tuple[Expr, ...] | None = None
    # Per-cache-axis extra extent added to the smem allocation (not to the
    # cooperative-load extent). Empty tuple = no padding. Used by the bank-
    # conflict pass to break stride-aliased smem layouts: padding dim ``d``
    # by ``+1`` shifts every higher-stride row by one float, eliminating
    # 32-way bank conflicts that arise when adjacent threads stride through
    # power-of-2 multiples of bank width.
    pad: tuple[int, ...] = ()
    # Number of distinct smem buffers allocated for this Stage. ``1`` is the
    # ordinary single-buffer form. ``> 1`` enables ping-pong: each iteration
    # of the surrounding K-outer loop writes to slab ``[phase, ...]`` and
    # reads from the same. With ``buffer_count == 2`` and ``phase`` set to
    # an expression like ``a5 % 2``, consecutive iterations write to
    # different physical buffers, eliminating the leading
    # ``__syncthreads`` between prev-compute and next-load (because
    # they target different memory regions).
    buffer_count: int = 1
    # Expression naming the phase index for double-buffered Stages. Must
    # be set when ``buffer_count > 1``. Typically ``Var(K_outer_axis) %
    # buffer_count``.
    phase: Expr | None = None
    # When True, the cooperative load uses ``cp.async.cg.shared.global``
    # (one PTX instruction per thread per element) instead of
    # ``Load(reg) + Write(smem)``. Saves a register temp and uses the
    # async DRAM→smem path. Materialize emits a trailing
    # ``cp.async.commit_group`` + ``cp.async.wait_group(0)`` so the
    # behavior is synchronous w.r.t. the surrounding ``Sync``. Requires
    # sm_80+; the async-codegen pass sets this only when the target
    # supports it.
    async_load: bool = False
    # When True, this Stage participates in a software-pipelined
    # K-outer loop: the cooperative load is issued + committed but
    # *not* waited on. A surrounding ``AsyncWait`` Stmt (typically with
    # ``remaining = buffer_count - 1``) drains older groups while the
    # current iteration's load stays in flight, overlapping DRAM with
    # FMA. Only meaningful when ``async_load`` is also set.
    pipelined: bool = False

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        new_origin = tuple(sigma.apply(e) for e in self.origin)
        new_axes = tuple(axis_fn(a) for a in self.axes)
        new_template = tuple(sigma.apply(e) for e in self.source_index_template) if self.source_index_template is not None else None
        return Stage(
            name=self.name,
            buf=self.buf,
            origin=new_origin,
            axes=new_axes,
            slab_dims=self.slab_dims,
            source_index_template=new_template,
            pad=self.pad,
            buffer_count=self.buffer_count,
            phase=sigma.apply(self.phase) if self.phase is not None else None,
            async_load=self.async_load,
            pipelined=self.pipelined,
        )

    def pretty(self, indent: str = "") -> list[str]:
        origin = ", ".join(e.pretty() for e in self.origin)
        slab = ", ".join(f"{ax.name}:{ax.extent}@{d}" for ax, d in zip(self.axes, self.slab_dims, strict=True))
        pad = f" pad=({', '.join(str(p) for p in self.pad)})" if self.pad and any(self.pad) else ""
        buf = f" buffers={self.buffer_count}@{self.phase.pretty()}" if self.buffer_count > 1 and self.phase is not None else ""
        async_tag = " async" if self.async_load else ""
        pipe_tag = " pipelined" if self.pipelined else ""
        return [f"{indent}{self.name} = Stage({self.buf}, origin=({origin}), slab=({slab})){pad}{buf}{async_tag}{pipe_tag}"]


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

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.stmt import normalize_body

        new_body = normalize_body(self.body, hoist=False)
        if new_body != self.body:
            self.body = new_body

    def __iter__(self) -> Iterator[Stmt]:
        return iter_body(self.body)

    def pretty_body(self) -> str:
        """Render as an indented structural listing via per-stmt ``pretty``."""
        sig_in = ", ".join(self.inputs) or "-"
        sig_out = ", ".join(self.outputs) or "-"
        head = f"kernel {self.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}"
        return "\n".join([head, *pretty_body(self.body, "    ")])

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
    "AsyncWait",
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
