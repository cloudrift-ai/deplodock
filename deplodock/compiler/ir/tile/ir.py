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

**Schedule structure is Tile-IR-specific.** ``Enclosure``, ``Tile``,
plus the ``TileOp`` wrapper are new node types whose body is a broader
union (``Stmt``) that admits both Loop-IR leaves and Tile-IR additions.
Loop IR's ``Loop`` is reused directly for both free iteration and
reductions — a Loop is a reduce-Loop iff its body contains an ``Accum``
(detected structurally by the renderer / passes, never stored).
``Cond`` is reused for if/else. Strategies match on the schedule nodes;
the leaves they wrap pass through unchanged.

``Tile`` marks a cooperative block: its body shares a per-block scratch
indexed by ``live_axes`` (the free axes still in scope inside any
reduction). Sibling stmts execute in order; render is responsible for
inserting an implicit ``__syncthreads()`` between siblings whenever a
later sibling reads what an earlier one wrote. ``Accum`` targets inside
a ``Tile`` resolve to slots in the block's scratch — registers when one
thread per slot, ``__shared__`` when multiple threads per slot. Today
every cooperative block runs at one-thread-per-slot, so ``Tile`` renders
as pass-through; smem + barriers land once a strategy chooses a wider
block.

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
    Sigma,
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


@dataclass
class Tile(Stmt):
    """Cooperative block: body stmts share a per-block scratch indexed by ``live_axes``.

    ``live_axes`` are the free axes still in scope inside the block — i.e.
    the surviving thread axes whose values address one slot of scratch
    per output element. ``extents`` are the corresponding slot counts;
    today ``extents == tuple(ax.extent for ax in live_axes)`` because no
    strategy has reshaped the block yet.

    Sibling stmts execute in order. Synchronization is *explicit* via
    ``Sync`` Stmts placed by strategy passes — the renderer never infers
    barriers. ``Accum`` targets inside a ``Tile`` resolve to per-thread
    registers; cooperative reduction across threads is expressed by the
    strategy as ``Smem`` + a strided ``StridedLoop`` + ``Accum`` (per-thread
    partial) + assign-into-smem + ``Sync`` + ``TreeHalve`` + a ``Cond``
    guarding the final ``Write``.
    """

    live_axes: tuple[Axis, ...]
    extents: tuple[int, ...]
    body: tuple[Stmt, ...]


# ---------------------------------------------------------------------------
# Tile-IR primitives — smem, sync, tree-halve, strided loop
# ---------------------------------------------------------------------------


@dataclass
class Smem(Stmt):
    """Declare a per-block ``__shared__`` array.

    Renders to ``__shared__ <dtype> <name>[<prod(extents)>];`` at the
    point of declaration inside a ``Tile`` body. ``extents`` is the
    multi-dim shape used to flatten ``Load`` / ``Write`` indices against
    this buffer (the renderer treats the buffer name like any other
    indexed buffer once declared — ``ctx.shapes`` is updated for the
    enclosing scope).

    smem_bytes for ``CudaOp`` is computed by walking the TileOp body and
    summing ``prod(extents) * sizeof(dtype)`` across distinct ``Smem``
    declarations.
    """

    name: str
    extents: tuple[int, ...]
    dtype: str = "float"


@dataclass
class Sync(Stmt):
    """``__syncthreads();`` — block-wide barrier.

    Strategies place these explicitly. The renderer emits one line per
    ``Sync``; no inference, no deduplication. A future Tile-IR pass may
    coalesce adjacent ``Sync``s.
    """


@dataclass
class TreeHalve(Stmt):
    """Cooperative power-of-two tree reduction over a 1D smem buffer.

    Reduces ``buf[0..length)`` into ``buf[0]`` using ``op`` as the combine.
    ``tid_var`` names the cooperative thread axis (the participating
    threadIdx). Renders to::

        for (int s = length/2; s > 0; s >>= 1) {
            if (tid_var < s) buf[tid_var] = op(buf[tid_var], buf[tid_var + s]);
            __syncthreads();
        }

    ``length`` must be a power of two and ``≤ blockDim.x`` (the strategy
    chooses these together with the cooperative thread-axis extent).
    """

    buf: str
    op: ElementwiseImpl
    length: int
    tid_var: str


@dataclass
class StridedLoop(Stmt):
    """``for (int <axis.name> = <start>; <axis.name> < <axis.extent>; <axis.name> += <step>)``.

    Tile-IR-specific strided variant of ``Loop`` used by cooperative
    reductions to walk a reduction axis in ``step``-sized slabs across
    threads (typically ``start = Var("t")`` and ``step = blockDim.x``).
    Reduction detection is identical to ``Loop``: a ``StridedLoop`` is a
    reduce-loop iff its body contains an ``Accum``.
    """

    axis: Axis
    start: Expr
    step: int
    body: tuple[Stmt, ...]

    def rewrite(self, rename_ssa, sigma: Sigma = Sigma.IDENTITY):  # type: ignore[override]
        """Recursive rewrite mirroring ``Loop.rewrite``: rebuild ``body``
        applying ``rename_ssa`` / ``sigma`` to every child. ``start`` is
        an Expr — sigma-substituted; ``axis`` and ``step`` are left as-is.
        """
        return StridedLoop(
            axis=self.axis,
            start=sigma.apply(self.start),
            step=self.step,
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
        )


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
    def smem_names(self) -> frozenset[str]:
        """Names of all ``__shared__`` buffers declared in the body — these
        are render-internal and are excluded from kernel-parameter inference."""
        return frozenset(s.name for s in self if isinstance(s, Smem))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order — the
        kernel's input parameters. Smem buffers are excluded."""
        smem = self.smem_names
        return tuple(dict.fromkeys(s.input for s in self.loads if s.input not in smem))

    @property
    def writes(self) -> tuple[Write, ...]:
        return tuple(s for s in self if isinstance(s, Write))

    @property
    def outputs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order —
        the kernel's writeable output parameters. Smem buffers are excluded."""
        smem = self.smem_names
        return tuple(dict.fromkeys(s.output for s in self.writes if s.output not in smem))


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    for s in body:
        yield s
        if isinstance(s, (Loop, StridedLoop, Enclosure, Tile)):
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
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "StridedLoop",
    "Stmt",
    # Top-level
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff for the imported but optionally-used helper
