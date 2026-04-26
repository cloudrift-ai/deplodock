"""Kernel IR — the fully-scheduled kernel form, just above CUDA source.

Kernel IR sits between Tile IR (schedule decisions as structural Stmts)
and CUDA source (text). Its body contains the explicit hardware
machinery: ``Tile`` (thread/block coord bindings), ``Smem``
(``__shared__`` arrays), ``Sync`` (``__syncthreads`` barriers),
``TreeHalve`` (cross-thread reduction over smem), ``StridedLoop``
(strided per-thread loop).

Pipeline shape::

    Tile IR ──materialize_tile──▶ Kernel IR
                    ──render_kernelop──▶ CUDA source

**Leaf compute reuses Loop IR directly**. ``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond`` / ``Loop`` come straight
from ``ir.loop`` — buf names are strings so they're directly renderable.

Kernel IR deliberately contains no scheduling decisions — those live in
Tile IR and are materialized away before reaching this layer. A
``KernelOp`` is what the CUDA backend turns into a ``RawKernel`` launch.
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
# Hardware primitives
# ---------------------------------------------------------------------------


@dataclass
class Smem(Stmt):
    """Declare a per-block ``__shared__`` array.

    Renders to ``__shared__ <dtype> <name>[<prod(extents)>];``. ``extents``
    is the multi-dim shape used to flatten ``Load`` / ``Write`` indices
    against this buffer. smem_bytes for ``CudaOp`` is computed by walking
    the KernelOp body and summing ``prod(extents) * sizeof(dtype)`` across
    distinct ``Smem`` declarations.
    """

    name: str
    extents: tuple[int, ...]
    dtype: str = "float"


@dataclass
class Sync(Stmt):
    """``__syncthreads();`` — block-wide barrier."""


@dataclass
class TreeHalve(Stmt):
    """Cooperative power-of-two tree reduction over a 1D smem buffer.

    Reduces ``buf[0..length)`` into ``buf[0]`` using ``op`` as the combine.
    ``tid_var`` names the cooperative thread axis. ``length`` must be a
    power of two and ``≤ blockDim.x``.
    """

    buf: str
    op: ElementwiseImpl
    length: int
    tid_var: str


# ``StridedLoop`` is shared infrastructure — defined in ``ir/stmt.py``
# and re-exported here. Used at Tile IR for cooperative iteration and
# at Kernel IR for cooperative smem loads.


# ---------------------------------------------------------------------------
# Top-level: KernelOp
# ---------------------------------------------------------------------------


@dataclass
class KernelOp(Op):
    """One ``__global__`` GPU kernel as a Kernel IR program.

    Op subclass parallel to ``TileOp`` / ``LoopOp``: lives as a graph
    node, carries a body of Kernel IR stmts plus a kernel name.

    Buffer shapes are *not* baked in — the surrounding graph supplies
    them at render time, same as ``TileOp``. Kernel signature is derived
    from the body: distinct ``Load.input`` names become ``const float*``
    params, distinct ``Write.output`` names become ``float*`` params,
    ordered by first appearance. ``Smem`` buffers are excluded.
    """

    body: tuple[Stmt, ...] = ()
    name: str = ""

    def __iter__(self) -> Iterator[Stmt]:
        return iter_body(self.body)

    def pretty_body(self) -> str:
        """Render as an indented structural listing."""
        from deplodock.compiler.ir.expr import render as render_expr

        lines: list[str] = []

        def render_stmt(stmt: Stmt, indent: str) -> None:
            if isinstance(stmt, Tile):
                axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in stmt.axes) or "-"
                lines.append(f"{indent}Tile(axes=({axes})):")
                for s in stmt.body:
                    render_stmt(s, indent + "    ")
                return
            if isinstance(stmt, Smem):
                ext = ", ".join(str(e) for e in stmt.extents) or "-"
                lines.append(f"{indent}Smem {stmt.name}[{ext}] ({stmt.dtype})")
                return
            if isinstance(stmt, Sync):
                lines.append(f"{indent}Sync")
                return
            if isinstance(stmt, TreeHalve):
                lines.append(f"{indent}TreeHalve({stmt.buf}, op={stmt.op.name}, length={stmt.length}, tid={stmt.tid_var})")
                return
            if isinstance(stmt, StridedLoop):
                kind = "reduce" if stmt.is_reduce else "free"
                start = render_expr(stmt.start)
                lines.append(f"{indent}StridedLoop({stmt.axis.name} = {start}; < {stmt.axis.extent}; += {stmt.step}):  # {kind}")
                for s in stmt.body:
                    render_stmt(s, indent + "    ")
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
                    render_stmt(s, indent + "    ")
                return
            if isinstance(stmt, Cond):
                lines.append(f"{indent}if ({render_expr(stmt.cond)}):")
                for s in stmt.body:
                    render_stmt(s, indent + "    ")
                if stmt.else_body:
                    lines.append(f"{indent}else:")
                    for s in stmt.else_body:
                        render_stmt(s, indent + "    ")
                return
            lines.append(f"{indent}<unrecognized {type(stmt).__name__}>")

        sig_in = ", ".join(self.inputs) or "-"
        sig_out = ", ".join(self.outputs) or "-"
        lines.append(f"kernel {self.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}")
        for s in self.body:
            render_stmt(s, "    ")
        return "\n".join(lines)

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
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------

from deplodock.compiler.ir.stmt import iter_body  # noqa: E402, F401

__all__ = [
    # Shared expressions (re-exported)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (reused)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Kernel-IR statements
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "StridedLoop",
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "Stmt",
    # Top-level
    "KernelOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]


_ = field  # silence ruff
