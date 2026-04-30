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
    Body,
    Cond,
    Load,
    Loop,
    RenderCtx,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Tile,
    Write,
    _pad,
    pretty_body,
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

    def pretty(self, indent: str = "") -> list[str]:
        ext = ", ".join(str(e) for e in self.extents) or "-"
        return [f"{indent}Smem {self.name}[{ext}] ({self.dtype})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``__shared__ <dtype> <name>[<prod(extents)>];`` and register the
        buffer's shape so subsequent ``Load``/``Write`` flatten correctly."""
        total = 1
        for e in self.extents:
            total *= int(e)
        ctx.shapes[self.name] = tuple(int(e) for e in self.extents)
        return [f"{_pad(ctx.indent)}__shared__ {self.dtype} {self.name}[{total}];"]


@dataclass
class Sync(Stmt):
    """``__syncthreads();`` — block-wide barrier."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Sync"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f"{_pad(ctx.indent)}__syncthreads();"]


@dataclass
class CpAsyncCopy(Stmt):
    """Issue one ``cp.async.cg.shared.global`` instruction.

    Replaces the per-thread ``Load(reg) + Write(smem)`` pair in cooperative
    loads on sm_80+. The hardware copies 4 bytes (one fp32) directly from
    global to shared without a register staging slot, freeing one thread
    register and removing the LDG → STS dependency.

    Renders to inline PTX. The asm reads the smem address via
    ``cvta.to.shared.u32`` and the global pointer as a 64-bit value;
    indices flatten via ``render_index`` against the buffer's declared
    shape (same as ``Load`` / ``Write``)."""

    smem: str  # destination smem buffer name
    smem_index: tuple
    src: str  # source global buffer name
    src_index: tuple

    def pretty(self, indent: str = "") -> list[str]:
        smem_idx = ", ".join(e.pretty() for e in self.smem_index)
        src_idx = ", ".join(e.pretty() for e in self.src_index)
        return [f"{indent}cp.async {self.smem}[{smem_idx}] <- {self.src}[{src_idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        from deplodock.compiler.ir.stmt import render_index

        smem_flat = render_index(self.smem, self.smem_index, ctx)
        src_flat = render_index(self.src, self.src_index, ctx)
        pad = _pad(ctx.indent)
        asm = f'asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\\n" :: "r"(_smem_addr), "l"(&{self.src}[{src_flat}]) : "memory");'
        return [
            f"{pad}{{",
            f"{pad}    unsigned int _smem_addr = __cvta_generic_to_shared(&{self.smem}[{smem_flat}]);",
            f"{pad}    {asm}",
            f"{pad}}}",
        ]


@dataclass
class CpAsyncCommit(Stmt):
    """``cp.async.commit_group;`` — finalize the preceding cp.async copies
    issued by this thread into a commit group. Pairs with
    ``CpAsyncWait`` to wait for that group to drain."""

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.commit_group"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.commit_group;\\n" ::: "memory");']


@dataclass
class CpAsyncWait(Stmt):
    """``cp.async.wait_group N;`` — block this thread until ≤ N cp.async
    groups remain in flight. ``group=0`` waits for everything (synchronous
    style); larger values stagger waits for software pipelining."""

    group: int = 0

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}cp.async.wait_group({self.group})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        return [f'{_pad(ctx.indent)}asm volatile("cp.async.wait_group {self.group};\\n" ::: "memory");']


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

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}TreeHalve({self.buf}, op={self.op.name}, length={self.length}, tid={self.tid_var})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Power-of-two tree reduction over ``buf[0..length)`` into ``buf[0]``."""
        pad = _pad(ctx.indent)
        inner_pad = _pad(ctx.indent + 1)
        halve_pad = _pad(ctx.indent + 2)
        op_expr = _binary_combine_expr(self.op, f"{self.buf}[{self.tid_var}]", f"{self.buf}[{self.tid_var} + s]")
        half = int(self.length) // 2
        return [
            f"{pad}for (int s = {half}; s > 0; s >>= 1) {{",
            f"{inner_pad}if ({self.tid_var} < s) {{",
            f"{halve_pad}{self.buf}[{self.tid_var}] = {op_expr};",
            f"{inner_pad}}}",
            f"{inner_pad}__syncthreads();",
            f"{pad}}}",
        ]


def _binary_combine_expr(op: ElementwiseImpl, a: str, b: str) -> str:
    """Render a 2-arg combine for ``ElementwiseImpl`` reduce ops."""
    name = op.name
    if name in ("add", "sum"):
        return f"{a} + {b}"
    if name in ("multiply", "prod"):
        return f"{a} * {b}"
    if name in ("maximum", "amax"):
        return f"fmaxf({a}, {b})"
    if name == "minimum":
        return f"fminf({a}, {b})"
    raise ValueError(f"TreeHalve: unsupported op {name!r}")


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

    body: Body = ()
    name: str = ""

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
    def smem_names(self) -> frozenset[str]:
        """Names of all ``__shared__`` buffers declared in the body — these
        are render-internal and are excluded from kernel-parameter inference."""
        return frozenset(s.name for s in self if isinstance(s, Smem))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order — the
        kernel's input parameters. Smem buffers are excluded.

        ``CpAsyncCopy`` stmts also count as global-buffer reads — their
        ``src`` field names a kernel parameter same as ``Load.input`` does
        for the synchronous path."""
        smem = self.smem_names
        names: dict[str, None] = {}
        for s in self:
            if isinstance(s, Load) and s.input not in smem:
                names.setdefault(s.input, None)
            elif isinstance(s, CpAsyncCopy) and s.src not in smem:
                names.setdefault(s.src, None)
        return tuple(names)

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
    "CpAsyncCopy",
    "CpAsyncCommit",
    "CpAsyncWait",
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
