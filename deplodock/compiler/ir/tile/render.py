"""Tile IR → CUDA source.

Renders a ``Kernel`` to a complete ``extern "C" __global__`` CUDA function
text. The renderer is purely mechanical: every Tile IR node has a fixed
emission rule. Strategies that produce different Tile IR (smem-staged,
tiled, split-K, etc.) all flow through this same renderer; codegen has
no schedule awareness.

Translation tables (CUDA spelling for builtin / intrinsic names) and the
expression printer with operator-precedence-aware parenthesization are
duplicated from ``pipeline/passes/lowering/cuda/_emit.py``. Step 6 of the
Tile IR refactor deletes the old emitter and consolidates these tables;
until then, both copies live in parallel — kept in sync by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.tile.ir import (
    AccumFold,
    Cond,
    Coop,
    Expr,
    FreeLoop,
    Index,
    Kernel,
    Let,
    Reduce,
    SmemBuf,
    Stmt,
    Store,
    Sync,
    Tile,
)

# ---------------------------------------------------------------------------
# CUDA spelling translation — TODO step 6: dedupe with cuda/_emit.py
# ---------------------------------------------------------------------------

_BUILTIN_TO_CUDA: dict[str, str] = {
    "thread_idx.x": "threadIdx.x",
    "thread_idx.y": "threadIdx.y",
    "thread_idx.z": "threadIdx.z",
    "block_idx.x": "blockIdx.x",
    "block_idx.y": "blockIdx.y",
    "block_idx.z": "blockIdx.z",
    "block_dim.x": "blockDim.x",
    "block_dim.y": "blockDim.y",
    "block_dim.z": "blockDim.z",
    "grid_dim.x": "gridDim.x",
    "grid_dim.y": "gridDim.y",
    "grid_dim.z": "gridDim.z",
    "warp_size": "warpSize",
}

_INTRINSIC_TO_CUDA: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "abs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
}

_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    ">": 4,
    "<=": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


# ---------------------------------------------------------------------------
# Render context
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Per-kernel render state.

    ``shapes`` is the union of param shapes and smem dims, used to flatten
    multi-dim ``Index`` accesses to row-major. Built once at the start of
    ``render_kernel``. ``indent`` is the current indent level (tracked
    explicitly so each helper formats with the correct leading whitespace).
    """

    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1


def _pad(indent: int) -> str:
    return "    " * indent


# ---------------------------------------------------------------------------
# Top-level: render_kernel
# ---------------------------------------------------------------------------


def render_kernel(kernel: Kernel) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function."""
    shapes: dict[str, tuple[int, ...]] = {}
    for p in kernel.params:
        if p.shape:
            shapes[p.name] = p.shape
    for s in kernel.smem:
        shapes[s.name] = s.dims
    ctx = _Ctx(shapes=shapes, indent=1)

    # Header.
    params_text = ", ".join(f"{p.dtype} {p.name}" for p in kernel.params)
    bx, by, bz = kernel.block
    max_threads = bx * by * bz
    launch_bounds = f"\n__launch_bounds__({max_threads})" if max_threads <= 1024 else ""

    body_lines: list[str] = []

    # Smem decls.
    for sb in kernel.smem:
        body_lines.append(_render_smem_decl(sb, ctx))

    # Prologue (no tid guard).
    for s in kernel.prologue:
        body_lines.extend(_render_stmt(s, ctx))

    # Tid decode + bounds guard.
    if kernel.thread_axes:
        n_threads = 1
        for ax in kernel.thread_axes:
            n_threads *= int(ax.extent)
        body_lines.append(f"{_pad(ctx.indent)}long long tid = blockIdx.x * blockDim.x + threadIdx.x;")
        body_lines.append(f"{_pad(ctx.indent)}if (tid < {n_threads}) {{")
        # Axis bindings from tid (row-major, innermost axis = tid % extent).
        guarded = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1)
        body_lines.extend(_render_thread_axis_decode(kernel.thread_axes, guarded))
        for s in kernel.body:
            body_lines.extend(_render_stmt(s, guarded))
        body_lines.append(f"{_pad(ctx.indent)}}}")
    else:
        for s in kernel.body:
            body_lines.extend(_render_stmt(s, ctx))

    body_text = "\n".join(body_lines)
    return f'extern "C" __global__{launch_bounds} void {kernel.name}({params_text}) {{\n{body_text}\n}}\n'


def _render_smem_decl(sb: SmemBuf, ctx: _Ctx) -> str:
    dims = "".join(f"[{d}]" for d in sb.dims)
    return f"{_pad(ctx.indent)}__shared__ {sb.dtype} {sb.name}{dims};"


def _render_thread_axis_decode(axes: tuple, ctx: _Ctx) -> list[str]:
    """Emit ``int <axis> = (tid / stride) % extent;`` for each axis.

    Innermost axis (last in ``axes``) has stride 1 and no ``%`` (we trust
    the tid-bounds guard to bound the leading tile so the outer axis can be
    expressed as plain division).
    """
    lines: list[str] = []
    pad = _pad(ctx.indent)
    stride = 1
    # Walk innermost-first to compute strides, but emit outermost-first for
    # readability — the C output reads top-to-bottom matching declaration order.
    decoded: list[str] = []
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = tid % {extent};")
        else:
            decoded.append(f"int {ax.name} = (tid / {stride}) % {extent};")
        stride *= extent
    # The outermost axis is the leading slot; if axes has length 1 the lone
    # axis is `tid` directly (no `%` needed since the bounds guard caps it).
    if len(axes) == 1:
        decoded = [f"int {axes[0].name} = tid;"]
    else:
        # Leading axis: replace its `% extent` with plain division — the
        # bounds guard already caps tid below the full numel.
        outer = axes[0]
        outer_stride = 1
        for ax in axes[1:]:
            outer_stride *= int(ax.extent)
        decoded[-1] = f"int {outer.name} = tid / {outer_stride};"
    for line in reversed(decoded):
        lines.append(pad + line)
    return lines


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


def _render_stmt(stmt: Stmt, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)

    if isinstance(stmt, Let):
        return [f"{pad}float {stmt.name} = {_render_expr(stmt.init, ctx)};"]

    if isinstance(stmt, Store):
        flat = _render_index(stmt.buf, stmt.indices, ctx)
        return [f"{pad}{stmt.buf}[{flat}] = {_render_expr(stmt.value, ctx)};"]

    if isinstance(stmt, AccumFold):
        v = _render_expr(stmt.value, ctx)
        if stmt.op == "max":
            return [f"{pad}{stmt.target} = fmaxf({stmt.target}, {v});"]
        if stmt.op == "min":
            return [f"{pad}{stmt.target} = fminf({stmt.target}, {v});"]
        op = {"add": "+=", "sub": "-=", "mul": "*="}.get(stmt.op, "+=")
        return [f"{pad}{stmt.target} {op} {v};"]

    if isinstance(stmt, Sync):
        return [f"{pad}__syncthreads();"]

    if isinstance(stmt, Cond):
        return _render_cond(stmt, ctx)

    if isinstance(stmt, FreeLoop):
        extent = int(stmt.axis.extent)
        return _render_for(stmt.axis.name, 0, extent, step=None, body=stmt.body, ctx=ctx)

    if isinstance(stmt, Reduce):
        return _render_reduce(stmt, ctx)

    if isinstance(stmt, Tile):
        full_extent = int(stmt.axis.extent)
        return _render_for(stmt.axis.name, 0, full_extent, step=stmt.bk, body=stmt.body, ctx=ctx)

    if isinstance(stmt, Coop):
        return _render_for(stmt.var, "threadIdx.x", stmt.cover, step="blockDim.x", body=stmt.body, ctx=ctx)

    raise TypeError(f"render: unhandled Tile IR stmt {type(stmt).__name__}")


def _render_cond(stmt: Cond, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)
    cond = _render_expr(stmt.cond, ctx)
    inner = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1)
    body = []
    for s in stmt.body:
        body.extend(_render_stmt(s, inner))
    out = [f"{pad}if ({cond}) {{", *body, f"{pad}}}"]
    if stmt.else_body:
        out[-1] = f"{pad}}} else {{"
        for s in stmt.else_body:
            out.extend(_render_stmt(s, inner))
        out.append(f"{pad}}}")
    return out


def _render_reduce(stmt: Reduce, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)
    out: list[str] = []
    for acc in stmt.accs:
        out.append(f"{pad}float {acc.name} = {_render_expr(acc.init, ctx)};")
    extent = stmt.extent if stmt.extent is not None else int(stmt.axis.extent)
    out.extend(_render_for(stmt.axis.name, 0, extent, step=None, body=stmt.body, ctx=ctx))
    return out


def _render_for(var: str, start, end, step, body: tuple, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)
    inner = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1)
    body_lines: list[str] = []
    for s in body:
        body_lines.extend(_render_stmt(s, inner))
    start_s = start if isinstance(start, str) else str(start)
    end_s = end if isinstance(end, str) else str(end)
    if step is None:
        header = f"{pad}for (int {var} = {start_s}; {var} < {end_s}; {var}++) {{"
    else:
        step_s = step if isinstance(step, str) else str(step)
        header = f"{pad}for (int {var} = {start_s}; {var} < {end_s}; {var} += {step_s}) {{"
    return [header, *body_lines, f"{pad}}}"]


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


def _render_expr(expr: Expr, ctx: _Ctx, parent_prec: int = 0) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        if isinstance(expr.value, float) or expr.dtype == "float":
            return f"{float(expr.value):.1f}f"
        v = int(expr.value)
        return f"{v}LL" if abs(v) > 32768 else str(v)
    if isinstance(expr, BinaryExpr):
        prec = _PRECEDENCE.get(expr.op, 10)
        left = _render_expr(expr.left, ctx, prec)
        right = _render_expr(expr.right, ctx, prec + 1)
        result = f"{left} {expr.op} {right}"
        return f"({result})" if prec < parent_prec else result
    if isinstance(expr, Index):
        flat = _render_index(expr.buf, expr.indices, ctx)
        return f"{expr.buf}[{flat}]"
    if isinstance(expr, Builtin):
        return _BUILTIN_TO_CUDA.get(expr.name, expr.name)
    if isinstance(expr, FuncCallExpr):
        args = ", ".join(_render_expr(a, ctx) for a in expr.args)
        return f"{_INTRINSIC_TO_CUDA.get(expr.name, expr.name)}({args})"
    if isinstance(expr, TernaryExpr):
        c = _render_expr(expr.cond, ctx)
        t = _render_expr(expr.if_true, ctx)
        f = _render_expr(expr.if_false, ctx)
        return f"(({c}) ? ({t}) : ({f}))"
    if isinstance(expr, CastExpr):
        inner = _render_expr(expr.expr, ctx)
        return f"(({expr.dtype})({inner}))"
    raise TypeError(f"render_expr: unhandled {type(expr).__name__}")


def _render_index(buf: str, indices: tuple, ctx: _Ctx) -> str:
    """Row-major flatten ``buf[i0][i1]...`` into a single C expression.

    Uses the buffer's shape from ``ctx.shapes`` (populated from
    ``Kernel.params`` + ``Kernel.smem``). For 1D buffers the flatten is a
    single expression; for higher-dim, the result is
    ``((i0 * d1 + i1) * d2 + i2) * ... + iN``.

    Falls back to the index expressions concatenated with ``+`` when the
    buffer has no declared shape (scalar broadcasts where the caller
    passes ``Index("X", (Literal(0),))``).
    """
    if len(indices) == 0:
        return "0"
    if len(indices) == 1:
        return _render_expr(indices[0], ctx)
    shape = ctx.shapes.get(buf)
    if shape is None or len(shape) != len(indices):
        # No shape known — fall back to summing terms (only valid for the
        # 1-element scalar case where every index is Literal(0)).
        return " + ".join(_render_expr(i, ctx) for i in indices)
    # Row-major: outermost index has stride = product of trailing dims.
    flat = _render_expr(indices[0], ctx, _PRECEDENCE["*"])
    for i in range(1, len(indices)):
        stride = 1
        for d in shape[i:]:
            stride *= int(d)
        # `flat` is the running offset; multiply by the next dim's extent
        # before adding the next index.
        next_idx = _render_expr(indices[i], ctx, _PRECEDENCE["+"] + 1)
        # Emit as ((<flat>) * <next_dim> + <next_idx>) by fusing one stride at a time.
        outer_dim = int(shape[i])
        flat = f"{flat} * {outer_dim} + {next_idx}"
    return flat


__all__ = ["render_kernel"]
