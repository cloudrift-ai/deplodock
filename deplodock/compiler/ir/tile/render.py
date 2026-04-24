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

Loop IR's ``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` are
rendered directly: ``Load`` becomes ``float <name> = <buf>[<flat>];``,
``Assign``'s op-name is translated to a C expression via
``_op_to_expr``, ``Accum`` inside a ``Reduce`` becomes the per-iteration
fold (the register declaration is emitted by ``_render_reduce`` from the
distinct accumulator names found in the body).
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
from deplodock.compiler.ir.loop import Accum, Assign, Cond, Load, Select, Write
from deplodock.compiler.ir.tile.ir import (
    Coop,
    Expr,
    FreeLoop,
    Kernel,
    Reduce,
    SmemBuf,
    Stmt,
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

    ``shapes`` maps every buffer name (kernel param + smem) to its declared
    shape, used to flatten multi-dim ``Load`` / ``Write`` indices to
    row-major. ``indent`` tracks the current indent level so each helper
    formats with the correct leading whitespace.
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

    params_text = ", ".join(f"{p.dtype} {p.name}" for p in kernel.params)
    bx, by, bz = kernel.block
    max_threads = bx * by * bz
    launch_bounds = f"\n__launch_bounds__({max_threads})" if max_threads <= 1024 else ""

    body_lines: list[str] = []

    for sb in kernel.smem:
        body_lines.append(_render_smem_decl(sb, ctx))

    for s in kernel.prologue:
        body_lines.extend(_render_stmt(s, ctx))

    if kernel.thread_axes:
        n_threads = 1
        for ax in kernel.thread_axes:
            n_threads *= int(ax.extent)
        body_lines.append(f"{_pad(ctx.indent)}long long tid = blockIdx.x * blockDim.x + threadIdx.x;")
        body_lines.append(f"{_pad(ctx.indent)}if (tid < {n_threads}) {{")
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

    Innermost axis (last in ``axes``) has stride 1 and uses ``tid % extent``.
    The leading (outermost) axis uses ``tid / outer_stride`` (no ``%``) since
    the bounds guard already caps tid below the full numel.
    """
    pad = _pad(ctx.indent)
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = tid % {extent};")
        else:
            decoded.append(f"int {ax.name} = (tid / {stride}) % {extent};")
        stride *= extent
    if len(axes) == 1:
        decoded = [f"int {axes[0].name} = tid;"]
    else:
        outer = axes[0]
        outer_stride = 1
        for ax in axes[1:]:
            outer_stride *= int(ax.extent)
        decoded[-1] = f"int {outer.name} = tid / {outer_stride};"
    return [pad + line for line in reversed(decoded)]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


def _render_stmt(stmt: Stmt, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)

    if isinstance(stmt, Load):
        flat = _render_index(stmt.input, stmt.index, ctx)
        return [f"{pad}float {stmt.name} = {stmt.input}[{flat}];"]

    if isinstance(stmt, Assign):
        args = [Var(a) for a in stmt.args]
        return [f"{pad}float {stmt.name} = {_render_expr(_op_to_expr(stmt.op.name, args), ctx)};"]

    if isinstance(stmt, Select):
        return [f"{pad}float {stmt.name} = {_render_expr(_select_to_ternary(stmt), ctx)};"]

    if isinstance(stmt, Write):
        flat = _render_index(stmt.output, stmt.index, ctx)
        return [f"{pad}{stmt.output}[{flat}] = {stmt.value};"]

    if isinstance(stmt, Accum):
        op_name = stmt.op.name
        if op_name in ("maximum", "amax"):
            return [f"{pad}{stmt.name} = fmaxf({stmt.name}, {stmt.value});"]
        if op_name == "minimum":
            return [f"{pad}{stmt.name} = fminf({stmt.name}, {stmt.value});"]
        op = {"add": "+=", "sum": "+=", "multiply": "*=", "prod": "*="}.get(op_name, "+=")
        return [f"{pad}{stmt.name} {op} {stmt.value};"]

    if isinstance(stmt, Sync):
        return [f"{pad}__syncthreads();"]

    if isinstance(stmt, Cond):
        return _render_cond(stmt, ctx)

    if isinstance(stmt, FreeLoop):
        return _render_for(stmt.axis.name, 0, int(stmt.axis.extent), step=None, body=stmt.body, ctx=ctx)

    if isinstance(stmt, Reduce):
        return _render_reduce(stmt, ctx)

    if isinstance(stmt, Tile):
        return _render_for(stmt.axis.name, 0, int(stmt.axis.extent), step=stmt.bk, body=stmt.body, ctx=ctx)

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
    """Declare each accumulator (one per distinct ``Accum.name`` in the body)
    before the for-loop, then emit the body as the loop interior. The body
    contains plain ``Accum`` stmts whose render emits the per-iteration fold.
    """
    pad = _pad(ctx.indent)
    out: list[str] = []
    seen: set[str] = set()
    for s in _walk(stmt.body):
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            out.append(f"{pad}float {s.name} = {float(identity):.1f}f;")
    extent = stmt.extent if stmt.extent is not None else int(stmt.axis.extent)
    out.extend(_render_for(stmt.axis.name, 0, extent, step=None, body=stmt.body, ctx=ctx))
    return out


def _walk(stmts: tuple[Stmt, ...]):
    """Pre-order walk over a Tile-IR body for accumulator collection."""
    for s in stmts:
        yield s
        sub = getattr(s, "body", None)
        if isinstance(sub, tuple):
            yield from _walk(sub)


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
    """Row-major flatten ``buf[i0][i1]...``."""
    if len(indices) == 0:
        return "0"
    if len(indices) == 1:
        return _render_expr(indices[0], ctx)
    shape = ctx.shapes.get(buf)
    if shape is None or len(shape) != len(indices):
        return " + ".join(_render_expr(i, ctx) for i in indices)
    flat = _render_expr(indices[0], ctx, _PRECEDENCE["*"])
    for i in range(1, len(indices)):
        next_idx = _render_expr(indices[i], ctx, _PRECEDENCE["+"] + 1)
        outer_dim = int(shape[i])
        flat = f"{flat} * {outer_dim} + {next_idx}"
    return flat


# ---------------------------------------------------------------------------
# Op-name → Expr translation (mirrors _common.apply_elementwise)
# ---------------------------------------------------------------------------


_BINARY_OP: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "mod": "%",
}

_SUPPORTED_UNARY_INTRINSIC: dict[str, str] = {
    "exp": "exp",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "abs": "fabs",
}


def _op_to_expr(fn: str, inputs: list[Expr]) -> Expr:
    if fn in _BINARY_OP:
        return BinaryExpr(_BINARY_OP[fn], inputs[0], inputs[1])
    if fn == "maximum":
        return FuncCallExpr("fmax", list(inputs))
    if fn == "minimum":
        return FuncCallExpr("fmin", list(inputs))
    if fn == "pow":
        return FuncCallExpr("pow", list(inputs))
    if fn == "negative":
        return BinaryExpr("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "reciprocal":
        return BinaryExpr("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCallExpr("fmax", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        neg_x = BinaryExpr("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCallExpr("exp", [neg_x])
        return BinaryExpr("/", Literal(1.0, "float"), BinaryExpr("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY_INTRINSIC:
        return FuncCallExpr(_SUPPORTED_UNARY_INTRINSIC[fn], list(inputs))
    raise NotImplementedError(f"render: elementwise fn={fn!r} not yet supported")


def _select_to_ternary(s: Select) -> Expr:
    """Build a chained ternary from a Loop IR ``Select``."""
    branches = list(s.branches)
    result: Expr = Var(branches[-1].value)
    for b in reversed(branches[:-1]):
        result = TernaryExpr(cond=b.select, if_true=Var(b.value), if_false=result)
    return result


__all__ = ["render_kernel"]
