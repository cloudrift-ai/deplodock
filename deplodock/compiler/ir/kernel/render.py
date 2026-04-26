"""Kernel IR → CUDA source.

Renders a ``KernelOp`` to a complete ``extern "C" __global__`` CUDA
function text. The renderer is purely mechanical: every Kernel-IR node
has a fixed emission rule. Kernel IR is the fully-scheduled form — no
strategy / decision logic reaches here.

Loop IR's ``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` are
rendered directly: ``Load`` becomes ``float <name> = <buf>[<flat>];``,
``Assign``'s op-name is translated to a C expression via
``_op_to_expr``, ``Accum`` inside a reduce ``Loop`` (a Loop whose body
contains any ``Accum``) becomes the per-iteration fold; the register
declaration is emitted by ``_render_loop`` from the distinct accumulator
names found in the body.
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
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    Expr,
    KernelOp,
    Smem,
    Stmt,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Init, Load, Loop, Select, Write

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
    # Accumulator names whose init has been emitted by an ``Init`` Stmt at
    # an enclosing scope. ``_render_loop`` uses this to suppress its
    # default Loop-immediate Accum init for those names — preventing
    # nested-reduce reset bugs (e.g. matmul ``Loop(k_o) > Loop(k_i) >
    # Accum`` resetting per-k_o iteration).
    explicit_inits: set[str] = field(default_factory=set)


def _pad(indent: int) -> str:
    return "    " * indent


# ---------------------------------------------------------------------------
# Top-level: render_kernelop
# ---------------------------------------------------------------------------


# Default block size for thread-axes flattening when no strategy has chosen one
# (i.e. ``block_axes`` is empty). When ``block_axes`` is populated, the block
# size comes from ``prod(thread_axes.extents)``.
_BLOCK_SIZE = 256


def render_kernelop(kernel_op: KernelOp, shapes: dict[str, tuple[int, ...]] | None = None) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function for a ``KernelOp``.

    ``shapes`` maps each global-buffer name (anything appearing on a
    ``Load.input`` or ``Write.output``) to its declared shape; the
    renderer uses it to row-major-flatten multi-dim indices. Production
    callers typically build ``shapes`` from the surrounding graph
    (``{nid: graph.nodes[nid].output.shape for nid in ...}``); tests pass
    it as a literal dict.

    Kernel signature is derived from the body: ``kernel_op.inputs`` (distinct
    ``Load.input`` names) become ``const float*`` params, ``kernel_op.outputs``
    (distinct ``Write.output`` names) become ``float*`` params, ordered
    by first appearance.
    """
    ctx = _Ctx(shapes=dict(shapes or {}), indent=1)

    sig_parts = [f"const float* {n}" for n in kernel_op.inputs]
    sig_parts.extend(f"float* {n}" for n in kernel_op.outputs)
    params_text = ", ".join(sig_parts)
    bounds = _launch_bounds_for(kernel_op)
    launch_bounds = f"\n__launch_bounds__({bounds})"

    body_lines: list[str] = []
    for s in kernel_op.body:
        body_lines.extend(_render_stmt(s, ctx))

    body_text = "\n".join(body_lines)
    return f'extern "C" __global__{launch_bounds} void {kernel_op.name}({params_text}) {{\n{body_text}\n}}\n'


def _launch_bounds_for(kernel_op: KernelOp) -> int:
    """Derive ``__launch_bounds__`` from the first ``Enclosure``'s thread axes
    when ``block_axes`` is populated; otherwise fall back to the legacy
    ``_BLOCK_SIZE`` cap (which the host-side launcher rounds up the grid for)."""
    for s in kernel_op.body:
        if isinstance(s, Enclosure):
            if s.block_axes:
                bsize = 1
                for ax in s.thread_axes:
                    bsize *= int(ax.extent)
                return max(bsize, 1)
            return _BLOCK_SIZE
    return _BLOCK_SIZE


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

    if isinstance(stmt, Init):
        identity = stmt.op.identity
        if identity is None:
            raise ValueError(f"Init {stmt.name!r} op {stmt.op.name!r} has no identity")
        ctx.explicit_inits.add(stmt.name)
        return [f"{pad}float {stmt.name} = {float(identity):.1f}f;"]

    if isinstance(stmt, Cond):
        return _render_cond(stmt, ctx)

    if isinstance(stmt, Loop):
        return _render_loop(stmt, ctx)

    if isinstance(stmt, Enclosure):
        return _render_enclosure(stmt, ctx)

    if isinstance(stmt, Smem):
        return _render_smem(stmt, ctx)

    if isinstance(stmt, Sync):
        return [f"{pad}__syncthreads();"]

    if isinstance(stmt, TreeHalve):
        return _render_tree_halve(stmt, ctx)

    if isinstance(stmt, StridedLoop):
        return _render_strided_loop(stmt, ctx)

    raise TypeError(f"render: unhandled Kernel IR stmt {type(stmt).__name__}")


def _render_smem(stmt: Smem, ctx: _Ctx) -> list[str]:
    """``__shared__ <dtype> <name>[<prod(extents)>];`` and register the
    buffer's shape so subsequent ``Load``/``Write`` flatten correctly."""
    pad = _pad(ctx.indent)
    total = 1
    for e in stmt.extents:
        total *= int(e)
    ctx.shapes[stmt.name] = tuple(int(e) for e in stmt.extents)
    return [f"{pad}__shared__ {stmt.dtype} {stmt.name}[{total}];"]


def _render_tree_halve(stmt: TreeHalve, ctx: _Ctx) -> list[str]:
    """Power-of-two tree reduction over ``buf[0..length)`` into ``buf[0]``."""
    pad = _pad(ctx.indent)
    inner_pad = _pad(ctx.indent + 1)
    halve_pad = _pad(ctx.indent + 2)
    op_expr = _binary_combine_expr(stmt.op, f"{stmt.buf}[{stmt.tid_var}]", f"{stmt.buf}[{stmt.tid_var} + s]")
    half = int(stmt.length) // 2
    return [
        f"{pad}for (int s = {half}; s > 0; s >>= 1) {{",
        f"{inner_pad}if ({stmt.tid_var} < s) {{",
        f"{halve_pad}{stmt.buf}[{stmt.tid_var}] = {op_expr};",
        f"{inner_pad}}}",
        f"{inner_pad}__syncthreads();",
        f"{pad}}}",
    ]


def _binary_combine_expr(op: object, a: str, b: str) -> str:
    """Render a 2-arg combine for ``ElementwiseImpl`` reduce ops."""
    name = getattr(op, "name", "add")
    if name in ("add", "sum"):
        return f"{a} + {b}"
    if name in ("multiply", "prod"):
        return f"{a} * {b}"
    if name in ("maximum", "amax"):
        return f"fmaxf({a}, {b})"
    if name == "minimum":
        return f"fminf({a}, {b})"
    raise ValueError(f"TreeHalve: unsupported op {name!r}")


def _render_strided_loop(stmt: StridedLoop, ctx: _Ctx) -> list[str]:
    """``for (int <axis> = <start>; <axis> < <extent>; <axis> += <step>)``.

    Reduce-loop detection (any ``Accum`` in immediate body) mirrors
    ``_render_loop`` — accumulators are declared with their identity above
    the loop, then folded inside.
    """
    pad = _pad(ctx.indent)
    out: list[str] = []
    seen: set[str] = set()
    for s in stmt.body:
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            if s.name in ctx.explicit_inits:
                continue
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            out.append(f"{pad}float {s.name} = {float(identity):.1f}f;")
    start_str = _render_expr(stmt.start, ctx)
    step = stmt.step if isinstance(stmt.step, int) else _render_expr(stmt.step, ctx)
    out.extend(_render_for(stmt.axis.name, start_str, int(stmt.axis.extent), step=step, body=stmt.body, ctx=ctx))
    return out


def _render_enclosure(stmt: Enclosure, ctx: _Ctx) -> list[str]:
    """Emit thread / block index decodes plus body.

    Two forms:

    - **Legacy (``block_axes`` empty):** flatten all ``thread_axes`` into
      one linear ``tid = blockIdx.x * blockDim.x + threadIdx.x`` index,
      bounds-guarded against the product of extents. Used by today's
      one-thread-per-output-slot kernels (pointwise, per-thread-serial
      reductions).

    - **Cooperative (``block_axes`` populated):** one CUDA block per
      ``block_axes`` slot, ``thread_axes`` index threads inside the block.
      Decodes ``blockIdx.x`` into the block-axis Vars and ``threadIdx.x``
      into the thread-axis Vars. No tid bounds guard — the strategy is
      responsible for picking extents that match the launch geometry.
    """
    pad = _pad(ctx.indent)
    inner = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1, explicit_inits=ctx.explicit_inits)

    if stmt.block_axes:
        out = [f"{pad}{{"]
        out.extend(_render_grid_axis_decode(stmt.block_axes, "blockIdx.x", inner))
        out.extend(_render_grid_axis_decode(stmt.thread_axes, "threadIdx.x", inner))
        for s in stmt.body:
            out.extend(_render_stmt(s, inner))
        out.append(f"{pad}}}")
        return out

    n_threads = 1
    for ax in stmt.thread_axes:
        n_threads *= int(ax.extent)
    out = [
        f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
        f"{pad}if (tid < {n_threads}) {{",
    ]
    out.extend(_render_thread_axis_decode(stmt.thread_axes, inner))
    for s in stmt.body:
        out.extend(_render_stmt(s, inner))
    out.append(f"{pad}}}")
    return out


def _render_grid_axis_decode(axes: tuple, idx_expr: str, ctx: _Ctx) -> list[str]:
    """Decode ``idx_expr`` (``blockIdx.x`` or ``threadIdx.x``) into per-axis ints.

    Single-axis: ``int <ax> = <idx_expr>;``. Multi-axis: row-major flatten
    using the same shape rule as ``_render_thread_axis_decode``.
    """
    pad = _pad(ctx.indent)
    if not axes:
        return []
    if len(axes) == 1:
        return [f"{pad}int {axes[0].name} = {idx_expr};"]
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = {idx_expr} % {extent};")
        else:
            decoded.append(f"int {ax.name} = ({idx_expr} / {stride}) % {extent};")
        stride *= extent
    outer = axes[0]
    outer_stride = 1
    for ax in axes[1:]:
        outer_stride *= int(ax.extent)
    decoded[-1] = f"int {outer.name} = {idx_expr} / {outer_stride};"
    return [pad + line for line in reversed(decoded)]


def _render_cond(stmt: Cond, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)
    cond = _render_expr(stmt.cond, ctx)
    inner = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1, explicit_inits=ctx.explicit_inits)
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


def _render_loop(stmt: Loop, ctx: _Ctx) -> list[str]:
    """Render a ``Loop``. If its immediate body contains any ``Accum``, treat
    it as a reduce-Loop: declare each distinct accumulator (initialized to
    the op's identity) before the for-loop, then emit the body as the loop
    interior. A free Loop (no Accum in immediate body) renders as a plain
    for-loop.

    Accums whose name appears in ``ctx.explicit_inits`` skip the default
    init — an enclosing ``Init`` Stmt has already declared them. This
    handles nested-reduce shapes (chunked-K matmul) where the
    accumulator must persist across the outer loop's iterations.
    """
    pad = _pad(ctx.indent)
    out: list[str] = []
    seen: set[str] = set()
    for s in stmt.body:
        if isinstance(s, Accum) and s.name not in seen:
            seen.add(s.name)
            if s.name in ctx.explicit_inits:
                continue
            identity = s.op.identity
            if identity is None:
                raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
            out.append(f"{pad}float {s.name} = {float(identity):.1f}f;")
    out.extend(_render_for(stmt.axis.name, 0, int(stmt.axis.extent), step=None, body=stmt.body, ctx=ctx))
    return out


def _render_for(var: str, start, end, step, body: tuple, ctx: _Ctx) -> list[str]:
    pad = _pad(ctx.indent)
    inner = _Ctx(shapes=ctx.shapes, indent=ctx.indent + 1, explicit_inits=ctx.explicit_inits)
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
    # Row-major strides: stride[d] = prod(shape[d+1:]).
    parts: list[str] = []
    for d, idx in enumerate(indices):
        stride = 1
        for k in range(d + 1, len(shape)):
            stride *= int(shape[k])
        idx_str = _render_expr(idx, ctx, _PRECEDENCE["*"])
        parts.append(idx_str if stride == 1 else f"{idx_str} * {stride}")
    return " + ".join(parts)


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


__all__ = ["render_kernelop"]
