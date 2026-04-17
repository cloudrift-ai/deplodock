"""Naive recursive-descent CUDA codegen: ``LoopProgram`` → ``GpuProgram``.

Consumes a ``LoopProgram`` (one ``LoopLaunch`` per GPU kernel, with
authoritative buffer shapes) and emits CUDA C source packaged as a
``GpuProgram``. The ``LoopOp`` body is an SSA program over named axes:
``Assign``/``Update``/``Write``/``Select`` statements interpreted in
order. The positionally-last ``Update`` marks the end of the reduce
sweep; statements after (and final ``Write``s) run once per free point.
"""

from __future__ import annotations

import logging
import math

from deplodock.compiler.backend.cuda.program import CudaLaunch
from deplodock.compiler.ir.expr import BinOp, Expr, FuncCall, Literal, Ternary, Var, substitute
from deplodock.compiler.ir.gpu import (
    ArrayAccess,
    AugAssign,
    ForLoop,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    Stmt,
    VarDecl,
)
from deplodock.compiler.ir.gpu import (
    Assign as IrAssign,
)
from deplodock.compiler.ir.loop import Assign as IrAssignStmt
from deplodock.compiler.ir.loop import Axis, LoopOp, Port, Select, Update
from deplodock.compiler.program.gpu import GpuBuffer, GpuProgram
from deplodock.compiler.program.loop import LoopLaunch, LoopProgram

logger = logging.getLogger(__name__)

_BLOCK = 256


# ---------------------------------------------------------------------------
# Program-level entry
# ---------------------------------------------------------------------------


def compile_kernels(program: LoopProgram) -> GpuProgram:
    """Lower a ``LoopProgram`` to a ``GpuProgram``."""
    referenced: set[str] = set()
    for launch in program.launches:
        referenced.update(launch.input_names)
        referenced.add(launch.output_name)
    buf_names = {b.name for b in program.buffers}
    referenced |= set(program.graph_constants) & buf_names

    buffers = [GpuBuffer(name=b.name, shape=tuple(b.shape), dtype="float", role=b.role) for b in program.buffers if b.name in referenced]

    launches: list[CudaLaunch] = []
    for i, launch in enumerate(program.launches):
        if not isinstance(launch.loop, LoopOp):
            raise TypeError(
                f"CudaBackend: launch {i} has non-LoopOp "
                f"{type(launch.loop).__name__!r}; fusion must wrap every primitive "
                f"into a LoopOp before CUDA codegen."
            )
        for w in check_port_bounds(launch, program, launch_idx=i):
            logger.warning("OOB port access: %s", w)
        kname = _kernel_name(launch.loop, i)
        gpu_kernel, arg_order = emit_kernel(launch, kname, program)
        source = _emit_kernel_source(gpu_kernel)
        grid, block = _launch_config(launch, program)
        launches.append(
            CudaLaunch(
                kernel_source=source,
                kernel_name=kname,
                grid=grid,
                block=block,
                args=arg_order,
                comment=program.pretty_print_launch(i),
            )
        )

    return GpuProgram(
        name=program.name,
        buffers=buffers,
        launches=launches,
        constant_values=dict(program.constant_values),
        comment=program.pretty_print(),
    )


def check_port_bounds(launch: LoopLaunch, program: LoopProgram, launch_idx: int = -1) -> list[str]:
    """Symbolically check whether any Port.index can go out of bounds.

    For each per-dim index Expr, compute the maximum value attained over
    all combinations of axis extents and compare against the source
    buffer's dim extent. Ports that are only referenced inside ``Select``
    branches have their bounds evaluated *under* the branch's predicate —
    the rotary ``rotate_half`` pattern (two halves each reading a slice of
    a D-wide tensor and disjoint on ``axis < D/2``/``>=D/2``) is correctly
    recognized as in-bounds because the predicate constrains the axis.

    Returns a list of human-readable OOB warnings (empty when the kernel
    is in-bounds). CUDA codegen clamps port reads via
    ``min(max(coord, 0), extent-1)`` regardless, so OOB findings are
    informational — they flag kernels where some predicate is expected to
    mask the stray load but the checker couldn't prove it.
    """
    from deplodock.compiler.ir.loop import Select

    warnings: list[str] = []
    if not isinstance(launch.loop, LoopOp):
        return warnings
    loop = launch.loop
    axes_extents = {a.name: int(a.extent) - 1 for a in loop.axes}
    buf_name_set = {b.name for b in program.buffers}

    # Collect Select predicates gating each port (by SSA name ``$N``).
    select_preds: dict[str, list[Expr]] = {}
    for stmt in loop.body:
        if isinstance(stmt, Select):
            for branch in stmt.branches:
                select_preds.setdefault(branch.value, []).append(branch.select)

    for pi, port in enumerate(loop.inputs):
        buf_name = launch.input_names[pi] if pi < len(launch.input_names) else None
        if buf_name is None or buf_name not in buf_name_set:
            continue
        src_shape = tuple(int(d) for d in program.shape(buf_name))
        port_key = f"${pi}"
        preds = select_preds.get(port_key, [])
        # Per-port axis bounds, tightened by each predicate (conservatively:
        # if any predicate proves tighter bounds for this port, use them).
        tight_axes = _tighten_axes_from_preds(axes_extents, preds) if preds else axes_extents

        for d, idx_expr in enumerate(port.index):
            if d >= len(src_shape):
                warnings.append(
                    f"launch {launch_idx} ({launch.output_name}): port ${pi} ({buf_name} shape={src_shape}) "
                    f"has index rank {len(port.index)} > buf rank {len(src_shape)}"
                )
                continue
            max_coord = _max_index_value(idx_expr, tight_axes)
            if max_coord is None:
                continue
            if max_coord >= src_shape[d]:
                warnings.append(
                    f"launch {launch_idx} ({launch.output_name}): port ${pi} ({buf_name} shape={src_shape}) "
                    f"reads out of bounds at dim {d}: max_index={max_coord}, dim_extent={src_shape[d]}, "
                    f"expr={idx_expr}"
                )
    return warnings


def _tighten_axes_from_preds(axes_extents: dict[str, int], preds: list[Expr]) -> dict[str, int]:
    """Return per-axis max values tightened under a set of Select predicates.

    The checker handles a small pattern library: ``Var(a) < K``,
    ``Var(a) >= K``, and ``K > Var(a)`` / ``K <= Var(a)`` — enough to
    cover the ``rotate_half`` predicate ``axis < D/2``. Unsupported
    predicates leave the bounds untouched (conservative: warning may
    fire on a stray read that's actually safe).
    """
    # Start with the loose bounds; each predicate may reduce an axis max.
    tight = dict(axes_extents)
    for pred in preds:
        for axis_name, new_max in _axis_constraints_from_pred(pred).items():
            if axis_name in tight:
                tight[axis_name] = min(tight[axis_name], new_max)
    return tight


def _axis_constraints_from_pred(pred: Expr) -> dict[str, int]:
    """Extract ``{axis_name: max_value}`` constraints from a predicate.

    Recognizes ``Var(a) < K``, ``K > Var(a)``, ``Var(a) <= K``,
    ``K >= Var(a)`` and their AND-conjunctions (``BinOp("&&", ...)``).
    Returns an empty dict for unrecognized predicates (no tightening).
    """
    if isinstance(pred, BinOp):
        if pred.op in ("&&", "and"):
            left = _axis_constraints_from_pred(pred.left)
            right = _axis_constraints_from_pred(pred.right)
            merged = dict(left)
            for k, v in right.items():
                merged[k] = min(merged.get(k, v), v)
            return merged
        # axis < K  →  axis ≤ K-1
        if pred.op == "<":
            axis, k = _match_var_const(pred.left, pred.right)
            if axis is not None:
                return {axis: k - 1}
            axis, k = _match_var_const(pred.right, pred.left)
            if axis is not None:  # K > axis  →  axis ≤ K-1
                return {axis: k - 1}
        if pred.op == "<=":
            axis, k = _match_var_const(pred.left, pred.right)
            if axis is not None:
                return {axis: k}
            axis, k = _match_var_const(pred.right, pred.left)
            if axis is not None:  # K >= axis
                return {axis: k}
    return {}


def _match_var_const(a: Expr, b: Expr) -> tuple[str | None, int]:
    """If ``a`` is a Var and ``b`` is an int Literal, return (a.name, b.value)."""
    if isinstance(a, Var) and isinstance(b, Literal):
        try:
            return a.name, int(b.value)
        except (TypeError, ValueError):
            return None, 0
    return None, 0


def _max_index_value(expr: Expr, axis_max: dict[str, int]) -> int | None:
    """Maximum non-negative value an Expr can attain given per-axis maxes.

    Conservatively bounds: Literals → value, Vars → axis_max, BinOps →
    recursive max for +/*/%//. Returns ``None`` for unsupported shapes.
    """
    if isinstance(expr, Literal):
        try:
            return int(expr.value)
        except (TypeError, ValueError):
            return None
    if isinstance(expr, Var):
        if expr.name in axis_max:
            return axis_max[expr.name]
        return None
    if isinstance(expr, BinOp):
        left = _max_index_value(expr.left, axis_max)
        right = _max_index_value(expr.right, axis_max)
        if left is None or right is None:
            return None
        if expr.op == "+":
            return left + right
        if expr.op == "*":
            return left * right
        if expr.op == "%":
            right_val = right
            if right_val <= 0:
                return None
            return right_val - 1
        if expr.op == "/":
            if right <= 0:
                return None
            return left // right
        if expr.op == "-":
            # Conservative upper bound: for `a - b`, max is left - min(b).
            # We only know min(b) statically when b is a Literal (then
            # min == max == value). Otherwise fall back to ``left``.
            if isinstance(expr.right, Literal):
                try:
                    return left - int(expr.right.value)
                except (TypeError, ValueError):
                    return left
            return left
        return None
    return None


# ---------------------------------------------------------------------------
# Per-kernel emission
# ---------------------------------------------------------------------------


def emit_kernel(launch: LoopLaunch, kernel_name: str, program: LoopProgram) -> tuple[GpuKernel, list[str]]:
    """Emit one ``GpuKernel`` for a single ``LoopLaunch``."""
    dollar_shapes = program.dollar_shapes(launch)
    out_shape = program.output_shape(launch)
    params, arg_order = _build_params(launch)
    body, block_size = _emit_body(launch, out_shape, dollar_shapes, program)
    kd = GpuKernel(name=kernel_name, params=params, body=body, block_size=block_size)
    return kd, arg_order


# ---------------------------------------------------------------------------
# Axis-env helpers
# ---------------------------------------------------------------------------


def _axis_env_for_flat(axes: tuple[Axis, ...], flat_idx: Expr) -> dict[str, Expr]:
    """Decompose a flat iteration index into per-axis Exprs."""
    env: dict[str, Expr] = {}
    if not axes:
        return env
    remainder = flat_idx
    extents = [int(a.extent) for a in axes]
    for i in range(len(axes) - 1, -1, -1):
        dim = extents[i]
        if i == 0:
            env[axes[i].name] = remainder
        else:
            env[axes[i].name] = BinOp("%", remainder, Literal(dim, "int"))
            remainder = BinOp("/", remainder, Literal(dim, "int"))
    return env


def _axis_env_for_reduce(axes: tuple[Axis, ...], row_idx: Expr, k: Expr) -> dict[str, Expr]:
    """Axis env for a reduce kernel: row_idx unpacks free axes, k is the reduce axis."""
    env: dict[str, Expr] = {}
    free = [a for a in axes if a.kind == "free"]
    reduce_a = next((a for a in axes if a.kind == "reduce"), None)

    remainder = row_idx
    extents = [int(a.extent) for a in free]
    for i in range(len(free) - 1, -1, -1):
        dim = extents[i]
        if i == 0:
            env[free[i].name] = remainder
        else:
            env[free[i].name] = BinOp("%", remainder, Literal(dim, "int"))
            remainder = BinOp("/", remainder, Literal(dim, "int"))

    if reduce_a is not None:
        env[reduce_a.name] = k
    return env


def _emit_port_load(port: Port, buf_name: str, src_shape: tuple, axis_env: dict[str, Expr]) -> Expr:
    """Evaluate ``port.index`` under ``axis_env`` and emit an ArrayAccess.

    Each per-dim coord is clamped to ``[0, dim_extent-1]`` via
    ``max(0, min(coord, extent-1))`` so that out-of-bounds reads from
    Select-masked branches stay in the allocated buffer (matches
    LoopBackend's ``np.clip`` behavior). Const coords and coords that are
    already bounded (a single Var matching the dim's extent) skip the
    clamp to keep the emitted code tight.
    """
    if not port.index:
        return ArrayAccess(array=buf_name, index=Literal(0, "int"))
    coords = [substitute(e, axis_env) for e in port.index]
    coords = [_clamp_coord(c, src_shape[d] if d < len(src_shape) else None) for d, c in enumerate(coords)]
    flat = _flatten_coords(coords, src_shape)
    return ArrayAccess(array=buf_name, index=flat)


def _clamp_coord(coord: Expr, dim_extent) -> Expr:
    """Wrap ``coord`` in a clamp to ``[0, dim_extent-1]`` unless provably in-bounds."""
    if dim_extent is None or not isinstance(dim_extent, int) or dim_extent <= 0:
        return coord
    if isinstance(coord, Literal):
        try:
            v = int(coord.value)
            if 0 <= v < dim_extent:
                return coord
        except (TypeError, ValueError):
            return coord
    if dim_extent == 1:
        return Literal(0, "int")
    upper = Literal(dim_extent - 1, "int")
    # (coord > upper) ? upper : ((coord < 0) ? 0 : coord)
    clamped_high = Ternary(cond=BinOp(">", coord, upper), if_true=upper, if_false=coord)
    return Ternary(cond=BinOp("<", clamped_high, Literal(0, "int")), if_true=Literal(0, "int"), if_false=clamped_high)


def _flatten_coords(coords: list[Expr], shape: tuple) -> Expr:
    """Combine per-dim coord Exprs into a flat row-major index."""
    if not coords:
        return Literal(0, "int")
    flat: Expr = Literal(0, "int")
    stride = 1
    dims = [int(d) if isinstance(d, int) else 1 for d in shape]
    for d in range(len(coords) - 1, -1, -1):
        coord = coords[d]
        term = coord if stride == 1 else BinOp("*", coord, Literal(stride, "int"))
        if isinstance(flat, Literal) and flat.value == 0:
            flat = term
        else:
            flat = BinOp("+", term, flat)
        if d > 0 and d < len(dims):
            stride *= dims[d]
    return flat


# ---------------------------------------------------------------------------
# Body dispatch
# ---------------------------------------------------------------------------


def _emit_body(
    launch: LoopLaunch, out_shape: tuple, dollar_shapes: dict[str, tuple], program: LoopProgram
) -> tuple[list[Stmt], tuple[int, int, int]]:
    from deplodock.compiler.ir.loop_plan import analyze_kernel

    plan = analyze_kernel(launch.loop, dollar_shapes, out_shape)
    idx = Var("tid")
    idx_init = VarDecl(
        dtype="long long",
        name="tid",
        init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
    )
    guarded = _emit_plan(plan, launch, dollar_shapes, program, idx)
    stmts: list[Stmt] = [idx_init, IfStmt(cond=BinOp("<", idx, Literal(plan.n_output, "int")), body=guarded)]
    return stmts, (_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Plan-based emitter
# ---------------------------------------------------------------------------


def _emit_plan(plan, launch: LoopLaunch, dollar_shapes: dict[str, tuple], program: LoopProgram, idx: Expr) -> list[Stmt]:
    from deplodock.compiler.ir.loop_plan import Inline, Loop

    loop: LoopOp = launch.loop
    stmts: list[Stmt] = []
    name_seq = [0]
    values: dict[str, Expr] = {}

    port_info = _collect_port_info(loop, launch, program, dollar_shapes)

    free_axes = tuple(a for a in loop.axes if a.kind == "free")
    flat_env = _axis_env_for_flat(free_axes, idx)

    # Load per-row ports upfront. Threading ``load_env`` forward lets a later
    # port's index Expr reference an earlier port by name (e.g. gather: the
    # data port's axis index reads ``$0`` — the already-loaded index value).
    load_env: dict[str, Expr] = dict(flat_env)
    for key, (port, buf_name, src_shape) in port_info.items():
        if key not in plan.per_elem_ports:
            values[key] = _emit_port_load(port, buf_name, src_shape, load_env)
            load_env[key] = values[key]

    loop_count = 0
    for step in plan.steps:
        if isinstance(step, Inline):
            for stmt in step.body:
                if isinstance(stmt, IrAssignStmt):
                    arg_exprs = [values[a] for a in stmt.args]
                    value = _apply_elementwise(stmt.op.fn, arg_exprs)
                    tname = _fresh(name_seq)
                    stmts.append(VarDecl(dtype="float", name=tname, init=value))
                    values[stmt.name] = Var(tname)
                elif isinstance(stmt, Select):
                    value = _emit_select(stmt, values, flat_env)
                    tname = _fresh(name_seq)
                    stmts.append(VarDecl(dtype="float", name=tname, init=value))
                    values[stmt.name] = Var(tname)

        elif isinstance(step, Loop):
            k_var_name = f"k{loop_count}"
            k_var = Var(k_var_name)

            if step.accum:
                stmts.append(VarDecl(dtype="float", name=step.accum.var, init=Literal(step.accum.identity, "float")))

            inner: list[Stmt] = []
            loop_values: dict[str, Expr] = dict(values)

            axis_env = _axis_env_for_reduce(loop.axes, idx, k_var)

            for key, (port, buf_name, src_shape) in port_info.items():
                if key in plan.per_elem_ports:
                    loop_values[key] = _emit_port_load(port, buf_name, src_shape, axis_env)

            for assign in step.recompute:
                arg_exprs = [loop_values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[assign.name] = Var(tname)

            for assign in step.body:
                arg_exprs = [loop_values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[assign.name] = Var(tname)

            if step.accum:
                src = loop_values[step.accum.src]
                inner.append(_emit_reduce_accum(step.accum.var, step.accum.fn, src))
                values[step.accum.result] = Var(step.accum.var)

            if step.stores_output:
                # Per-element store inside the loop.
                store_value = loop_values[step.store_value]
                store_idx_coords = [substitute(e, axis_env) for e in step.store_index]
                buf_shape = program.shape(launch.output_name)
                store_idx = _flatten_coords(store_idx_coords, buf_shape)
                inner.append(IrAssign(target=ArrayAccess(array=launch.output_name, index=store_idx), value=store_value))

            stmts.append(ForLoop(var=k_var_name, start=Literal(0, "int"), end=Literal(step.k_size, "int"), body=inner))
            loop_count += 1

    # Trailing writes: run once per free point, post-reduce.
    for tw in plan.trailing_writes:
        value = values.get(tw.value, Literal(0.0, "float"))
        coords = [substitute(e, flat_env) for e in tw.index]
        buf_shape = program.shape(launch.output_name)
        store_idx = _flatten_coords(coords, buf_shape)
        stmts.append(IrAssign(target=ArrayAccess(array=launch.output_name, index=store_idx), value=value))

    return stmts


def _collect_port_info(
    loop: LoopOp,
    launch: LoopLaunch,
    program: LoopProgram,
    dollar_shapes: dict[str, tuple],
) -> dict[str, tuple[Port, str, tuple]]:
    """Build the $N → (port, buffer_name, source_shape) mapping for codegen."""
    port_info: dict[str, tuple[Port, str, tuple]] = {}
    buf_name_set = {b.name for b in program.buffers}
    for i, port in enumerate(loop.inputs):
        key = f"${i}"
        buf_name = launch.input_names[i] if i < len(launch.input_names) else key
        src_shape = program.shape(buf_name) if buf_name in buf_name_set else dollar_shapes.get(key, ())
        port_info[key] = (port, buf_name, src_shape)
    return port_info


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    if fn == "max":
        from deplodock.compiler.ir.gpu import VarAssign

        return VarAssign(name=acc_name, value=FuncCall("fmaxf", [Var(acc_name), value]))
    if fn == "min":
        from deplodock.compiler.ir.gpu import VarAssign

        return VarAssign(name=acc_name, value=FuncCall("fminf", [Var(acc_name), value]))
    op = {"add": "+=", "sum": "+=", "mul": "*=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


# ---------------------------------------------------------------------------
# Elementwise / reduce helpers
# ---------------------------------------------------------------------------

_SUPPORTED_UNARY = {
    "exp": "expf",
    "neg": None,
    "rsqrt": "rsqrtf",
    "recip": None,
    "relu": None,
    "tanh": "tanhf",
    "sigmoid": None,
    "abs": "fabsf",
}


def _apply_elementwise(fn: str, inputs: list[Expr]) -> Expr:
    if fn in {"add", "sub", "mul", "div", "mod"}:
        assert len(inputs) == 2
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[fn]
        return BinOp(op, inputs[0], inputs[1])
    if fn == "max":
        assert len(inputs) == 2
        return FuncCall("fmaxf", list(inputs))
    if fn == "min":
        assert len(inputs) == 2
        return FuncCall("fminf", list(inputs))
    if fn == "pow":
        assert len(inputs) == 2
        return FuncCall("powf", list(inputs))
    if fn == "neg":
        assert len(inputs) == 1
        return BinOp("-", Literal(0.0, "float"), inputs[0])
    if fn == "recip":
        assert len(inputs) == 1
        return BinOp("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        assert len(inputs) == 1
        return FuncCall("fmaxf", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        assert len(inputs) == 1
        neg_x = BinOp("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCall("expf", [neg_x])
        return BinOp("/", Literal(1.0, "float"), BinOp("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY:
        callee = _SUPPORTED_UNARY[fn]
        if callee is not None:
            return FuncCall(callee, list(inputs))
    raise NotImplementedError(f"elementwise fn={fn} not yet supported by emit")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _build_params(launch: LoopLaunch) -> tuple[list[GpuKernelParam], list[str]]:
    seen: list[str] = []
    for buf_name in launch.input_names:
        if buf_name not in seen and buf_name != launch.output_name:
            seen.append(buf_name)
    params = [GpuKernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(GpuKernelParam(dtype="float*", name=launch.output_name))
    return params, seen + [launch.output_name]


def _kernel_name(loop: LoopOp, idx: int) -> str:
    if any(isinstance(s, Update) for s in loop.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(launch: LoopLaunch, program: LoopProgram) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    loop: LoopOp = launch.loop
    out_shape = program.output_shape(launch)
    has_reduce = any(isinstance(s, Update) for s in loop.body)
    if has_reduce:
        free_extents = [int(a.extent) for a in loop.axes if a.kind == "free"]
        n_output = _numel(tuple(free_extents)) if free_extents else 1
    else:
        n_output = _numel(out_shape) if out_shape else _numel(tuple(a.extent for a in loop.axes))
    n_blocks = (n_output + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def _emit_kernel_source(gpu_kernel: GpuKernel) -> str:
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as _emit

    return _emit(gpu_kernel)


def _fresh(seq: list[int]) -> str:
    name = f"t{seq[0]}"
    seq[0] += 1
    return name


def _emit_select(stmt: Select, values: dict[str, Expr], axis_env: dict[str, Expr]) -> Expr:
    """Emit a chained ternary for a Select: each branch's predicate selects its value."""
    branches = list(stmt.branches)
    # Last branch is the catch-all (by convention: select=Literal(1) when the
    # rule had no explicit predicate). Build the chain right-to-left.
    result: Expr = values[branches[-1].value]
    for branch in reversed(branches[:-1]):
        cond = substitute(branch.select, axis_env)
        result = Ternary(cond=cond, if_true=values[branch.value], if_false=result)
    return result
