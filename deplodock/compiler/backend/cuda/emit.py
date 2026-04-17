"""Naive recursive-descent CUDA codegen: ``LoopProgram`` → ``GpuProgram``.

Consumes a ``LoopProgram`` (one ``LoopLaunch`` per GPU kernel, with
authoritative buffer shapes) and emits CUDA C source packaged as a
``GpuProgram``. The ``LoopOp`` body uses ``$N`` references for input
Ports; the codegen maps these to actual buffer names via
``LoopLaunch.input_names``.

Shapes are read from the ``LoopProgram`` (never recomputed) — see
``program.shape(...)``, ``program.dollar_shapes(launch)``,
``program.output_shape(launch)``.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.program import CudaLaunch
from deplodock.compiler.ir.expr import BinOp, Expr, FuncCall, Literal, Ternary, Var
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
from deplodock.compiler.ir.loop import Combine, LoopInput, LoopOp, Mux, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.program.gpu import GpuBuffer, GpuProgram
from deplodock.compiler.program.loop import LoopLaunch, LoopProgram

_BLOCK = 256


# ---------------------------------------------------------------------------
# Program-level entry
# ---------------------------------------------------------------------------


def compile_kernels(program: LoopProgram) -> GpuProgram:
    """Lower a ``LoopProgram`` to a ``GpuProgram``.

    One ``CudaLaunch`` is produced per ``LoopLaunch``. Buffer set is
    filtered to those actually referenced (either as launch input/output
    or as a graph-level constant that's referenced and shape-known).
    """
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
        kname = _kernel_name(launch.loop, i)
        gpu_kernel, arg_order = emit_kernel(launch, kname, program)
        source = _emit_kernel_source(gpu_kernel)
        grid, block = _launch_config(launch, program)
        launches.append(CudaLaunch(kernel_source=source, kernel_name=kname, grid=grid, block=block, args=arg_order))

    return GpuProgram(
        name=program.name,
        buffers=buffers,
        launches=launches,
        constant_values=dict(program.constant_values),
    )


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
# Input-tree walker
# ---------------------------------------------------------------------------


def _emit_input_value(
    inp: LoopInput,
    coord: Expr,
    buf_name: str,
    src_shape: tuple,
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Emit code to evaluate ``inp`` at the linear output coord ``coord``.

    ``buf_name`` is the external buffer name for this Port.
    ``src_shape`` is the source buffer's shape (for indexmap stride computation).
    """
    if isinstance(inp, Port):
        if inp.indexmap is not None:
            return _emit_indexmap_load(buf_name, inp.indexmap, coord, src_shape, stmts, name_seq)
        return ArrayAccess(array=buf_name, index=coord)

    if isinstance(inp, Mux):
        result: Expr | None = None
        for branch in reversed(inp.branches):
            sub = _emit_input_value(branch.input, coord, buf_name, src_shape, stmts, name_seq)
            result = sub if result is None else Ternary(cond=branch.select, if_true=sub, if_false=result)
        assert result is not None
        return result

    if isinstance(inp, Combine):
        source_vals: list[Expr] = []
        for src in inp.sources:
            expr = _emit_input_value(src, coord, buf_name, src_shape, stmts, name_seq)
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=expr))
            source_vals.append(Var(tname))

        source_iter = iter(source_vals)
        value = next(source_iter)
        for op in inp.ops:
            assert isinstance(op, ElementwiseOp)
            if op.info.arity == 1:
                value = _apply_elementwise(op.fn, [value])
            else:
                extra = next(source_iter, ArrayAccess(array="?", index=coord))
                value = _apply_elementwise(op.fn, [value, extra])
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=value))
            value = Var(tname)
        return value

    raise NotImplementedError(f"unexpected LoopInput variant: {type(inp).__name__}")


def _emit_mux_value(
    mux: Mux,
    coord: Expr,
    branch_infos: list[tuple[str, tuple]],
    out_shape: tuple,
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Emit code for a Mux input with per-branch buffer names.

    Decomposes the flat coord into per-axis output coordinates and
    substitutes them into the Mux branch select predicates.
    """
    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, substitute

    # Decompose flat coord into per-axis output coords.
    ndim = len(out_shape)
    mapping: dict[str, Expr] = {}
    remainder = coord
    for d in range(ndim - 1, -1, -1):
        dim_size = int(out_shape[d]) if isinstance(out_shape[d], int) else 1
        axis_var = f"{PLACEHOLDER_PREFIX}{d}"
        if d == 0:
            mapping[axis_var] = remainder
        else:
            mapping[axis_var] = BinOp("%", remainder, Literal(dim_size, "int"))
            remainder = BinOp("/", remainder, Literal(dim_size, "int"))

    result: Expr | None = None
    for i, branch in enumerate(reversed(mux.branches)):
        bi = len(mux.branches) - 1 - i
        buf_name, src_shape = branch_infos[bi] if bi < len(branch_infos) else ("?", ())

        # Substitute placeholders in the branch's select predicate.
        select_expr = substitute(branch.select, mapping) if branch.select is not None else None

        # Emit the branch's input value with coordinate mapping.
        sub = _emit_input_value(branch.input, coord, buf_name, src_shape, stmts, name_seq)
        if select_expr is not None:
            result = sub if result is None else Ternary(cond=select_expr, if_true=sub, if_false=result)
        else:
            result = sub if result is None else result
    assert result is not None
    return result


def _emit_indexmap_load(buf_name: str, indexmap, coord: Expr, src_shape: tuple, stmts: list[Stmt], name_seq: list[int]) -> Expr:
    from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, substitute

    assert len(indexmap.sources) == 1
    src = indexmap.sources[0]
    out_shape = indexmap.out_shape
    ndim = len(out_shape)

    max_placeholder = ndim - 1
    for cm in src.coord_map:
        for var_name in _collect_var_names(cm):
            if var_name.startswith(PLACEHOLDER_PREFIX):
                idx = int(var_name[len(PLACEHOLDER_PREFIX) :])
                max_placeholder = max(max_placeholder, idx)

    mapping: dict[str, Expr] = {}
    remainder = coord
    effective_ndim = max_placeholder + 1
    effective_shape = tuple(out_shape) + (1,) * (effective_ndim - ndim)
    for d in range(effective_ndim - 1, -1, -1):
        dim_size = int(effective_shape[d]) if d < len(effective_shape) and isinstance(effective_shape[d], int) else 1
        axis_var = f"{PLACEHOLDER_PREFIX}{d}"
        if d == 0:
            mapping[axis_var] = remainder
        else:
            mapping[axis_var] = BinOp("%", remainder, Literal(dim_size, "int"))
            remainder = BinOp("/", remainder, Literal(dim_size, "int"))

    input_coords = [substitute(cm, mapping) for cm in src.coord_map]

    flat_idx: Expr = Literal(0, "int")
    stride = 1
    for d in range(len(input_coords) - 1, -1, -1):
        if stride == 1:
            flat_idx = input_coords[d]
        else:
            flat_idx = BinOp("+", BinOp("*", input_coords[d], Literal(stride, "int")), flat_idx)
        if d > 0:
            dim = int(src_shape[d]) if d < len(src_shape) and isinstance(src_shape[d], int) else 1
            stride *= dim

    return ArrayAccess(array=buf_name, index=flat_idx)


# ---------------------------------------------------------------------------
# Body dispatch
# ---------------------------------------------------------------------------


def _emit_body(
    launch: LoopLaunch, out_shape: tuple, dollar_shapes: dict[str, tuple], program: LoopProgram
) -> tuple[list[Stmt], tuple[int, int, int]]:
    from deplodock.compiler.backend.kernel_plan import analyze_kernel

    plan = analyze_kernel(launch.loop, dollar_shapes, out_shape)
    idx = Var("idx")
    idx_init = VarDecl(
        dtype="long long",
        name="idx",
        init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
    )
    guarded = _emit_plan(plan, launch, dollar_shapes, program, idx)
    stmts: list[Stmt] = [idx_init, IfStmt(cond=BinOp("<", idx, Literal(plan.n_output, "int")), body=guarded)]
    return stmts, (_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Plan-based emitter
# ---------------------------------------------------------------------------


def _emit_plan(plan, launch: LoopLaunch, dollar_shapes: dict[str, tuple], program: LoopProgram, idx: Expr) -> list[Stmt]:
    from deplodock.compiler.backend.kernel_plan import Inline, Loop

    loop = launch.loop
    stmts: list[Stmt] = []
    name_seq = [0]
    values: dict[str, Expr] = {}
    out_shape = program.output_shape(launch)

    # Build $N → (port, buf_name, src_shape) mapping.
    # For Mux inputs, enumerate all leaf Ports inside branches.
    port_info: dict[str, tuple[Port, str, tuple]] = {}
    mux_info: dict[str, list[tuple[str, str, tuple]]] = {}  # key → [(branch_buf, branch_src_shape), ...]
    port_idx = 0
    for inp in loop.inputs:
        if isinstance(inp, Port):
            key = f"${port_idx}"
            buf_name = launch.input_names[port_idx] if port_idx < len(launch.input_names) else key
            src_shape = program.shape(buf_name) if buf_name in {b.name for b in program.buffers} else dollar_shapes.get(key, ())
            port_info[key] = (inp, buf_name, src_shape)
            port_idx += 1
        elif isinstance(inp, Mux):
            key = f"${port_idx}"
            branch_infos = []
            for branch in inp.branches:
                if isinstance(branch.input, Port):
                    bname = launch.input_names[port_idx] if port_idx < len(launch.input_names) else f"${port_idx}"
                    bshape = program.shape(bname) if bname in {b.name for b in program.buffers} else ()
                    branch_infos.append((bname, bshape))
                    port_idx += 1
            mux_info[key] = branch_infos
            # For port_info, use the first branch as representative
            if branch_infos:
                port_info[key] = (Port(), branch_infos[0][0], branch_infos[0][1])

    # Load per-row ports upfront.
    for key, (port, buf_name, src_shape) in port_info.items():
        if key not in plan.per_elem_ports:
            inp = loop.inputs[int(key[1:])] if key[1:].isdigit() else port
            if isinstance(inp, Mux) and key in mux_info:
                values[key] = _emit_mux_value(inp, idx, mux_info[key], out_shape, stmts, name_seq)
            else:
                values[key] = _emit_input_value(port, idx, buf_name, src_shape, stmts, name_seq)

    loop_count = 0
    for step in plan.steps:
        if isinstance(step, Inline):
            for assign in step.body:
                arg_exprs = [values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                stmts.append(VarDecl(dtype="float", name=tname, init=value))
                values[assign.name] = Var(tname)

        elif isinstance(step, Loop):
            k_var_name = f"k{loop_count}"
            k_var = Var(k_var_name)

            if step.accum:
                stmts.append(VarDecl(dtype="float", name=step.accum.var, init=Literal(step.accum.identity, "float")))

            inner: list[Stmt] = []
            loop_values: dict[str, Expr] = dict(values)

            # Load per-element ports at broadcast coords.
            bcast = _decompose_broadcast_coords(idx, k_var, step.iter_shape, step.reduce_axis)
            for key, (port, buf_name, src_shape) in port_info.items():
                if key in plan.per_elem_ports:
                    ps = _port_shape_from_info(port, src_shape)
                    pidx = _flatten_coords_for_port(bcast, step.iter_shape, ps)
                    loop_values[key] = _emit_input_value(port, pidx, buf_name, src_shape, inner, name_seq)

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
                store_idx = _broadcast_load_idx(idx, k_var, step.iter_shape, step.reduce_axis)
                last = step.body[-1]
                inner.append(IrAssign(target=ArrayAccess(array=launch.output_name, index=store_idx), value=loop_values[last.name]))

            stmts.append(ForLoop(var=k_var_name, start=Literal(0, "int"), end=Literal(step.k_size, "int"), body=inner))
            loop_count += 1

    if plan.stores_final:
        if loop.body:
            out_val = values[loop.body[-1].name]
        else:
            # Copy kernel: use first available value (Port or Mux).
            first_key = next((k for k in sorted(values) if k.startswith("$")), None)
            out_val = values[first_key] if first_key else Literal(0.0, "float")
        stmts.append(IrAssign(target=ArrayAccess(array=launch.output_name, index=idx), value=out_val))

    return stmts


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    if fn == "max":
        from deplodock.compiler.ir.gpu import VarAssign

        return VarAssign(name=acc_name, value=FuncCall("fmaxf", [Var(acc_name), value]))
    op = {"sum": "+=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _decompose_broadcast_coords(row: Expr, k: Expr, broadcast_shape: tuple, reduce_axis: int) -> list[Expr]:
    ndim = len(broadcast_shape)
    if ndim <= 1:
        return [k]
    outer_shape = list(broadcast_shape)
    del outer_shape[reduce_axis]
    outer_coords: list[Expr] = []
    remainder = row
    for d in range(len(outer_shape) - 1, -1, -1):
        dim = int(outer_shape[d])
        if d == 0:
            outer_coords.insert(0, remainder)
        else:
            outer_coords.insert(0, BinOp("/", remainder, Literal(dim, "int")))
            remainder = BinOp("%", remainder, Literal(dim, "int"))
            outer_coords[0], remainder = remainder, outer_coords[0]
    full_coords = list(outer_coords)
    full_coords.insert(reduce_axis, k)
    return full_coords


def _flatten_coords_for_port(broadcast_coords: list[Expr], broadcast_shape: tuple, port_shape: tuple) -> Expr:
    ndim = len(broadcast_shape)
    port_ndim = len(port_shape)
    if port_ndim == 0:
        return Literal(0, "int")
    pad = ndim - port_ndim
    flat: Expr = Literal(0, "int")
    stride = 1
    for d in range(ndim - 1, -1, -1):
        pd = d - pad
        if pd < 0:
            continue
        p_dim = int(port_shape[pd])
        b_dim = int(broadcast_shape[d])
        coord = Literal(0, "int") if (p_dim == 1 and b_dim > 1) else broadcast_coords[d]
        term = coord if stride == 1 else BinOp("*", coord, Literal(stride, "int"))
        flat = term if (isinstance(flat, Literal) and flat.value == 0) else BinOp("+", term, flat)
        stride *= p_dim
    return flat


def _broadcast_load_idx(row: Expr, k: Expr, broadcast_shape: tuple, reduce_axis: int) -> Expr:
    coords = _decompose_broadcast_coords(row, k, broadcast_shape, reduce_axis)
    return _flatten_coords_for_port(coords, broadcast_shape, broadcast_shape)


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


def _port_shape_from_info(port: Port, src_shape: tuple) -> tuple:
    if port.indexmap is not None:
        return tuple(port.indexmap.out_shape)
    return tuple(src_shape)


def _build_params(launch: LoopLaunch) -> tuple[list[GpuKernelParam], list[str]]:
    seen: list[str] = []
    for buf_name in launch.input_names:
        if buf_name not in seen and buf_name != launch.output_name:
            seen.append(buf_name)
    params = [GpuKernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(GpuKernelParam(dtype="float*", name=launch.output_name))
    return params, seen + [launch.output_name]


def _kernel_name(loop: LoopOp, idx: int) -> str:
    if any(isinstance(a.op, ReduceOp) for a in loop.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(launch: LoopLaunch, program: LoopProgram) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    loop = launch.loop
    dollar_shapes = program.dollar_shapes(launch)
    out_shape = program.output_shape(launch)
    has_reduce = any(isinstance(a.op, ReduceOp) for a in loop.body)
    if has_reduce:
        n_output = _reduce_n_rows(loop, dollar_shapes)
    else:
        n_output = _numel(out_shape) if out_shape else _numel(loop.infer_output_shape(dollar_shapes))
    n_blocks = (n_output + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def _reduce_n_rows(loop: LoopOp, shapes: dict[str, tuple]) -> int:
    ssa_shapes = loop.infer_shapes(shapes)
    for assign in loop.body:
        if isinstance(assign.op, ReduceOp):
            pre_shape = tuple(int(d) for d in ssa_shapes[assign.args[0]])
            reduce_axis = assign.op.axis % len(pre_shape)
            outer = list(pre_shape)
            del outer[reduce_axis]
            return _numel(tuple(outer)) if outer else 1
    return _numel(loop.infer_output_shape(shapes))


def _emit_kernel_source(gpu_kernel: GpuKernel) -> str:
    from deplodock.compiler.backend.kernel_codegen import emit_kernel as _emit

    return _emit(gpu_kernel)


def _collect_var_names(expr) -> list[str]:
    if isinstance(expr, Var):
        return [expr.name]
    names: list[str] = []
    for attr in ("left", "right", "expr", "cond", "then", "else_"):
        child = getattr(expr, attr, None)
        if child is not None:
            names.extend(_collect_var_names(child))
    for attr in ("args",):
        children = getattr(expr, attr, None)
        if children is not None and isinstance(children, (list, tuple)):
            for c in children:
                names.extend(_collect_var_names(c))
    return names


def _fresh(seq: list[int]) -> str:
    name = f"t{seq[0]}"
    seq[0] += 1
    return name
