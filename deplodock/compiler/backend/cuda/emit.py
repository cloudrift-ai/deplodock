"""Naive recursive-descent CUDA codegen from structural ``KernelOp``.

Receives ``KernelInfo`` objects (KernelOp + buffer name mappings) and
emits CUDA C source. The KernelOp body uses ``$N`` references for input
Ports; the codegen maps these to actual buffer names via ``KernelInfo``.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.program import CudaLaunch
from deplodock.compiler.backend.ir.expr import BinOp, Expr, FuncCall, Literal, Ternary, Var
from deplodock.compiler.backend.ir.kernel_ir import (
    ArrayAccess,
    AugAssign,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    Stmt,
    VarDecl,
)
from deplodock.compiler.backend.ir.kernel_ir import (
    Assign as IrAssign,
)
from deplodock.compiler.backend.program import Buffer, Program
from deplodock.compiler.lower import KernelInfo
from deplodock.compiler.ops import (
    Combine,
    ElementwiseOp,
    KernelInput,
    KernelOp,
    Mux,
    Port,
    ReduceOp,
)

_BLOCK = 256


# ---------------------------------------------------------------------------
# Program-level entry
# ---------------------------------------------------------------------------


def compile_kernels(
    kernels: list[KernelInfo],
    *,
    name: str = "prog",
    buf_shapes: dict[str, tuple] | None = None,
    graph_inputs: list[str] | None = None,
    graph_outputs: list[str] | None = None,
    graph_constants: list[str] | None = None,
) -> Program:
    shapes = dict(buf_shapes or {})
    graph_input_set = set(graph_inputs or [])
    graph_output_set = set(graph_outputs or [])
    graph_constant_set = set(graph_constants or [])

    # Collect referenced buffer shapes.
    for info in kernels:
        for i, port in enumerate(_leaf_ports(info.kernel)):
            buf_name = info.input_names[i] if i < len(info.input_names) else f"${i}"
            if buf_name not in shapes and port.indexmap is not None:
                shapes[buf_name] = tuple(port.indexmap.out_shape)
        dollar_shapes = _dollar_shapes(info, shapes)
        out_shape = info.output_shape or info.kernel.infer_output_shape(dollar_shapes)
        shapes.setdefault(info.output_name, out_shape)

    def role_for(bid: str) -> str:
        if bid in graph_input_set:
            return "input"
        if bid in graph_constant_set:
            return "constant"
        if bid in graph_output_set:
            return "output"
        return "scratch"

    referenced: set[str] = set()
    for info in kernels:
        referenced.update(info.input_names)
        referenced.add(info.output_name)
    referenced |= graph_constant_set & set(shapes.keys())

    buffers = [
        Buffer(name=bid, size=_numel(shape), dtype="float", role=role_for(bid)) for bid, shape in shapes.items() if bid in referenced
    ]

    launches: list[CudaLaunch] = []
    for i, info in enumerate(kernels):
        kname = _kernel_name(info.kernel, i)
        dollar_shapes = _dollar_shapes(info, shapes)
        kernel_def, arg_order = emit_kernel(info, kname, dollar_shapes, shapes)
        source = _emit_kernel_source(kernel_def)
        grid, block = _launch_config(info.kernel, dollar_shapes, info.output_shape)
        launches.append(CudaLaunch(kernel_source=source, kernel_name=kname, grid=grid, block=block, args=arg_order))

    return Program(name=name, buffers=buffers, launches=launches)


def _dollar_shapes(info: KernelInfo, buf_shapes: dict[str, tuple] | None = None) -> dict[str, tuple]:
    """Build $N → shape mapping from KernelInfo for infer_shapes."""
    ext = buf_shapes or {}
    result: dict[str, tuple] = {}
    port_idx = 0
    for inp in info.kernel.inputs:
        if isinstance(inp, Port):
            key = f"${port_idx}"
            buf_name = info.input_names[port_idx] if port_idx < len(info.input_names) else key
            if inp.indexmap is not None:
                result[key] = tuple(inp.indexmap.out_shape)
            elif buf_name in ext:
                result[key] = tuple(ext[buf_name])
            port_idx += 1
    return result


# ---------------------------------------------------------------------------
# Per-kernel emission
# ---------------------------------------------------------------------------


def emit_kernel(
    info: KernelInfo, kernel_name: str, shapes: dict[str, tuple], buf_shapes: dict[str, tuple] | None = None
) -> tuple[KernelDef, list[str]]:
    out_shape = info.output_shape or info.kernel.infer_output_shape(shapes)
    params, arg_order = _build_params(info)
    body, block_size = _emit_body(info, out_shape, shapes, buf_shapes or {})
    kd = KernelDef(name=kernel_name, params=params, body=body, block_size=block_size)
    return kd, arg_order


# ---------------------------------------------------------------------------
# Input-tree walker
# ---------------------------------------------------------------------------


def _emit_input_value(
    inp: KernelInput,
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

    raise NotImplementedError(f"unexpected KernelInput variant: {type(inp).__name__}")


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
    from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

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
    from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

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
    info: KernelInfo, out_shape: tuple, shapes: dict[str, tuple], buf_shapes: dict[str, tuple]
) -> tuple[list[Stmt], tuple[int, int, int]]:
    from deplodock.compiler.backend.kernel_plan import analyze_kernel

    plan = analyze_kernel(info.kernel, shapes, out_shape)
    idx = Var("idx")
    idx_init = VarDecl(
        dtype="long long",
        name="idx",
        init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
    )
    guarded = _emit_plan(plan, info, shapes, buf_shapes, idx)
    stmts: list[Stmt] = [idx_init, IfStmt(cond=BinOp("<", idx, Literal(plan.n_output, "int")), body=guarded)]
    return stmts, (_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Plan-based emitter
# ---------------------------------------------------------------------------


def _emit_plan(plan, info: KernelInfo, shapes: dict[str, tuple], buf_shapes: dict[str, tuple], idx: Expr) -> list[Stmt]:
    from deplodock.compiler.backend.kernel_plan import Inline, Loop

    kernel = info.kernel
    stmts: list[Stmt] = []
    name_seq = [0]
    values: dict[str, Expr] = {}

    # Build $N → (port, buf_name, src_shape) mapping.
    # For Mux inputs, enumerate all leaf Ports inside branches.
    port_info: dict[str, tuple[Port, str, tuple]] = {}
    mux_info: dict[str, list[tuple[str, str, tuple]]] = {}  # key → [(branch_buf, branch_src_shape), ...]
    port_idx = 0
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            key = f"${port_idx}"
            buf_name = info.input_names[port_idx] if port_idx < len(info.input_names) else key
            src_shape = buf_shapes.get(buf_name, shapes.get(key, ()))
            port_info[key] = (inp, buf_name, src_shape)
            port_idx += 1
        elif isinstance(inp, Mux):
            key = f"${port_idx}"
            branch_infos = []
            for branch in inp.branches:
                if isinstance(branch.input, Port):
                    bname = info.input_names[port_idx] if port_idx < len(info.input_names) else f"${port_idx}"
                    bshape = buf_shapes.get(bname, ())
                    branch_infos.append((bname, bshape))
                    port_idx += 1
            mux_info[key] = branch_infos
            # For port_info, use the first branch as representative
            if branch_infos:
                port_info[key] = (Port(), branch_infos[0][0], branch_infos[0][1])

    # Load per-row ports upfront.
    for key, (port, buf_name, src_shape) in port_info.items():
        if key not in plan.per_elem_ports:
            inp = kernel.inputs[int(key[1:])] if key[1:].isdigit() else port
            if isinstance(inp, Mux) and key in mux_info:
                values[key] = _emit_mux_value(inp, idx, mux_info[key], info.output_shape, stmts, name_seq)
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
                inner.append(IrAssign(target=ArrayAccess(array=info.output_name, index=store_idx), value=loop_values[last.name]))

            stmts.append(ForLoop(var=k_var_name, start=Literal(0, "int"), end=Literal(step.k_size, "int"), body=inner))
            loop_count += 1

    if plan.stores_final:
        if kernel.body:
            out_val = values[kernel.body[-1].name]
        else:
            # Copy kernel: use first available value (Port or Mux).
            first_key = next((k for k in sorted(values) if k.startswith("$")), None)
            out_val = values[first_key] if first_key else Literal(0.0, "float")
        stmts.append(IrAssign(target=ArrayAccess(array=info.output_name, index=idx), value=out_val))

    return stmts


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    if fn == "max":
        from deplodock.compiler.backend.ir.kernel_ir import VarAssign

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


def _leaf_ports(kernel: KernelOp) -> list[Port]:
    leaves: list[Port] = []

    def walk(inp: KernelInput) -> None:
        if isinstance(inp, Port):
            leaves.append(inp)
        elif isinstance(inp, Mux):
            for b in inp.branches:
                walk(b.input)
        elif isinstance(inp, Combine):
            for s in inp.sources:
                walk(s)

    for inp in kernel.inputs:
        walk(inp)
    return leaves


def _build_params(info: KernelInfo) -> tuple[list[KernelParam], list[str]]:
    seen: list[str] = []
    for buf_name in info.input_names:
        if buf_name not in seen and buf_name != info.output_name:
            seen.append(buf_name)
    params = [KernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(KernelParam(dtype="float*", name=info.output_name))
    return params, seen + [info.output_name]


def _kernel_name(kernel: KernelOp, idx: int) -> str:
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp, shapes: dict[str, tuple], out_shape: tuple = ()) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    has_reduce = any(isinstance(a.op, ReduceOp) for a in kernel.body)
    if has_reduce:
        n_output = _reduce_n_rows(kernel, shapes)
    else:
        n_output = _numel(out_shape) if out_shape else _numel(kernel.infer_output_shape(shapes))
    n_blocks = (n_output + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def _reduce_n_rows(kernel: KernelOp, shapes: dict[str, tuple]) -> int:
    ssa_shapes = kernel.infer_shapes(shapes)
    for assign in kernel.body:
        if isinstance(assign.op, ReduceOp):
            pre_shape = tuple(int(d) for d in ssa_shapes[assign.args[0]])
            reduce_axis = assign.op.axis % len(pre_shape)
            outer = list(pre_shape)
            del outer[reduce_axis]
            return _numel(tuple(outer)) if outer else 1
    return _numel(kernel.infer_output_shape(shapes))


def _emit_kernel_source(kernel_def: KernelDef) -> str:
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel as _emit

    return _emit(kernel_def)


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
