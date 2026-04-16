"""Naive recursive-descent CUDA codegen from structural ``KernelOp``.

Walks the structural IR (Port | Mux | Combine inputs, SSA body of
``Assign`` statements, outputs) and emits one ``KernelDef`` per
``KernelOp`` directly -- no classification pass, no Schedule, no
LoopIR intermediate.

Body emission dispatches to one of two code-shapes based solely on
the presence of ``ReduceOp`` in the SSA body:

- ``_emit_flat``: 1D grid over flat output numel, 256 threads/block.
  One thread per element; walks the SSA body as inline expressions.
- ``_emit_segments``: 1D grid over post-reduce rows, one thread per
  row.  The body is split into segments at ReduceOp boundaries; each
  segment that touches per-element values gets its own K-loop.
  Cross-boundary per-element dependencies are recomputed (not stored).

All shapes are treated as flat buffers indexed by a single linearized
offset.
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
from deplodock.compiler.ops import (
    Combine,
    ElementwiseOp,
    KernelInput,
    KernelOp,
    Mux,
    Port,
    ReduceOp,
)

_BLOCK = 256  # thread count per block for 1D pointwise / reduce-target parallelism


# ---------------------------------------------------------------------------
# Program-level entry: list[KernelOp] -> Program
# ---------------------------------------------------------------------------


def compile_kernels(
    kernels: list[KernelOp],
    *,
    name: str = "prog",
    buf_shapes: dict[str, tuple] | None = None,
    graph_inputs: list[str] | None = None,
    graph_outputs: list[str] | None = None,
    graph_constants: list[str] | None = None,
) -> Program:
    """Assemble a ``Program`` from a list of structural ``KernelOp``.

    ``buf_shapes`` maps buffer names → shapes for all external buffers.
    ``graph_inputs`` / ``graph_outputs`` / ``graph_constants`` mark roles.
    """
    shapes = dict(buf_shapes or {})
    graph_input_set = set(graph_inputs or [])
    graph_output_set = set(graph_outputs or [])
    graph_constant_set = set(graph_constants or [])

    # Collect ALL referenced buffer shapes: leaf Ports from inputs +
    # output Ports. Kernels are in topo order, so a prior kernel's output
    # shape is available for a later kernel's input.
    for k in kernels:
        for port in _leaf_ports(k):
            if port.buffer_id not in shapes:
                if port.indexmap is not None:
                    shapes[port.buffer_id] = tuple(port.indexmap.out_shape)
        out_shape = k.infer_output_shape(shapes)
        for out in k.outputs:
            if isinstance(out, Port):
                shapes.setdefault(out.buffer_id, out_shape)

    def role_for(bid: str) -> str:
        if bid in graph_input_set:
            return "input"
        if bid in graph_constant_set:
            return "constant"
        if bid in graph_output_set:
            return "output"
        return "scratch"

    # Only allocate buffers that are actually referenced by kernel launches.
    referenced: set[str] = set()
    for k in kernels:
        for port in _leaf_ports(k):
            referenced.add(port.buffer_id)
        for out in k.outputs:
            if isinstance(out, Port):
                referenced.add(out.buffer_id)
    referenced |= graph_constant_set & set(shapes.keys())

    buffers = [
        Buffer(name=bid, size=_numel(shape), dtype="float", role=role_for(bid)) for bid, shape in shapes.items() if bid in referenced
    ]

    launches: list[CudaLaunch] = []
    for i, k in enumerate(kernels):
        kname = _kernel_name(k, i, shapes)
        kernel_def, arg_order = emit_kernel(k, kname, shapes)
        source = _emit_kernel_source(kernel_def)
        grid, block = _launch_config(k, shapes)
        launches.append(
            CudaLaunch(
                kernel_source=source,
                kernel_name=kname,
                grid=grid,
                block=block,
                args=arg_order,
            )
        )

    return Program(name=name, buffers=buffers, launches=launches)


# ---------------------------------------------------------------------------
# Per-kernel emission
# ---------------------------------------------------------------------------


def emit_kernel(kernel: KernelOp, kernel_name: str, shapes: dict[str, tuple]) -> tuple[KernelDef, list[str]]:
    """Emit a single ``KernelOp`` as a ``KernelDef`` + ordered launch args."""
    out_shape = kernel.infer_output_shape(shapes)
    params, arg_order = _build_params(kernel)
    body, block_size = _emit_body(kernel, out_shape, shapes)
    kd = KernelDef(name=kernel_name, params=params, body=body, block_size=block_size)
    return kd, arg_order


# ---------------------------------------------------------------------------
# Input-tree walker: evaluate a KernelInput at the current output coord
# ---------------------------------------------------------------------------


def _emit_input_value(
    inp: KernelInput,
    coord: Expr,
    shapes: dict[str, tuple],
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Emit code to evaluate ``inp`` at the linear output coord ``coord``."""
    if isinstance(inp, Port):
        if inp.indexmap is not None:
            src_shape = shapes.get(inp.buffer_id, ())
            return _emit_indexmap_load(inp, coord, src_shape, stmts, name_seq)
        return ArrayAccess(array=inp.buffer_id, index=coord)

    if isinstance(inp, Mux):
        result: Expr | None = None
        for branch in reversed(inp.branches):
            sub = _emit_input_value(branch.input, coord, shapes, stmts, name_seq)
            result = sub if result is None else Ternary(cond=branch.select, then=sub, else_=result)
        assert result is not None
        return result

    if isinstance(inp, Combine):
        source_vals: list[Expr] = []
        for src in inp.sources:
            expr = _emit_input_value(src, coord, shapes, stmts, name_seq)
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


def _emit_indexmap_load(port: Port, coord: Expr, src_shape: tuple, stmts: list[Stmt], name_seq: list[int]) -> Expr:
    """Emit a load from ``port.buffer_id`` at the coord transformed by ``port.indexmap``."""
    from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

    indexmap = port.indexmap
    assert indexmap is not None and len(indexmap.sources) == 1
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

    return ArrayAccess(array=port.buffer_id, index=flat_idx)


# ---------------------------------------------------------------------------
# Body dispatch
# ---------------------------------------------------------------------------


def _emit_body(kernel: KernelOp, out_shape: tuple, shapes: dict[str, tuple]) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Analyze kernel into a plan, then emit statements from it."""
    from deplodock.compiler.backend.kernel_plan import analyze_kernel

    plan = analyze_kernel(kernel, shapes, out_shape)
    idx = Var("idx")
    idx_init = VarDecl(
        dtype="int",
        name="idx",
        init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
    )
    guarded = _emit_plan(plan, kernel, shapes, idx)
    stmts: list[Stmt] = [idx_init, IfStmt(cond=BinOp("<", idx, Literal(plan.n_output, "int")), body=guarded)]
    return stmts, (_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Plan-based emitter: reduce / contraction kernels
# ---------------------------------------------------------------------------


def _emit_plan(plan, kernel: KernelOp, shapes: dict[str, tuple], idx: Expr) -> list[Stmt]:
    """Walk a KernelPlan and emit statements. No analysis — just code generation."""
    from deplodock.compiler.backend.kernel_plan import Inline, Loop

    stmts: list[Stmt] = []
    name_seq = [0]
    values: dict[str, Expr] = {}

    input_ports = {p.buffer_id: p for p in kernel.inputs if isinstance(p, Port)}

    # Load per-row ports upfront (outside all loops).
    for name, port in input_ports.items():
        if name not in plan.per_elem_ports:
            values[name] = _emit_input_value(port, idx, shapes, stmts, name_seq)

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

            # Declare accumulator outside loop.
            if step.accum:
                stmts.append(VarDecl(dtype="float", name=step.accum.var, init=Literal(step.accum.identity, "float")))

            inner: list[Stmt] = []
            loop_values: dict[str, Expr] = dict(values)

            # Load per-element ports at broadcast coords inside the loop.
            bcast = _decompose_broadcast_coords(idx, k_var, step.iter_shape, step.reduce_axis)
            for name, port in input_ports.items():
                if name in plan.per_elem_ports:
                    ps = _port_shape(port, shapes)
                    pidx = _flatten_coords_for_port(bcast, step.iter_shape, ps)
                    loop_values[name] = _emit_input_value(port, pidx, shapes, inner, name_seq)

            # Recompute prior element-space deps.
            for assign in step.recompute:
                arg_exprs = [loop_values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[assign.name] = Var(tname)

            # This segment's elementwise body.
            for assign in step.body:
                arg_exprs = [loop_values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[assign.name] = Var(tname)

            # Accumulate or store.
            if step.accum:
                src = loop_values[step.accum.src]
                inner.append(_emit_reduce_accum(step.accum.var, step.accum.fn, src))
                values[step.accum.result] = Var(step.accum.var)

            if step.stores_output:
                out_port = kernel.outputs[0]
                assert isinstance(out_port, Port)
                store_idx = _broadcast_load_idx(idx, k_var, step.iter_shape, step.reduce_axis)
                last = step.body[-1]
                inner.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=store_idx), value=loop_values[last.name]))

            stmts.append(ForLoop(var=k_var_name, start=Literal(0, "int"), end=Literal(step.k_size, "int"), body=inner))
            loop_count += 1

    # Store final value after all loops / inline steps.
    if plan.stores_final:
        out_port = kernel.outputs[0]
        assert isinstance(out_port, Port)
        if kernel.body:
            out_val = values[kernel.body[-1].name]
        else:
            # Copy kernel (empty body): forward the first input port.
            first_port = next((p for p in kernel.inputs if isinstance(p, Port)), None)
            out_val = values[first_port.buffer_id] if first_port else Literal(0.0, "float")
        stmts.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=idx), value=out_val))

    return stmts


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    """Emit accumulator update: ``acc += val`` or ``acc = fmaxf(acc, val)``."""
    if fn == "max":
        from deplodock.compiler.backend.ir.kernel_ir import VarAssign

        return VarAssign(name=acc_name, value=FuncCall("fmaxf", [Var(acc_name), value]))
    op = {"sum": "+=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _decompose_broadcast_coords(row: Expr, k: Expr, broadcast_shape: tuple, reduce_axis: int) -> list[Expr]:
    """Decompose ``(row, k)`` into per-dimension broadcast coordinates."""
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
    """Flatten broadcast coords into a flat index, clamping broadcast dims to 0."""
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
    """Flat coord into ``broadcast_shape`` from ``row`` and ``k``."""
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


def _port_shape(inp, shapes: dict) -> tuple:
    if isinstance(inp, Port):
        if inp.indexmap is not None:
            return tuple(inp.indexmap.out_shape)
        return tuple(shapes.get(inp.buffer_id, ()))
    if isinstance(inp, Combine):
        if inp.sources:
            return _port_shape(inp.sources[0], shapes)
    if isinstance(inp, Mux):
        if inp.branches:
            return _port_shape(inp.branches[0].input, shapes)
    return ()


def _leaf_ports(kernel: KernelOp) -> list[Port]:
    """Collect every external ``Port`` leaf referenced by the kernel."""
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


def _build_params(kernel: KernelOp) -> tuple[list[KernelParam], list[str]]:
    """Determine kernel params + launch arg order."""
    seen: list[str] = []
    for p in _leaf_ports(kernel):
        if p.buffer_id not in seen and p.buffer_id not in {o.buffer_id for o in kernel.outputs if isinstance(o, Port)}:
            seen.append(p.buffer_id)
    output_ids = [o.buffer_id for o in kernel.outputs if isinstance(o, Port)]
    params = [KernelParam(dtype="const float*", name=bid) for bid in seen]
    params += [KernelParam(dtype="float*", name=bid) for bid in output_ids]
    return params, seen + output_ids


def _kernel_name(kernel: KernelOp, idx: int, shapes: dict[str, tuple]) -> str:
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp, shapes: dict[str, tuple]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    has_reduce = any(isinstance(a.op, ReduceOp) for a in kernel.body)
    if has_reduce:
        n_output = _reduce_n_rows(kernel, shapes)
    else:
        n_output = _numel(kernel.infer_output_shape(shapes))
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
