"""Naive recursive-descent CUDA codegen from structural ``KernelOp``.

Walks the structural IR (Port | Mux | Combine inputs, optional
ContractionCore, reduce_stages, epilogue, outputs) and emits one
``KernelDef`` per ``KernelOp`` directly — no classification pass, no
Schedule, no LoopIR intermediate.

Naive schedule policy:

- **Pointwise** (``contraction is None``, ``reduce_stages == ()``):
  1D grid over the flattened output numel; one thread per output coord;
  block size fixed at 256.
- **Reduce chain** (``reduce_stages`` non-empty, no contraction): 1D grid
  over the post-reduce outer shape; one block per reduced row; one
  thread per block. (Block-level parallelism is future work.)
- **Contraction** (``contraction is not None``): 2D grid over (M, N);
  one thread per output coord; the K-loop runs serially inside the
  thread.

All shapes are treated as flat buffers: each input tensor has a known
shape (``kernel.external_shapes[buffer_id]``) and is indexed by a single
linearized offset. Broadcast / transpose / multi-source reads via
``Mux`` are not yet handled — every Port is a direct contiguous read.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.program import CudaLaunch
from deplodock.compiler.backend.ir.expr import BinOp, Expr, FuncCall, Literal, Ternary, Var
from deplodock.compiler.backend.ir.kernel_ir import (
    ArrayAccess,
    Assign,
    AugAssign,
    ForLoop,
    IfStmt,
    KernelDef,
    KernelParam,
    RawCode,
    Stmt,
    VarDecl,
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
    graph_inputs: list[str] | None = None,
    graph_outputs: list[str] | None = None,
    graph_constants: list[str] | None = None,
) -> Program:
    """Assemble a ``Program`` from a list of structural ``KernelOp``.

    ``graph_inputs`` / ``graph_outputs`` / ``graph_constants`` mark buffer
    roles; anything else produced by one kernel and consumed by another
    is "scratch". Constants are initialized from ``input_data`` or with
    pseudorandom values at runtime (same as inputs).
    """
    graph_input_set = set(graph_inputs or [])
    graph_output_set = set(graph_outputs or [])
    graph_constant_set = set(graph_constants or [])

    # Discover every external buffer + its shape.
    buf_shapes: dict[str, tuple] = {}
    for k in kernels:
        for bid, shape in k.external_shapes.items():
            buf_shapes[bid] = tuple(shape)
        for out in k.outputs:
            if isinstance(out, Port):
                buf_shapes.setdefault(out.buffer_id, _infer_output_shape(k))

    def role_for(bid: str) -> str:
        if bid in graph_input_set:
            return "input"
        if bid in graph_constant_set:
            return "constant"
        if bid in graph_output_set:
            return "output"
        return "scratch"

    buffers = [Buffer(name=bid, size=_numel(shape), dtype="float", role=role_for(bid)) for bid, shape in buf_shapes.items()]

    launches: list[CudaLaunch] = []
    for i, k in enumerate(kernels):
        kname = _kernel_name(k, i)
        kernel_def, arg_order = emit_kernel(k, kname)
        source = _emit_kernel_source(kernel_def)
        grid, block = _launch_config(k)
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


def emit_kernel(kernel: KernelOp, kernel_name: str) -> tuple[KernelDef, list[str]]:
    """Emit a single ``KernelOp`` as a ``KernelDef`` + ordered launch args."""
    out_shape = _infer_output_shape(kernel)
    params, arg_order = _build_params(kernel)

    has_reduce = any(isinstance(op, ReduceOp) for op in kernel.body)
    if kernel.contraction is not None:
        body, block_size = _emit_contraction_body(kernel, out_shape)
    elif has_reduce:
        body, block_size = _emit_reduce_body(kernel, out_shape)
    else:
        body, block_size = _emit_pointwise_body(kernel, out_shape)

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
    """Emit code to evaluate ``inp`` at the linear output coord ``coord``.

    Appends any required ``VarDecl``s to ``stmts`` (for Combine chains)
    and returns an ``Expr`` that yields the value of ``inp`` at ``coord``.
    ``name_seq`` is a mutable counter used to generate fresh temporaries.
    """
    if isinstance(inp, Port):
        if inp.indexmap is not None:
            src_shape = shapes.get(inp.buffer_id, ())
            return _emit_indexmap_load(inp, coord, src_shape, stmts, name_seq)
        return ArrayAccess(array=inp.buffer_id, index=coord)

    if isinstance(inp, Mux):
        # Nested ternary: (cond0 ? branch0 : (cond1 ? branch1 : ...))
        result: Expr | None = None
        for branch in reversed(inp.branches):
            sub = _emit_input_value(branch.input, coord, shapes, stmts, name_seq)
            result = sub if result is None else Ternary(cond=branch.select, then=sub, else_=result)
        assert result is not None
        return result

    if isinstance(inp, Combine):
        # Load each source into a temporary.
        source_vals: list[Expr] = []
        for src in inp.sources:
            expr = _emit_input_value(src, coord, shapes, stmts, name_seq)
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=expr))
            source_vals.append(Var(tname))

        # Stack-machine: first op gets sources[0] (and sources[1] if
        # binary); each subsequent op gets the prior result as arg[0]
        # and the next unused source as arg[1].
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
    """Emit a load from ``port.buffer_id`` at the coord transformed by ``port.indexmap``.

    ``src_shape`` is the actual shape of the source buffer, used for
    computing flat input strides.
    """
    from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX, substitute

    indexmap = port.indexmap
    assert indexmap is not None and len(indexmap.sources) == 1
    src = indexmap.sources[0]
    out_shape = indexmap.out_shape
    ndim = len(out_shape)

    # Decompose flat coord into per-axis coords: coord_i = (coord / stride_i) % dim_i
    mapping: dict[str, Expr] = {}
    remainder = coord
    for d in range(ndim - 1, -1, -1):
        dim_size = int(out_shape[d])
        axis_var = f"{PLACEHOLDER_PREFIX}{d}"
        if d == 0:
            mapping[axis_var] = remainder
        else:
            mapping[axis_var] = BinOp("%", remainder, Literal(dim_size, "int"))
            remainder = BinOp("/", remainder, Literal(dim_size, "int"))

    # Apply coord_map substitution to get input coords.
    input_coords = [substitute(cm, mapping) for cm in src.coord_map]

    # Flatten input coords using the SOURCE buffer's strides (not the IndexMap output shape).
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
# Body emitters per kernel shape
# ---------------------------------------------------------------------------


def _emit_pointwise_body(kernel: KernelOp, out_shape: tuple) -> tuple[list[Stmt], tuple[int, int, int]]:
    numel = _numel(out_shape)
    idx = Var("idx")
    stmts: list[Stmt] = [
        VarDecl(
            dtype="int",
            name="idx",
            init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
        ),
    ]
    guarded: list[Stmt] = []
    name_seq = [0]

    # Load input Port/tree values. First input is the initial value;
    # remaining inputs are extras for binary epilogue ops.
    input_vals = [_emit_input_value(inp, idx, kernel.external_shapes, guarded, name_seq) for inp in kernel.inputs]
    value = input_vals[0] if input_vals else Literal(0.0, "float")
    extra_iter = iter(input_vals[1:])

    value = _apply_chain(kernel.body, value, extra_iter, idx, guarded, name_seq)

    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port), "naive pointwise: output must be a plain Port"
    guarded.append(Assign(target=ArrayAccess(array=out_port.buffer_id, index=idx), value=value))

    stmts.append(
        IfStmt(
            cond=BinOp("<", idx, Literal(numel, "int")),
            body=guarded,
        )
    )
    return stmts, (_BLOCK, 1, 1)


def _emit_reduce_body(kernel: KernelOp, out_shape: tuple) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Walk kernel.body (mixed ElementwiseOp / ReduceOp chain).

    Each ReduceOp emits a K-loop with an accumulator. ElementwiseOps
    between reduces apply to the current value (inside or outside the
    K-loop depending on position).
    """
    assert len(kernel.inputs) >= 1 and isinstance(kernel.inputs[0], Port)
    src_port = kernel.inputs[0]
    src_shape = kernel.external_shapes.get(src_port.buffer_id, ())
    k_size = src_shape[-1] if src_shape else 1

    row = Var("row")
    stmts: list[Stmt] = [
        VarDecl(dtype="int", name="row", init=Var("blockIdx.x")),
        IfStmt(
            cond=BinOp("!=", Var("threadIdx.x"), Literal(0, "int")),
            body=[RawCode("return;")],
        ),
    ]
    name_seq = [0]
    reduce_count = 0
    value: Expr | None = None

    for op in kernel.body:
        if isinstance(op, ReduceOp):
            fn = op.fn
            acc_name = f"acc{reduce_count}"
            stmts.append(VarDecl(dtype="float", name=acc_name, init=Literal(_identity(fn), "float")))
            k_var = Var(f"k{reduce_count}")
            if value is None:
                load_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), k_var)
                elem: Expr = ArrayAccess(array=src_port.buffer_id, index=load_idx)
            else:
                elem = value
            stmts.append(
                ForLoop(
                    var=k_var.name,
                    start=Literal(0, "int"),
                    end=Literal(k_size, "int"),
                    body=[AugAssign(target=acc_name, op=_reduce_op(fn), value=elem)],
                )
            )
            value = Var(acc_name)
            reduce_count += 1
        elif isinstance(op, ElementwiseOp):
            if value is None:
                load_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), Literal(0, "int"))
                value = ArrayAccess(array=src_port.buffer_id, index=load_idx)
            if op.info.arity == 1:
                value = _apply_elementwise(op.fn, [value])
            else:
                extra_vals = [_emit_input_value(inp, row, kernel.external_shapes, stmts, name_seq) for inp in kernel.inputs[1:]]
                extra = extra_vals[0] if extra_vals else Literal(0.0, "float")
                value = _apply_elementwise(op.fn, [value, extra])
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=value))
            value = Var(tname)

    assert value is not None
    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(Assign(target=ArrayAccess(array=out_port.buffer_id, index=row), value=value))
    return stmts, (1, 1, 1)


def _emit_contraction_body(kernel: KernelOp, out_shape: tuple) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Naive matmul with batch support.

    Output shape ``(...batch, M, N)``. Grid: (N, M, batch_size).
    Each thread computes one output element by iterating K serially.
    """
    contraction = kernel.contraction
    assert contraction is not None
    operand = contraction.operand

    k_size = _contraction_k_size(operand, kernel.external_shapes)

    # Parse output shape: (...batch, M, N).
    int_shape = tuple(int(d) for d in out_shape if isinstance(d, int))
    if len(int_shape) < 2:
        m_size, n_size, batch_size = 1, _numel(int_shape), 1
    else:
        m_size = int_shape[-2]
        n_size = int_shape[-1]
        batch_size = _numel(int_shape[:-2]) if len(int_shape) > 2 else 1

    stmts: list[Stmt] = []
    m = Var("m")
    n = Var("n")
    stmts.append(VarDecl(dtype="int", name="m", init=Var("blockIdx.y")))
    stmts.append(VarDecl(dtype="int", name="n", init=Var("blockIdx.x")))
    stmts.append(
        IfStmt(
            cond=BinOp("!=", Var("threadIdx.x"), Literal(0, "int")),
            body=[RawCode("return;")],
        )
    )

    # Batch: blockIdx.z indexes into the batch.
    if batch_size > 1:
        stmts.append(VarDecl(dtype="int", name="batch", init=Var("blockIdx.z")))
        batch = Var("batch")
    else:
        batch = None

    fn = contraction.reduce.fn
    stmts.append(VarDecl(dtype="float", name="acc", init=Literal(_identity(fn), "float")))

    k = Var("k")
    assert isinstance(operand, Combine), "naive contraction expects operand = Combine(a, b) with mul"
    assert len(operand.sources) == 2
    a_port, b_port = operand.sources
    assert isinstance(a_port, Port) and isinstance(b_port, Port)

    # Each operand has its own index space: A[..., m, k] and B[..., k, n].
    a_shape = kernel.external_shapes.get(a_port.buffer_id, ())
    b_shape = kernel.external_shapes.get(b_port.buffer_id, ())
    a_k = int(a_shape[-1]) if a_shape else k_size
    b_n = int(b_shape[-1]) if b_shape else n_size

    a_flat: Expr = BinOp("+", BinOp("*", m, Literal(a_k, "int")), k)
    b_flat: Expr = BinOp("+", BinOp("*", k, Literal(b_n, "int")), n)
    if batch is not None:
        a_flat = BinOp("+", BinOp("*", batch, Literal(m_size * a_k, "int")), a_flat)
        b_flat = BinOp("+", BinOp("*", batch, Literal(k_size * b_n, "int")), b_flat)

    name_seq_k = [0]
    inner_stmts: list[Stmt] = []
    a_val = _emit_input_value(a_port, a_flat, kernel.external_shapes, inner_stmts, name_seq_k)
    b_val = _emit_input_value(b_port, b_flat, kernel.external_shapes, inner_stmts, name_seq_k)
    assert len(operand.ops) == 1 and operand.ops[0].fn == "mul"
    prod_expr = BinOp("*", a_val, b_val)
    inner_stmts.append(AugAssign(target="acc", op=_reduce_op(fn), value=prod_expr))
    stmts.append(ForLoop(var=k.name, start=Literal(0, "int"), end=Literal(k_size, "int"), body=inner_stmts))

    name_seq = [0]
    value: Expr = Var("acc")
    out_flat: Expr = BinOp("+", BinOp("*", m, Literal(n_size, "int")), n)
    if batch is not None:
        out_flat = BinOp("+", BinOp("*", batch, Literal(m_size * n_size, "int")), out_flat)
    value = _apply_epilogue(kernel.body, value, list(kernel.inputs), out_flat, stmts, name_seq, kernel.external_shapes)
    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(Assign(target=ArrayAccess(array=out_port.buffer_id, index=out_flat), value=value))
    return stmts, (1, 1, 1)


# ---------------------------------------------------------------------------
# Epilogue chain (kernel-level elementwise after body)
# ---------------------------------------------------------------------------


def _apply_chain(
    chain: tuple[ElementwiseOp, ...],
    value: Expr,
    extra_iter: iter,
    coord: Expr,
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Apply an elementwise chain using positional (stack-machine) convention.

    ``value`` is the initial input (prior chain value or body output).
    Binary ops consume ``value`` as arg[0] and the next element from
    ``extra_iter`` as arg[1].
    """
    for op in chain:
        assert isinstance(op, ElementwiseOp)
        if op.info.arity == 1:
            value = _apply_elementwise(op.fn, [value])
        else:
            extra = next(extra_iter, ArrayAccess(array="?", index=coord))
            value = _apply_elementwise(op.fn, [value, extra])
        tname = _fresh(name_seq)
        stmts.append(VarDecl(dtype="float", name=tname, init=value))
        value = Var(tname)
    return value


def _apply_epilogue(
    chain: tuple[ElementwiseOp, ...],
    value: Expr,
    extra_inputs: list,
    coord: Expr,
    stmts: list[Stmt],
    name_seq: list[int],
    shapes: dict | None = None,
) -> Expr:
    """Apply epilogue after a contraction/reduce body output.

    ``extra_inputs`` are KernelInput trees for binary epilogue ops
    (e.g., bias Port for bias-add). Each binary op consumes the next
    extra input loaded at ``coord``.
    """
    if not chain:
        return value
    extra_vals = [_emit_input_value(inp, coord, shapes or {}, stmts, name_seq) for inp in extra_inputs]
    return _apply_chain(chain, value, iter(extra_vals), coord, stmts, name_seq)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SUPPORTED_UNARY = {
    "exp": "expf",
    "neg": None,  # -x via unary minus below
    "rsqrt": "rsqrtf",
    "recip": None,  # 1 / x
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
        # 1 / (1 + expf(-x))
        neg_x = BinOp("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCall("expf", [neg_x])
        return BinOp("/", Literal(1.0, "float"), BinOp("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY:
        callee = _SUPPORTED_UNARY[fn]
        if callee is not None:
            return FuncCall(callee, list(inputs))
    raise NotImplementedError(f"elementwise fn={fn} not yet supported by emit")


def _identity(reduce_fn: str) -> float:
    return {"sum": 0.0, "max": -1e30, "prod": 1.0}.get(reduce_fn, 0.0)


def _reduce_op(reduce_fn: str) -> str:
    return {"sum": "+=", "prod": "*="}.get(reduce_fn, "+=")


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _contraction_k_size(operand: KernelInput, shapes: dict[str, tuple]) -> int:
    if isinstance(operand, Combine) and operand.sources:
        first = operand.sources[0]
        if isinstance(first, Port):
            shape = shapes.get(first.buffer_id, ())
            if shape:
                return int(shape[-1])
    return 1


def _infer_output_shape(kernel: KernelOp) -> tuple:
    """Derive the kernel's output shape by walking the pipeline.

    Shape propagates: input shapes → [contraction reduce] →
    [reduce_stages] → [epilogue] → output shape.
    """
    shapes = kernel.external_shapes

    # Start from input shapes.
    input_shapes = [_port_shape(inp, shapes) for inp in kernel.inputs]
    if not input_shapes:
        input_shapes = [_port_shape(p, shapes) for p in _iter_leaf_ports(kernel)]

    # Contraction output shape: A(..., M, K) reduced over K → (..., M, N).
    if kernel.contraction is not None:
        operand = kernel.contraction.operand
        src_shapes = [_port_shape(s, shapes) for s in (operand.sources if isinstance(operand, Combine) else [operand])]
        if len(src_shapes) >= 2:
            a_shape, b_shape = src_shapes[0], src_shapes[1]
            mul_shape = tuple(a_shape) + (b_shape[-1],) if b_shape else tuple(a_shape)
        else:
            mul_shape = src_shapes[0] if src_shapes else ()
        shape = kernel.contraction.reduce.infer_output_shape([mul_shape])
    elif input_shapes:
        shape = input_shapes[0]
    else:
        return ()

    # Body chain (mixed elementwise + reduce).
    for op in kernel.body:
        shape = op.infer_output_shape([shape])

    return tuple(shape)


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


def _iter_leaf_ports(kernel: KernelOp):
    """Yield all leaf Ports from the kernel's trees."""
    if kernel.contraction is not None:
        yield from _leaf_ports_from(kernel.contraction.operand)
    for inp in kernel.inputs:
        yield from _leaf_ports_from(inp)


def _leaf_ports_from(inp):
    if isinstance(inp, Port):
        yield inp
    elif isinstance(inp, Mux):
        for b in inp.branches:
            yield from _leaf_ports_from(b.input)
    elif isinstance(inp, Combine):
        for s in inp.sources:
            yield from _leaf_ports_from(s)


def _build_params(kernel: KernelOp) -> tuple[list[KernelParam], list[str]]:
    """Determine kernel params + launch arg order.

    Order: all external input Port buffers (deduped), then output Port
    buffers. Launch args list the same names as string buffer ids.
    """
    seen: list[str] = []
    inputs_refs = _leaf_ports(kernel)
    for p in inputs_refs:
        if p.buffer_id not in seen and p.buffer_id not in {o.buffer_id for o in kernel.outputs if isinstance(o, Port)}:
            seen.append(p.buffer_id)
    output_ids = [o.buffer_id for o in kernel.outputs if isinstance(o, Port)]
    params = [KernelParam(dtype="const float*", name=bid) for bid in seen]
    params += [KernelParam(dtype="float*", name=bid) for bid in output_ids]
    args = seen + output_ids
    return params, args


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
    if kernel.contraction is not None:
        walk(kernel.contraction.operand)
    return leaves


def _has_reduce(kernel: KernelOp) -> bool:
    return any(isinstance(op, ReduceOp) for op in kernel.body)


def _kernel_name(kernel: KernelOp, idx: int) -> str:
    if kernel.contraction is not None:
        return f"k{idx}_contract"
    if _has_reduce(kernel):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    out_shape = _infer_output_shape(kernel)
    if kernel.contraction is not None:
        int_shape = tuple(int(d) for d in out_shape if isinstance(d, int))
        m = int_shape[-2] if len(int_shape) >= 2 else 1
        n = int_shape[-1] if int_shape else 1
        batch = _numel(int_shape[:-2]) if len(int_shape) > 2 else 1
        return (n, m, batch), (1, 1, 1)
    if _has_reduce(kernel):
        n_rows = _numel(out_shape)
        return (n_rows, 1, 1), (1, 1, 1)
    numel = _numel(out_shape)
    n_blocks = (numel + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def _emit_kernel_source(kernel_def: KernelDef) -> str:
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel as _emit

    return _emit(kernel_def)


def _fresh(seq: list[int]) -> str:
    name = f"t{seq[0]}"
    seq[0] += 1
    return name
