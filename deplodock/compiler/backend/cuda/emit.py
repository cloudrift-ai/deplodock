"""Naive recursive-descent CUDA codegen from structural ``KernelOp``.

Walks the structural IR (Port | Mux | Combine inputs, SSA body of
``Assign`` statements, outputs) and emits one ``KernelDef`` per
``KernelOp`` directly -- no classification pass, no Schedule, no
LoopIR intermediate.

Contraction (matmul) is detected by pattern-matching the SSA body:
a binary ``ElementwiseOp`` whose **both** args are input Port names,
followed by a ``ReduceOp`` consuming it.  When found, the emitter
produces a 2D grid with a serial K-loop.  Otherwise:

- **Pointwise** (no ReduceOp in body): 1D grid over the flattened
  output numel; one thread per output coord; block size fixed at 256.
- **Reduce chain** (ReduceOp in body, no contraction pattern): 1D grid
  over the post-reduce outer shape; one block per reduced row; one
  thread per block.

All shapes are treated as flat buffers: each input tensor has a known
shape (``kernel.external_shapes[buffer_id]``) and is indexed by a
single linearized offset.
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
    RawCode,
    Stmt,
    VarDecl,
)
from deplodock.compiler.backend.ir.kernel_ir import (
    Assign as IrAssign,
)
from deplodock.compiler.backend.program import Buffer, Program
from deplodock.compiler.ops import (
    Assign,
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
# Contraction detection: pattern-match SSA body
# ---------------------------------------------------------------------------


def _detect_contraction(kernel: KernelOp) -> tuple[int, Assign, Assign] | None:
    """Look for a binary ElementwiseOp whose BOTH args are input Port names,
    followed by a ReduceOp consuming it.  Returns (index, mul_assign, red_assign)
    or None.
    """
    input_names = {p.buffer_id for p in kernel.inputs if isinstance(p, Port)}
    for i, assign in enumerate(kernel.body):
        if isinstance(assign.op, ElementwiseOp) and len(assign.args) == 2 and all(a in input_names for a in assign.args):
            if i + 1 < len(kernel.body):
                next_a = kernel.body[i + 1]
                if isinstance(next_a.op, ReduceOp) and assign.name in next_a.args:
                    return i, assign, next_a
    return None


# ---------------------------------------------------------------------------
# Per-kernel emission
# ---------------------------------------------------------------------------


def emit_kernel(kernel: KernelOp, kernel_name: str) -> tuple[KernelDef, list[str]]:
    """Emit a single ``KernelOp`` as a ``KernelDef`` + ordered launch args."""
    out_shape = _infer_output_shape(kernel)
    params, arg_order = _build_params(kernel)

    contraction = _detect_contraction(kernel)
    if contraction is not None:
        body, block_size = _emit_contraction_body(kernel, out_shape, contraction)
    elif any(isinstance(a.op, ReduceOp) for a in kernel.body):
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
# Unified body emitter: walks the SSA Assign sequence
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

    # Build values dict: load all input Ports, then walk body Assigns.
    values: dict[str, Expr] = {}
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            values[inp.buffer_id] = _emit_input_value(inp, idx, kernel.external_shapes, guarded, name_seq)

    for assign in kernel.body:
        arg_exprs = [values[a] for a in assign.args]
        if isinstance(assign.op, ElementwiseOp):
            value = _apply_elementwise(assign.op.fn, arg_exprs)
        else:
            raise NotImplementedError(f"pointwise body: unexpected op {type(assign.op).__name__}")
        tname = _fresh(name_seq)
        guarded.append(VarDecl(dtype="float", name=tname, init=value))
        values[assign.name] = Var(tname)

    # Store last Assign's value to the output.
    last_name = kernel.body[-1].name if kernel.body else None
    out_val = values.get(last_name, Literal(0.0, "float")) if last_name else Literal(0.0, "float")

    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port), "naive pointwise: output must be a plain Port"
    guarded.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=idx), value=out_val))

    stmts.append(
        IfStmt(
            cond=BinOp("<", idx, Literal(numel, "int")),
            body=guarded,
        )
    )
    return stmts, (_BLOCK, 1, 1)


def _emit_reduce_body(kernel: KernelOp, out_shape: tuple) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Walk kernel.body SSA (mixed ElementwiseOp / ReduceOp Assigns).

    Each ReduceOp emits a K-loop with an accumulator. ElementwiseOps
    apply inline to the current value.
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

    # Build values dict for SSA names.
    values: dict[str, Expr] = {}
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            values[inp.buffer_id] = None  # placeholder; loaded inside loops

    for assign in kernel.body:
        if isinstance(assign.op, ReduceOp):
            fn = assign.op.fn
            acc_name = f"acc{reduce_count}"
            stmts.append(VarDecl(dtype="float", name=acc_name, init=Literal(_identity(fn), "float")))
            k_var = Var(f"k{reduce_count}")

            # The arg to reduce references a prior Assign or an input Port.
            assert len(assign.args) == 1
            src_name = assign.args[0]

            # If the source is a raw input Port, load it inside the K-loop.
            inner_stmts: list[Stmt] = []
            if src_name in {p.buffer_id for p in kernel.inputs if isinstance(p, Port)} and values.get(src_name) is None:
                load_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), k_var)
                elem: Expr = ArrayAccess(array=src_name, index=load_idx)
            else:
                # Source is a prior Assign; its expression is already in values.
                elem = values[src_name]

            inner_stmts.append(AugAssign(target=acc_name, op=_reduce_op(fn), value=elem))
            stmts.append(
                ForLoop(
                    var=k_var.name,
                    start=Literal(0, "int"),
                    end=Literal(k_size, "int"),
                    body=inner_stmts,
                )
            )
            values[assign.name] = Var(acc_name)
            reduce_count += 1

        elif isinstance(assign.op, ElementwiseOp):
            arg_exprs = []
            for a in assign.args:
                if a in values and values[a] is not None:
                    arg_exprs.append(values[a])
                else:
                    # Load extra inputs at the row coord.
                    extra_vals = [_emit_input_value(inp, row, kernel.external_shapes, stmts, name_seq) for inp in kernel.inputs[1:]]
                    arg_exprs.append(extra_vals[0] if extra_vals else Literal(0.0, "float"))
            value = _apply_elementwise(assign.op.fn, arg_exprs)
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=value))
            values[assign.name] = Var(tname)

    # Store the last Assign's value.
    last_name = kernel.body[-1].name if kernel.body else None
    out_val = values.get(last_name, Literal(0.0, "float")) if last_name else Literal(0.0, "float")

    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=row), value=out_val))
    return stmts, (1, 1, 1)


def _emit_contraction_body(
    kernel: KernelOp,
    out_shape: tuple,
    contraction: tuple[int, Assign, Assign],
) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Emit a contraction (matmul) body with batch support.

    Output shape ``(...batch, M, N)``. Grid: (N, M, batch_size).
    Each thread computes one output element by iterating K serially.
    Post-contraction Assigns (epilogue) are emitted as flat elementwise.
    """
    con_idx, mul_assign, red_assign = contraction

    # Find the a and b Ports from the mul Assign's args.
    a_name, b_name = mul_assign.args
    a_port = next(p for p in kernel.inputs if isinstance(p, Port) and p.buffer_id == a_name)
    b_port = next(p for p in kernel.inputs if isinstance(p, Port) and p.buffer_id == b_name)

    a_shape = kernel.external_shapes.get(a_port.buffer_id, ())
    b_shape = kernel.external_shapes.get(b_port.buffer_id, ())
    k_size = int(a_shape[-1]) if a_shape else 1

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

    fn = red_assign.op.fn
    stmts.append(VarDecl(dtype="float", name="acc", init=Literal(_identity(fn), "float")))

    k = Var("k")
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
    prod_expr = _apply_elementwise(mul_assign.op.fn, [a_val, b_val])
    inner_stmts.append(AugAssign(target="acc", op=_reduce_op(fn), value=prod_expr))
    stmts.append(ForLoop(var=k.name, start=Literal(0, "int"), end=Literal(k_size, "int"), body=inner_stmts))

    # Build values dict with contraction output, then walk post-contraction Assigns.
    values: dict[str, Expr] = {}
    values[red_assign.name] = Var("acc")

    # Also register input Port buffer_ids for any post-contraction refs.
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            values[inp.buffer_id] = None  # loaded on demand below

    name_seq = [0]
    out_flat: Expr = BinOp("+", BinOp("*", m, Literal(n_size, "int")), n)
    if batch is not None:
        out_flat = BinOp("+", BinOp("*", batch, Literal(m_size * n_size, "int")), out_flat)

    # Post-contraction body: everything after the reduce Assign.
    post_start = con_idx + 2  # skip the mul and reduce Assigns
    for assign in kernel.body[post_start:]:
        arg_exprs = []
        for a in assign.args:
            if a in values and values[a] is not None:
                arg_exprs.append(values[a])
            else:
                # Load extra input at the output coord.
                port = next((p for p in kernel.inputs if isinstance(p, Port) and p.buffer_id == a), None)
                if port is not None:
                    val = _emit_input_value(port, out_flat, kernel.external_shapes, stmts, name_seq)
                    arg_exprs.append(val)
                else:
                    arg_exprs.append(Literal(0.0, "float"))
        if isinstance(assign.op, ElementwiseOp):
            value = _apply_elementwise(assign.op.fn, arg_exprs)
        else:
            raise NotImplementedError(f"post-contraction: unexpected op {type(assign.op).__name__}")
        tname = _fresh(name_seq)
        stmts.append(VarDecl(dtype="float", name=tname, init=value))
        values[assign.name] = Var(tname)

    # Store the final value.
    last_name = kernel.body[-1].name if kernel.body else red_assign.name
    out_val = values.get(last_name, Var("acc"))

    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=out_flat), value=out_val))
    return stmts, (1, 1, 1)


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


def _infer_output_shape(kernel: KernelOp) -> tuple:
    """Derive the kernel's output shape by walking the SSA body.

    Tracks shapes per Assign name: starts from input Port shapes,
    then propagates through each Assign's op.infer_output_shape.

    For contractions (mul+reduce pattern), the mul's output shape is
    taken from the higher-rank operand (the one with the indexmap that
    broadcasts into the (M,K,N) space), bypassing broadcast_shapes.
    """
    shapes: dict[str, tuple] = {}
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            shapes[inp.buffer_id] = _port_shape(inp, kernel.external_shapes)
        elif isinstance(inp, Combine):
            for src in inp.sources:
                if isinstance(src, Port):
                    shapes[src.buffer_id] = _port_shape(src, kernel.external_shapes)
        elif isinstance(inp, Mux):
            for branch in inp.branches:
                if isinstance(branch.input, Port):
                    shapes[branch.input.buffer_id] = _port_shape(branch.input, kernel.external_shapes)

    # Detect contraction to handle mul shape specially.
    contraction = _detect_contraction(kernel)
    contraction_names: set[str] = set()
    if contraction is not None:
        _, mul_assign, red_assign = contraction
        contraction_names = {mul_assign.name, red_assign.name}

    for assign in kernel.body:
        arg_shapes = [shapes[a] for a in assign.args if a in shapes]
        if not arg_shapes:
            continue
        if assign.name in contraction_names and isinstance(assign.op, ElementwiseOp):
            # Contraction mul: take the highest-rank arg shape (the
            # indexmap-expanded one) instead of broadcasting.
            shapes[assign.name] = max(arg_shapes, key=len)
        else:
            shapes[assign.name] = assign.op.infer_output_shape(arg_shapes)

    last = kernel.body[-1].name if kernel.body else None
    if last and last in shapes:
        return tuple(shapes[last])

    # Fallback: use input shapes directly (e.g., empty body = copy kernel).
    input_shapes = [_port_shape(inp, kernel.external_shapes) for inp in kernel.inputs]
    return tuple(input_shapes[0]) if input_shapes else ()


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


def _leaf_ports_from(inp):
    if isinstance(inp, Port):
        yield inp
    elif isinstance(inp, Mux):
        for b in inp.branches:
            yield from _leaf_ports_from(b.input)
    elif isinstance(inp, Combine):
        for s in inp.sources:
            yield from _leaf_ports_from(s)


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


def _kernel_name(kernel: KernelOp, idx: int) -> str:
    if _detect_contraction(kernel) is not None:
        return f"k{idx}_contract"
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    out_shape = _infer_output_shape(kernel)
    if _detect_contraction(kernel) is not None:
        int_shape = tuple(int(d) for d in out_shape if isinstance(d, int))
        m = int_shape[-2] if len(int_shape) >= 2 else 1
        n = int_shape[-1] if int_shape else 1
        batch = _numel(int_shape[:-2]) if len(int_shape) > 2 else 1
        return (n, m, batch), (1, 1, 1)
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
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
