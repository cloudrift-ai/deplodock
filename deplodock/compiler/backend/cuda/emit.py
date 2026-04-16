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
) -> Program:
    """Assemble a ``Program`` from a list of structural ``KernelOp``.

    ``graph_inputs`` / ``graph_outputs`` are buffer-id sets used to mark
    buffer roles; anything produced by one kernel and consumed by another
    is "scratch".
    """
    graph_input_set = set(graph_inputs or [])
    graph_output_set = set(graph_outputs or [])

    # Discover every external buffer + its shape.
    buf_shapes: dict[str, tuple] = {}
    for k in kernels:
        for bid, shape in k.external_shapes.items():
            buf_shapes[bid] = tuple(shape)
        for out in k.outputs:
            if isinstance(out, Port):
                buf_shapes.setdefault(out.buffer_id, _infer_output_shape(k))

    # Assign roles: inputs override scratch; outputs override scratch.
    def role_for(bid: str) -> str:
        if bid in graph_input_set:
            return "input"
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

    if kernel.contraction is not None:
        body, block_size = _emit_contraction_body(kernel, out_shape)
    elif kernel.reduce_stages:
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
        # Evaluate each source, bind to a named temporary, then fold the
        # ops chain top-down substituting source results by buffer_id.
        source_exprs: dict[str, Expr] = {}
        for src in inp.sources:
            expr = _emit_input_value(src, coord, shapes, stmts, name_seq)
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=expr))
            # Each Combine source is rooted at some buffer; expose its tmp
            # name by the same id so downstream ops can reference it.
            source_exprs[_source_id(src)] = Var(tname)

        # Apply the elementwise chain. Each node's inputs are buffer_ids
        # referenced either from source_exprs or from prior ops' outputs
        # (keyed by node.id).
        value_by_id: dict[str, Expr] = dict(source_exprs)
        last: Expr | None = None
        for node in inp.ops:
            assert isinstance(node.op, ElementwiseOp)
            node_inputs = [value_by_id.get(nid, Var(nid)) for nid in node.inputs]
            value = _apply_elementwise(node.op.fn, node_inputs)
            tname = _fresh(name_seq)
            stmts.append(VarDecl(dtype="float", name=tname, init=value))
            value_by_id[node.id] = Var(tname)
            last = Var(tname)
        assert last is not None
        return last

    raise NotImplementedError(f"unexpected KernelInput variant: {type(inp).__name__}")


def _source_id(src: KernelInput) -> str:
    """Return a buffer-id-like key for a KernelInput, so downstream ops can reference it."""
    if isinstance(src, Port):
        return src.buffer_id
    if isinstance(src, Combine):
        # The combine's last op's id identifies its output value.
        if src.ops:
            return src.ops[-1].id
    return f"<anon_{id(src)}>"


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

    # Load each input Port/tree → build value map keyed by buffer_id.
    value_map: dict[str, Expr] = {}
    for inp in kernel.inputs:
        expr = _emit_input_value(inp, idx, kernel.external_shapes, guarded, name_seq)
        value_map[_source_id(inp)] = expr

    # Apply the epilogue chain, resolving each op's inputs from value_map
    # or from prior ops.
    value = _apply_chain(kernel.epilogue, value_map, idx, guarded, name_seq)

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
    """Single-thread-per-block naive reduction: one block per output row."""
    stages = kernel.reduce_stages
    assert len(stages) >= 1
    # First stage reads from the first kernel input (assumed a Port).
    assert len(kernel.inputs) == 1 and isinstance(kernel.inputs[0], Port)
    src_port = kernel.inputs[0]
    src_shape = kernel.external_shapes.get(src_port.buffer_id, ())
    k_size = src_shape[-1] if src_shape else 1
    # Output coord = row index along leading axes (flattened).
    row = Var("row")
    stmts: list[Stmt] = [
        VarDecl(dtype="int", name="row", init=Var("blockIdx.x")),
        IfStmt(
            cond=BinOp("!=", Var("threadIdx.x"), Literal(0, "int")),
            body=[RawCode("return;")],
        ),
    ]
    # Run stages in sequence. For the very first stage, the input is a
    # straight reduction over ``src_port`` along its last axis. Subsequent
    # stages operate on per-row accumulators (pre_ops) and reduce again.
    prev_value_expr: Expr | None = None
    name_seq = [0]
    for si, stage in enumerate(stages):
        fn = stage.reduce.op.fn
        acc_name = f"acc{si}"
        stmts.append(VarDecl(dtype="float", name=acc_name, init=Literal(_identity(fn), "float")))
        k_var = Var(f"k{si}")
        inner: list[Stmt] = []
        if si == 0:
            # Load element from src_port at (row * K + k).
            load_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), k_var)
            elem_expr: Expr = ArrayAccess(array=src_port.buffer_id, index=load_idx)
        else:
            # Feed pre_ops starting from prev_value_expr (unused for naive;
            # pre_ops operate on per-row accumulators, but we only emit a
            # single-row-per-block path). For a future per-block parallel
            # reduce this would be a shared-mem chain.
            elem_expr = prev_value_expr  # type: ignore[assignment]
            for pre in stage.pre_ops:
                assert isinstance(pre.op, ElementwiseOp)
                elem_expr = _apply_elementwise(pre.op.fn, [elem_expr] + [Literal(0.0, "float")] * (pre.op.info.arity - 1))
        inner.append(AugAssign(target=acc_name, op=_reduce_op(fn), value=elem_expr))
        stmts.append(
            ForLoop(
                var=k_var.name,
                start=Literal(0, "int"),
                end=Literal(k_size, "int"),
                body=inner,
            )
        )
        prev_value_expr = Var(acc_name)

    value: Expr = prev_value_expr  # type: ignore[assignment]
    last_reduce_id = stages[-1].reduce.id
    value = _apply_epilogue(kernel.epilogue, value, last_reduce_id, row, stmts, name_seq)
    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(Assign(target=ArrayAccess(array=out_port.buffer_id, index=row), value=value))
    return stmts, (1, 1, 1)


def _emit_contraction_body(kernel: KernelOp, out_shape: tuple) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Naive matmul: 2D grid over (M, N), one thread per output element."""
    contraction = kernel.contraction
    assert contraction is not None
    operand = contraction.operand
    # Determine K: look at any Port inside operand; assume the two sources
    # share the K dimension along the last axis (matmul convention).
    k_size = _contraction_k_size(operand, kernel.external_shapes)
    # Output is 2D: (M, N). For our singleton matmul pattern, M*N = numel.
    n_size = int(out_shape[-1]) if out_shape else 1

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

    fn = contraction.reduce.op.fn
    stmts.append(VarDecl(dtype="float", name="acc", init=Literal(_identity(fn), "float")))

    k = Var("k")
    # For naive matmul mul(a,b) → operand is a Combine with two sources
    # (a and b). Compute the per-K product by walking the tree at coord
    # = m*K + k for the A-operand and k*N + n for the B-operand. We
    # special-case this shape because the generic tree walker can't know
    # the matmul layout convention.
    assert isinstance(operand, Combine), "naive contraction expects operand = Combine(a, b) with mul"
    assert len(operand.sources) == 2
    a_port, b_port = operand.sources
    assert isinstance(a_port, Port) and isinstance(b_port, Port)
    a_idx = BinOp("+", BinOp("*", m, Literal(k_size, "int")), k)
    b_idx = BinOp("+", BinOp("*", k, Literal(n_size, "int")), n)
    a_load = ArrayAccess(array=a_port.buffer_id, index=a_idx)
    b_load = ArrayAccess(array=b_port.buffer_id, index=b_idx)
    # Apply the operand.ops chain (expected: [mul]).
    assert len(operand.ops) == 1 and operand.ops[0].op.fn == "mul"
    prod_expr = BinOp("*", a_load, b_load)

    stmts.append(
        ForLoop(
            var=k.name,
            start=Literal(0, "int"),
            end=Literal(k_size, "int"),
            body=[AugAssign(target="acc", op=_reduce_op(fn), value=prod_expr)],
        )
    )

    name_seq = [0]
    # Post-contraction reduce_stages + epilogue at (m, n). For the naive
    # singleton matmul, both are typically empty.
    value: Expr = Var("acc")
    contraction_reduce_id = contraction.reduce.id
    out_coord = BinOp("+", BinOp("*", m, Literal(n_size, "int")), n)
    value = _apply_epilogue(kernel.epilogue, value, contraction_reduce_id, out_coord, stmts, name_seq)
    out_port = kernel.outputs[0]
    assert isinstance(out_port, Port)
    stmts.append(
        Assign(
            target=ArrayAccess(
                array=out_port.buffer_id,
                index=BinOp("+", BinOp("*", m, Literal(n_size, "int")), n),
            ),
            value=value,
        )
    )
    return stmts, (1, 1, 1)


# ---------------------------------------------------------------------------
# Epilogue chain (kernel-level elementwise after body)
# ---------------------------------------------------------------------------


def _apply_chain(
    chain: tuple,
    value_map: dict[str, Expr],
    coord: Expr,
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Apply an elementwise chain, resolving each op's inputs from ``value_map``
    (prior ops + external Ports) or by loading from the buffer at ``coord``."""
    last: Expr | None = None
    for node in chain:
        assert isinstance(node.op, ElementwiseOp)
        node_inputs = [value_map.get(nid, ArrayAccess(array=nid, index=coord)) for nid in node.inputs]
        applied = _apply_elementwise(node.op.fn, node_inputs)
        tname = _fresh(name_seq)
        stmts.append(VarDecl(dtype="float", name=tname, init=applied))
        value_map[node.id] = Var(tname)
        last = Var(tname)
    return last if last is not None else Literal(0.0, "float")


def _apply_epilogue(
    chain: tuple,
    value: Expr,
    body_node_id: str,
    coord: Expr,
    stmts: list[Stmt],
    name_seq: list[int],
) -> Expr:
    """Apply epilogue after a contraction/reduce body output.

    Returns ``value`` unchanged if the chain is empty.
    """
    if not chain:
        return value
    value_map: dict[str, Expr] = {body_node_id: value}
    return _apply_chain(chain, value_map, coord, stmts, name_seq)


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
    # Priority: epilogue last node > last reduce > contraction > body input.
    if kernel.epilogue:
        return tuple(kernel.epilogue[-1].output.shape)
    if kernel.reduce_stages:
        return tuple(kernel.reduce_stages[-1].reduce.output.shape)
    if kernel.contraction is not None:
        return tuple(kernel.contraction.reduce.output.shape)
    if kernel.inputs and isinstance(kernel.inputs[0], Combine):
        return tuple(kernel.inputs[0].ops[-1].output.shape)
    return ()


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


def _kernel_name(kernel: KernelOp, idx: int) -> str:
    if kernel.contraction is not None:
        return f"k{idx}_contract"
    if kernel.reduce_stages:
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    out_shape = _infer_output_shape(kernel)
    if kernel.contraction is not None:
        m = int(out_shape[-2]) if len(out_shape) >= 2 else 1
        n = int(out_shape[-1]) if out_shape else 1
        return (n, m, 1), (1, 1, 1)
    if kernel.reduce_stages:
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
