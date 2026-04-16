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
shape (looked up from an externally-provided ``shapes`` dict keyed by
``buffer_id``) and is indexed by a single linearized offset.
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
        # Register input leaf Ports not yet in shapes (e.g., Ports with indexmap
        # whose buffer_id is the original source, already in buf_shapes).
        for port in _leaf_ports(k):
            if port.buffer_id not in shapes:
                if port.indexmap is not None:
                    # The source shape isn't known — use indexmap.out_shape
                    # as a size hint (conservative: at least this many elements).
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

    buffers = [Buffer(name=bid, size=_numel(shape), dtype="float", role=role_for(bid)) for bid, shape in shapes.items()]

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


def emit_kernel(kernel: KernelOp, kernel_name: str, shapes: dict[str, tuple]) -> tuple[KernelDef, list[str]]:
    """Emit a single ``KernelOp`` as a ``KernelDef`` + ordered launch args."""
    out_shape = kernel.infer_output_shape(shapes)
    params, arg_order = _build_params(kernel)

    contraction = _detect_contraction(kernel)
    if contraction is not None:
        body, block_size = _emit_contraction_body(kernel, out_shape, contraction, shapes)
    elif any(isinstance(a.op, ReduceOp) for a in kernel.body):
        body, block_size = _emit_reduce_body(kernel, out_shape, shapes)
    else:
        body, block_size = _emit_pointwise_body(kernel, out_shape, shapes)

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

    # Scan coord_map for the max placeholder index (composed IndexMaps may
    # reference higher dims than out_shape has).
    max_placeholder = ndim - 1
    for cm in src.coord_map:
        for var_name in _collect_var_names(cm):
            if var_name.startswith(PLACEHOLDER_PREFIX):
                idx = int(var_name[len(PLACEHOLDER_PREFIX) :])
                max_placeholder = max(max_placeholder, idx)

    # Decompose flat coord into per-axis coords for all needed placeholder dims.
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


def _emit_pointwise_body(kernel: KernelOp, out_shape: tuple, shapes: dict[str, tuple]) -> tuple[list[Stmt], tuple[int, int, int]]:
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
            values[inp.buffer_id] = _emit_input_value(inp, idx, shapes, guarded, name_seq)

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


def _emit_reduce_body(kernel: KernelOp, out_shape: tuple, shapes: dict[str, tuple]) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Walk kernel.body SSA (mixed ElementwiseOp / ReduceOp Assigns).

    The body is split into "segments" separated by ReduceOps. Each segment
    is a sequence of ElementwiseOps optionally followed by a ReduceOp:

        [elementwise*, reduce] or [elementwise*]  (trailing)

    For each segment, if any value references a per-element input (an input
    Port or a pre-reduce Assign), the entire segment is emitted inside a
    K-loop. Per-element values from *prior* segments (which were computed
    in their own K-loops) are recomputed from the inputs inside the current
    loop. Post-reduce accumulators (per-row scalars) are referenced directly.
    """
    assert len(kernel.inputs) >= 1 and isinstance(kernel.inputs[0], Port)
    src_port = kernel.inputs[0]
    src_shape = shapes.get(src_port.buffer_id, ())
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

    # Identify all input Port names (per-element iteration space).
    input_port_names: set[str] = {p.buffer_id for p in kernel.inputs if isinstance(p, Port)}

    # Track which SSA names are per-row (post-reduce scalars) vs per-element.
    per_row_values: set[str] = set()

    # Final values dict: maps SSA name -> Expr for per-row values only.
    # Per-element values are recomputed inside each K-loop.
    values: dict[str, Expr] = {}

    # Split body into segments: [elementwise*, reduce?]
    segments: list[list[Assign]] = []
    current_seg: list[Assign] = []
    for assign in kernel.body:
        current_seg.append(assign)
        if isinstance(assign.op, ReduceOp):
            segments.append(current_seg)
            current_seg = []
    if current_seg:
        segments.append(current_seg)

    for segment in segments:
        last = segment[-1]
        has_reduce = isinstance(last.op, ReduceOp)
        ew_assigns = segment[:-1] if has_reduce else segment
        reduce_assign = last if has_reduce else None

        # Check if any elementwise op in this segment needs per-element access.
        needs_k_loop = False
        for assign in ew_assigns:
            for a in assign.args:
                if a in input_port_names or a not in per_row_values:
                    needs_k_loop = True
                    break
            if needs_k_loop:
                break
        # A reduce always needs a K-loop (its input is per-element).
        if has_reduce:
            assert reduce_assign is not None
            for a in reduce_assign.args:
                if a in input_port_names or a not in per_row_values:
                    needs_k_loop = True

        if not needs_k_loop and not has_reduce:
            # All args are per-row scalars; emit inline.
            for assign in ew_assigns:
                arg_exprs = [values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                stmts.append(VarDecl(dtype="float", name=tname, init=value))
                values[assign.name] = Var(tname)
                per_row_values.add(assign.name)
        else:
            # Emit a K-loop for this segment.
            k_var_name = f"k{reduce_count}"
            k_var = Var(k_var_name)

            # Set up accumulator if segment ends with a reduce.
            if has_reduce:
                assert reduce_assign is not None
                fn = reduce_assign.op.fn
                acc_name = f"acc{reduce_count}"
                stmts.append(VarDecl(dtype="float", name=acc_name, init=Literal(_identity(fn), "float")))

            inner_stmts: list[Stmt] = []
            # Inside the loop, build per-element values by recomputing
            # from inputs through all prior elementwise chains.
            loop_values: dict[str, Expr] = dict(values)  # inherit per-row scalars

            # Load input Ports at per-element coords.
            load_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), k_var)
            for port_name in input_port_names:
                loop_values[port_name] = ArrayAccess(array=port_name, index=load_idx)

            # Recompute any prior per-element Assigns needed by this segment.
            # Collect all prior elementwise assigns (in body order).
            prior_assigns: list[Assign] = []
            for seg in segments:
                if seg is segment:
                    break
                for a in seg:
                    if not isinstance(a.op, ReduceOp):
                        prior_assigns.append(a)

            # Compute transitive closure of needed prior names.
            needed_names = _transitive_deps(segment, prior_assigns, per_row_values)

            for prior in prior_assigns:
                if prior.name in per_row_values:
                    continue
                if prior.name not in needed_names:
                    continue
                arg_exprs = [loop_values[a] for a in prior.args]
                value = _apply_elementwise(prior.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner_stmts.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[prior.name] = Var(tname)

            # Emit this segment's elementwise ops.
            for assign in ew_assigns:
                arg_exprs = [loop_values[a] for a in assign.args]
                value = _apply_elementwise(assign.op.fn, arg_exprs)
                tname = _fresh(name_seq)
                inner_stmts.append(VarDecl(dtype="float", name=tname, init=value))
                loop_values[assign.name] = Var(tname)

            if has_reduce:
                assert reduce_assign is not None
                fn = reduce_assign.op.fn
                reduce_src = loop_values[reduce_assign.args[0]]
                inner_stmts.append(_emit_reduce_accum(acc_name, fn, reduce_src))
                values[reduce_assign.name] = Var(acc_name)
                per_row_values.add(reduce_assign.name)
            else:
                # Trailing per-element segment: store inside the loop.
                last_ew = ew_assigns[-1]
                out_port = kernel.outputs[0]
                assert isinstance(out_port, Port)
                store_idx = BinOp("+", BinOp("*", row, Literal(k_size, "int")), k_var)
                inner_stmts.append(
                    IrAssign(
                        target=ArrayAccess(array=out_port.buffer_id, index=store_idx),
                        value=loop_values[last_ew.name],
                    )
                )

            stmts.append(
                ForLoop(
                    var=k_var_name,
                    start=Literal(0, "int"),
                    end=Literal(k_size, "int"),
                    body=inner_stmts,
                )
            )
            reduce_count += 1

    # Store the final value (only if the last segment was NOT a per-element
    # trailing segment, which already stored inside its K-loop).
    last_assign = kernel.body[-1] if kernel.body else None
    if last_assign is not None and last_assign.name in per_row_values:
        out_val = values.get(last_assign.name, Literal(0.0, "float"))
        out_port = kernel.outputs[0]
        assert isinstance(out_port, Port)
        stmts.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=row), value=out_val))
    elif last_assign is not None and not isinstance(last_assign.op, ReduceOp):
        # Trailing per-element segment already stored inside its K-loop.
        pass
    else:
        # Fallback: last was a reduce, store the accumulator.
        if last_assign is not None:
            out_val = values.get(last_assign.name, Literal(0.0, "float"))
            out_port = kernel.outputs[0]
            assert isinstance(out_port, Port)
            stmts.append(IrAssign(target=ArrayAccess(array=out_port.buffer_id, index=row), value=out_val))

    return stmts, (1, 1, 1)


def _transitive_deps(
    segment: list[Assign],
    prior_assigns: list[Assign],
    per_row_values: set[str],
) -> set[str]:
    """Compute the transitive closure of prior per-element values needed by segment.

    Walks backwards: starts with direct args of the segment's assigns, then
    expands to include any prior assign whose name is in the needed set.
    Per-row (post-reduce) values are excluded since they don't need recomputation.
    """
    needed: set[str] = set()
    for assign in segment:
        needed.update(assign.args)

    # Build a lookup from prior assign names to their args.
    prior_args: dict[str, tuple[str, ...]] = {a.name: a.args for a in prior_assigns}

    # Expand transitively.
    changed = True
    while changed:
        changed = False
        for name, args in prior_args.items():
            if name in needed and name not in per_row_values:
                for arg in args:
                    if arg not in needed and arg not in per_row_values:
                        needed.add(arg)
                        changed = True

    return needed


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    """Emit the accumulator update for a reduce op.

    For sum/prod, uses AugAssign (``acc += val``).
    For max, uses VarAssign with fmaxf (``acc = fmaxf(acc, val)``).
    """
    if fn == "max":
        from deplodock.compiler.backend.ir.kernel_ir import VarAssign

        return VarAssign(
            name=acc_name,
            value=FuncCall("fmaxf", [Var(acc_name), value]),
        )
    return AugAssign(target=acc_name, op=_reduce_op(fn), value=value)


def _emit_contraction_body(
    kernel: KernelOp,
    out_shape: tuple,
    contraction: tuple[int, Assign, Assign],
    shapes: dict[str, tuple],
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

    a_shape = shapes.get(a_port.buffer_id, ())
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

    # Both operands are in the same broadcast space (M, K, N) via their
    # IndexMapOps.  Use a single flat coord for both.
    kn = BinOp("+", BinOp("*", k, Literal(n_size, "int")), n)
    flat_coord: Expr = BinOp("+", BinOp("*", m, Literal(k_size * n_size, "int")), kn)
    if batch is not None:
        flat_coord = BinOp("+", BinOp("*", batch, Literal(m_size * k_size * n_size, "int")), flat_coord)

    name_seq_k = [0]
    inner_stmts: list[Stmt] = []
    a_val = _emit_input_value(a_port, flat_coord, shapes, inner_stmts, name_seq_k)
    b_val = _emit_input_value(b_port, flat_coord, shapes, inner_stmts, name_seq_k)
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
                    val = _emit_input_value(port, out_flat, shapes, stmts, name_seq)
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


def _kernel_name(kernel: KernelOp, idx: int, shapes: dict[str, tuple]) -> str:
    if _detect_contraction(kernel) is not None:
        return f"k{idx}_contract"
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
        return f"k{idx}_reduce"
    return f"k{idx}_pointwise"


def _launch_config(kernel: KernelOp, shapes: dict[str, tuple]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    out_shape = kernel.infer_output_shape(shapes)
    if _detect_contraction(kernel) is not None:
        int_shape = tuple(int(d) for d in out_shape if isinstance(d, int))
        m = int_shape[-2] if len(int_shape) >= 2 else 1
        n = int_shape[-1] if int_shape else 1
        batch = _numel(int_shape[:-2]) if len(int_shape) > 2 else 1
        return (n, m, batch), (1, 1, 1)
    if any(isinstance(a.op, ReduceOp) for a in kernel.body):
        # For reduce kernels, the grid is over outer rows (one block per row).
        # Use the first input Port's shape minus the last axis.
        n_rows = _reduce_n_rows(kernel, shapes)
        return (n_rows, 1, 1), (1, 1, 1)
    numel = _numel(out_shape)
    n_blocks = (numel + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def _reduce_n_rows(kernel: KernelOp, shapes: dict[str, tuple]) -> int:
    """Number of outer rows for a reduce kernel's grid.

    Uses the first input Port's shape minus the last axis (the reduce dim).
    """
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            src_shape = shapes.get(inp.buffer_id, ())
            if len(src_shape) >= 2:
                return _numel(src_shape[:-1])
            return _numel(src_shape) if src_shape else 1
    # Fallback: use output shape.
    return _numel(kernel.infer_output_shape(shapes))


def _emit_kernel_source(kernel_def: KernelDef) -> str:
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel as _emit

    return _emit(kernel_def)


def _collect_var_names(expr) -> list[str]:
    """Recursively collect all Var names from an expression tree."""
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
