"""Per-kernel GPU codegen: ``LoopOp`` node → ``GpuKernel``.

Two strategies, dispatched by ``_pick_strategy`` from the loop's axis geometry:

- **A — thread-per-output, serial reduce.** Flatten every free axis into the
  thread id; each thread runs its reduce loops serially in a register. Fits
  matmul (reduce extent modest, output numel huge) and every pure-elementwise
  shape.
- **B — block-per-outer, smem reduce.** One block per outer-free tuple; threads
  cooperate across the reduce axis via shared-memory tree reduction, and the
  inner-free loop is parallelized across threads. Fits softmax, RMSNorm, and
  any fused kernel with a large reduce axis and a small outer batch.

The strategy only changes how reduce Loops are emitted — Loads, elementwise
Assigns, Selects, and Writes are walked the same way in both paths.

Emitted kernel IR is backend-neutral: ``Builtin("thread_idx.x")`` /
``Builtin("block_idx.x")`` for launch indices, neutral intrinsic names
(``"exp"`` / ``"rsqrt"`` / ``"fmax"`` / ``"fmin"`` / ``"pow"`` / ``"fabs"``),
``ArrayDecl(storage="shared", ...)`` for smem tiles. The CUDA emitter
translates these to CUDA spellings at source-render time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinOp, Builtin, Expr, FuncCall, Literal, Ternary, Var, substitute
from deplodock.compiler.ir.kernel.ir import (
    ArrayAccess,
    ArrayDecl,
    AugAssign,
    ForLoop,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    Stmt,
    SyncThreads,
    VarAssign,
    VarDecl,
)
from deplodock.compiler.ir.kernel.ir import (
    Assign as IrAssign,
)
from deplodock.compiler.ir.loop import Accum, Axis, Load, LoopOp, Select, SelectBranch
from deplodock.compiler.ir.loop import Assign as IrAssignStmt
from deplodock.compiler.ir.loop import Loop as IrLoop
from deplodock.compiler.ir.loop import Write as IrWrite

_BLOCK = 256
_SMEM_OUTER_CAP = 16 * _BLOCK  # outer_free_numel below this → prefer Strategy B when reduce is large


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Strategy:
    """Chosen lowering strategy for one LoopOp.

    ``kind`` is ``"A"`` (thread-per-output / serial reduce) or ``"B"``
    (block-per-outer / smem reduce). The other fields are derived geometry
    that both the body walker and ``launch_config`` consult.
    """

    kind: str  # "A" or "B"
    outer_free: tuple[Axis, ...]
    inner_free: tuple[Axis, ...]
    reduce_axes: tuple[Axis, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


def pick_strategy(loop: LoopOp) -> _Strategy:
    """Classify a LoopOp and pick a lowering strategy.

    Heuristic: use B (smem) when there is at least one reduce axis whose
    extent ≥ BLOCK AND the outer-free numel isn't already large enough to
    saturate the device under Strategy A.
    """
    reduce_names = loop.reduce_axis_names
    reduce_axes = tuple(a for a in loop.axes if a.name in reduce_names)
    free_axes = tuple(a for a in loop.axes if a.name not in reduce_names)

    # Partition free axes into "outer" (those that wrap the whole body at the
    # top) and "inner" (those that only wrap the output-write subtree).
    outer_free, inner_free = _split_free_axes(loop, free_axes)

    outer_numel = _numel_axes(outer_free) if outer_free else 1
    max_reduce_extent = max((int(a.extent) for a in reduce_axes), default=1)

    pick_b = bool(reduce_axes) and max_reduce_extent >= _BLOCK and outer_numel < _SMEM_OUTER_CAP

    if pick_b:
        grid = (max(outer_numel, 1), 1, 1)
        block = (_BLOCK, 1, 1)
        return _Strategy("B", outer_free, inner_free, reduce_axes, grid, block)

    # Strategy A: flatten all free axes into tid.
    n_threads = _numel_axes(free_axes) if free_axes else 1
    n_blocks = (n_threads + _BLOCK - 1) // _BLOCK
    grid = (max(n_blocks, 1), 1, 1)
    block = (_BLOCK, 1, 1)
    # In Strategy A we don't distinguish outer/inner — all free axes are threaded together.
    return _Strategy("A", free_axes, (), reduce_axes, grid, block)


def _split_free_axes(loop: LoopOp, free_axes: tuple[Axis, ...]) -> tuple[tuple[Axis, ...], tuple[Axis, ...]]:
    """Partition ``free_axes`` into ``(outer, inner)``.

    An outer-free axis is one whose Loop is the *only* Loop at its level in the
    body tree (Loop-invariant sibling stmts like loading a scalar constant are
    allowed) and wraps every reduce Loop / Write below. We descend into such a
    Loop's body and repeat until we hit a level with multiple Loops, a reduce
    Loop, or no Loops.
    """
    if not loop.body:
        return (), free_axes
    reduce_names = loop.reduce_axis_names
    body: tuple = loop.body
    outer_names: list[str] = []
    while True:
        loops_here = [s for s in body if isinstance(s, IrLoop)]
        if len(loops_here) != 1:
            break
        only = loops_here[0]
        if only.axis.name in reduce_names:
            break
        outer_names.append(only.axis.name)
        body = only.body
    outer_set = set(outer_names)
    outer = tuple(a for a in free_axes if a.name in outer_set)
    inner = tuple(a for a in free_axes if a.name not in outer_set)
    return outer, inner


# ---------------------------------------------------------------------------
# Top-level entry points (used by the rule)
# ---------------------------------------------------------------------------


def emit_kernel(node: Node, kernel_name: str, graph: Graph) -> tuple[GpuKernel, list[str]]:
    """Emit one ``GpuKernel`` for a single ``LoopOp`` node."""
    params, arg_order = _build_params(node)
    strategy = pick_strategy(node.op)
    body, block_size = _emit_body(node, graph, strategy)
    kd = GpuKernel(name=kernel_name, params=params, body=body, block_size=block_size)
    return kd, arg_order


def kernel_name_for(loop: LoopOp, node_id: str) -> str:
    if any(isinstance(s, Accum) for s in loop):
        return f"k_{node_id}_reduce"
    return f"k_{node_id}_pointwise"


def launch_config(node: Node) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Pick grid/block for a LoopOp. Uses the same strategy logic as ``emit_kernel``."""
    strategy = pick_strategy(node.op)
    return strategy.grid, strategy.block


def emit_kernel_source(gpu_kernel: GpuKernel) -> str:
    from deplodock.compiler.pipeline.passes.lowering.cuda._emit import emit_kernel as _emit

    return _emit(gpu_kernel)


# ---------------------------------------------------------------------------
# Body walk
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Emission state threaded through recursive body walk."""

    graph: Graph
    node: Node
    strategy: _Strategy
    env: dict[str, Expr] = field(default_factory=dict)  # axis name → Expr
    values: dict[str, Expr] = field(default_factory=dict)  # SSA name → Var
    name_seq: list[int] = field(default_factory=lambda: [0])
    acc_seq: list[int] = field(default_factory=lambda: [0])
    loop_seq: list[int] = field(default_factory=lambda: [0])
    smem_seq: list[int] = field(default_factory=lambda: [0])

    def fresh(self) -> str:
        n = self.name_seq[0]
        self.name_seq[0] += 1
        return f"t{n}"

    def fresh_smem(self) -> str:
        n = self.smem_seq[0]
        self.smem_seq[0] += 1
        return f"smem{n}"

    def input_name(self, source: int) -> str:
        return self.node.inputs[source]

    def output_name(self) -> str:
        return self.node.id

    def buffer_shape(self, name: str) -> tuple:
        n = self.graph.nodes.get(name)
        return tuple(n.output.shape) if n is not None else ()


def _emit_body(node: Node, graph: Graph, strategy: _Strategy) -> tuple[list[Stmt], tuple[int, int, int]]:
    loop: LoopOp = node.op
    if strategy.kind == "A":
        return _emit_body_a(node, graph, loop, strategy)
    return _emit_body_b(node, graph, loop, strategy)


def _emit_body_a(node: Node, graph: Graph, loop: LoopOp, strategy: _Strategy) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Strategy A: flatten every free axis into a single tid; serial reduce per thread."""
    free_axes = strategy.outer_free  # in A, all free axes are "outer"
    n_threads = _numel_axes(free_axes) if free_axes else 1
    tid = Var("tid")
    ctx = _Ctx(graph=graph, node=node, strategy=strategy, env=_axis_env_for_flat(free_axes, tid))
    guarded = _emit_stmts(loop.body, ctx)
    stmts: list[Stmt] = [
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinOp("+", BinOp("*", Builtin("block_idx.x"), Builtin("block_dim.x")), Builtin("thread_idx.x")),
        ),
        IfStmt(cond=BinOp("<", tid, Literal(n_threads, "int")), body=guarded),
    ]
    return stmts, strategy.block


def _emit_body_b(node: Node, graph: Graph, loop: LoopOp, strategy: _Strategy) -> tuple[list[Stmt], tuple[int, int, int]]:
    """Strategy B: block_idx = outer-free tile; thread_idx cooperates on reduce and inner-free."""
    ctx = _Ctx(
        graph=graph,
        node=node,
        strategy=strategy,
        env=_axis_env_for_flat(strategy.outer_free, Builtin("block_idx.x")),
    )
    body = _emit_stmts(loop.body, ctx)
    return body, strategy.block


def _emit_stmts(stmts: tuple, ctx: _Ctx) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        _emit_stmt(s, ctx, out)
    return out


def _emit_stmt(s, ctx: _Ctx, out: list[Stmt]) -> None:
    if isinstance(s, Load):
        buf_name = ctx.input_name(s.source)
        src_shape = ctx.buffer_shape(buf_name)
        coords = [substitute(e, _env_with_values(ctx)) for e in s.index]
        flat = _flatten_coords(coords, src_shape)
        tname = ctx.fresh()
        access: Expr = ArrayAccess(array=buf_name, index=flat) if s.index else ArrayAccess(array=buf_name, index=Literal(0, "int"))
        out.append(VarDecl(dtype="float", name=tname, init=access))
        ctx.values[s.name] = Var(tname)
        return

    if isinstance(s, IrAssignStmt):
        args = [ctx.values[a] for a in s.args]
        tname = ctx.fresh()
        out.append(VarDecl(dtype="float", name=tname, init=_apply_elementwise(s.op.fn, args)))
        ctx.values[s.name] = Var(tname)
        return

    if isinstance(s, Select):
        tname = ctx.fresh()
        out.append(VarDecl(dtype="float", name=tname, init=_emit_select(s, ctx.values, ctx.env)))
        ctx.values[s.name] = Var(tname)
        return

    if isinstance(s, IrWrite):
        _emit_write(s, ctx, out)
        return

    if isinstance(s, Accum):
        acc_var = ctx.values[s.name].name
        out.append(_emit_reduce_accum(acc_var, s.op.fn, ctx.values[s.value]))
        return

    if isinstance(s, IrLoop):
        _emit_loop(s, ctx, out)
        return

    raise NotImplementedError(f"unhandled Loop IR stmt: {type(s).__name__}")


def _emit_write(s: IrWrite, ctx: _Ctx, out: list[Stmt]) -> None:
    """Emit a Write to the output buffer.

    In Strategy B, writes happen inside an inner-free loop whose axis is
    bound to ``thread_idx.x``; no extra guard is needed because the loop
    body only runs when the thread owns the output element.
    """
    buf_name = ctx.output_name()
    buf_shape = ctx.buffer_shape(buf_name)
    coords = [substitute(e, ctx.env) for e in s.index]
    flat = _flatten_coords(coords, buf_shape)
    out.append(IrAssign(target=ArrayAccess(array=buf_name, index=flat), value=ctx.values[s.value]))


def _emit_loop(loop: IrLoop, ctx: _Ctx, out: list[Stmt]) -> None:
    is_reduce = any(isinstance(x, Accum) for x in loop.body)
    if is_reduce:
        if ctx.strategy.kind == "B":
            _emit_reduce_loop_smem(loop, ctx, out)
        else:
            _emit_reduce_loop_serial(loop, ctx, out)
        return

    # Non-reduce Loop. If the axis is already bound (Strategy A folds all frees
    # into tid up-front), inline. Otherwise — Strategy B's inner-free loop —
    # bind the axis to thread_idx.x and inline (each thread owns one value).
    if loop.axis.name in ctx.env:
        out.extend(_emit_stmts(loop.body, ctx))
        return

    # Strategy B inner-free loop. Each thread owns one (or more, if extent > BLOCK)
    # output elements along this axis. When extent <= BLOCK, emit a tid-guard;
    # otherwise, a strided for-loop (``for k=tid; k<extent; k+=BLOCK``).
    tid = Builtin("thread_idx.x")
    extent = int(loop.axis.extent)

    if extent <= _BLOCK:
        saved_env = dict(ctx.env)
        saved_values = dict(ctx.values)
        ctx.env[loop.axis.name] = tid
        inner = _emit_stmts(loop.body, ctx)
        ctx.env = saved_env
        ctx.values = saved_values
        out.append(IfStmt(cond=BinOp("<", tid, Literal(extent, "int")), body=inner))
        return

    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[loop.axis.name] = Var(k_name)
    inner = _emit_stmts(loop.body, ctx)
    ctx.env = saved_env
    ctx.values = saved_values
    out.append(
        ForLoop(
            var=k_name,
            start=tid,
            end=Literal(extent, "int"),
            body=inner,
            step=Literal(_BLOCK, "int"),
        )
    )


def _emit_reduce_loop_serial(loop: IrLoop, ctx: _Ctx, out: list[Stmt]) -> None:
    """Strategy A reduce: serial for-loop with register accumulator."""
    accums = [x for x in loop.body if isinstance(x, Accum)]
    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    k_var = Var(k_name)

    seen: set[str] = set()
    for a in accums:
        if a.name in seen:
            continue
        seen.add(a.name)
        acc_var = f"acc{ctx.acc_seq[0]}"
        ctx.acc_seq[0] += 1
        identity = a.op.identity if a.op.identity is not None else 0.0
        out.append(VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")))
        ctx.values[a.name] = Var(acc_var)

    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[loop.axis.name] = k_var
    inner = _emit_stmts(loop.body, ctx)
    ctx.env = saved_env
    ctx.values = saved_values

    out.append(ForLoop(var=k_name, start=Literal(0, "int"), end=Literal(int(loop.axis.extent), "int"), body=inner))


def _emit_reduce_loop_smem(loop: IrLoop, ctx: _Ctx, out: list[Stmt]) -> None:
    """Strategy B reduce: threads cooperate via shared-memory tree reduction.

    Shape emitted per distinct accumulator (for a loop with a single Accum — the
    common case from fusion output):

        float acc0 = identity;
        for (int k = thread_idx.x; k < R; k += BLOCK) { <body, env[axis]=k>; acc0 op= v }
        __shared__ float smem0[BLOCK];
        smem0[thread_idx.x] = acc0;
        __syncthreads();
        for (int s = BLOCK/2; s > 0; s >>= 1) {
            if (thread_idx.x < s) smem0[thread_idx.x] op= smem0[thread_idx.x + s];
            __syncthreads();
        }
        acc0 = smem0[0];  // broadcast, safe after barrier
    """
    accums = [x for x in loop.body if isinstance(x, Accum)]
    # Declare register accumulators.
    seen: dict[str, str] = {}
    for a in accums:
        if a.name in seen:
            continue
        acc_var = f"acc{ctx.acc_seq[0]}"
        ctx.acc_seq[0] += 1
        identity = a.op.identity if a.op.identity is not None else 0.0
        out.append(VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")))
        ctx.values[a.name] = Var(acc_var)
        seen[a.name] = acc_var

    # Emit strided partial loop: for (k = tid; k < R; k += BLOCK) { body }.
    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    k_var = Var(k_name)
    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[loop.axis.name] = k_var
    inner = _emit_stmts(loop.body, ctx)
    ctx.env = saved_env
    ctx.values = saved_values
    out.append(
        ForLoop(
            var=k_name,
            start=Builtin("thread_idx.x"),
            end=Literal(int(loop.axis.extent), "int"),
            body=inner,
            step=Literal(_BLOCK, "int"),
        )
    )

    # Per accumulator: smem tile + tree-halve (unrolled) + broadcast from smem[0].
    tid = Builtin("thread_idx.x")
    for name, acc_var in seen.items():
        accum = next(a for a in accums if a.name == name)
        smem_name = ctx.fresh_smem()
        out.append(ArrayDecl(dtype="float", name=smem_name, dimensions=[_BLOCK], storage="shared"))
        out.append(IrAssign(target=ArrayAccess(array=smem_name, index=tid), value=Var(acc_var)))
        out.append(SyncThreads())

        # Unrolled tree halve: BLOCK is compile-time, so emit one round per stride.
        stride = _BLOCK // 2
        while stride > 0:
            combine = _smem_combine_stmt(smem_name, stride, tid, accum.op.fn)
            out.append(IfStmt(cond=BinOp("<", tid, Literal(stride, "int")), body=[combine]))
            out.append(SyncThreads())
            stride //= 2

        # Broadcast final value to every thread's register (safe after the last barrier).
        out.append(VarAssign(name=acc_var, value=ArrayAccess(array=smem_name, index=Literal(0, "int"))))


def _smem_combine_stmt(smem_name: str, stride: int, tid: Expr, op_fn: str) -> Stmt:
    """One combine step of the tree reduction: ``smem[tid] = smem[tid] op smem[tid+stride]``."""
    lhs = ArrayAccess(array=smem_name, index=tid)
    rhs = ArrayAccess(array=smem_name, index=BinOp("+", tid, Literal(stride, "int")))
    if op_fn == "max":
        return IrAssign(target=lhs, value=FuncCall("fmax", [lhs, rhs]))
    if op_fn == "min":
        return IrAssign(target=lhs, value=FuncCall("fmin", [lhs, rhs]))
    if op_fn in ("add", "sum"):
        return IrAssign(target=lhs, value=BinOp("+", lhs, rhs))
    if op_fn in ("mul", "prod"):
        return IrAssign(target=lhs, value=BinOp("*", lhs, rhs))
    return IrAssign(target=lhs, value=BinOp("+", lhs, rhs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_with_values(ctx: _Ctx) -> dict[str, Expr]:
    return {**ctx.env, **ctx.values}


def _axis_env_for_flat(axes: tuple[Axis, ...], flat_idx: Expr) -> dict[str, Expr]:
    env: dict[str, Expr] = {}
    if not axes:
        return env
    remainder = flat_idx
    for i in range(len(axes) - 1, -1, -1):
        dim = int(axes[i].extent)
        if i == 0:
            env[axes[i].name] = remainder
        else:
            env[axes[i].name] = BinOp("%", remainder, Literal(dim, "int"))
            remainder = BinOp("/", remainder, Literal(dim, "int"))
    return env


def _flatten_coords(coords: list[Expr], shape: tuple) -> Expr:
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


def _emit_select(stmt: Select, values: dict[str, Expr], axis_env: dict[str, Expr]) -> Expr:
    branches: list[SelectBranch] = list(stmt.branches)
    result: Expr = values[branches[-1].value]
    for branch in reversed(branches[:-1]):
        cond = substitute(branch.select, axis_env)
        result = Ternary(cond=cond, if_true=values[branch.value], if_false=result)
    return result


def _emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    if fn == "max":
        return VarAssign(name=acc_name, value=FuncCall("fmax", [Var(acc_name), value]))
    if fn == "min":
        return VarAssign(name=acc_name, value=FuncCall("fmin", [Var(acc_name), value]))
    op = {"add": "+=", "sum": "+=", "mul": "*=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


_SUPPORTED_UNARY = {
    "exp": "exp",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "abs": "fabs",
}


def _apply_elementwise(fn: str, inputs: list[Expr]) -> Expr:
    if fn in {"add", "sub", "mul", "div", "mod"}:
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[fn]
        return BinOp(op, inputs[0], inputs[1])
    if fn == "max":
        return FuncCall("fmax", list(inputs))
    if fn == "min":
        return FuncCall("fmin", list(inputs))
    if fn == "pow":
        return FuncCall("pow", list(inputs))
    if fn == "neg":
        return BinOp("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "reciprocal":
        return BinOp("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCall("fmax", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        neg_x = BinOp("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCall("exp", [neg_x])
        return BinOp("/", Literal(1.0, "float"), BinOp("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY:
        return FuncCall(_SUPPORTED_UNARY[fn], list(inputs))
    raise NotImplementedError(f"elementwise fn={fn} not yet supported by emit")


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _numel_axes(axes: tuple[Axis, ...]) -> int:
    return int(math.prod(int(a.extent) for a in axes) or 1)


def _build_params(node: Node) -> tuple[list[GpuKernelParam], list[str]]:
    output_name = node.id
    seen: list[str] = []
    for buf_name in node.inputs:
        if buf_name not in seen and buf_name != output_name:
            seen.append(buf_name)
    params = [GpuKernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(GpuKernelParam(dtype="float*", name=output_name))
    return params, seen + [output_name]
