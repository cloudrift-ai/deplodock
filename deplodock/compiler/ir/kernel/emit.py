"""Per-kernel CUDA codegen helpers: ``LoopOp`` node → ``GpuKernel``.

The lowering passes under ``passes/lowering/`` call these helpers — this
module just carries the recursive body walk that turns each ``LoopOp``
into a kernel-level AST. It does not know about launch ordering or
program-level buffers; callers pass in the graph + node to resolve
buffer shapes by node id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.ir.expr import BinOp, Expr, FuncCall, Literal, Ternary, Var, substitute
from deplodock.compiler.ir.graph import Graph, Node
from deplodock.compiler.ir.kernel.ir import (
    ArrayAccess,
    AugAssign,
    ForLoop,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    Stmt,
    VarAssign,
    VarDecl,
)
from deplodock.compiler.ir.kernel.ir import (
    Assign as IrAssign,
)
from deplodock.compiler.ir.loop import ACCUM_IDENTITY, Accum, Axis, Load, LoopOp, Select, SelectBranch
from deplodock.compiler.ir.loop import Assign as IrAssignStmt
from deplodock.compiler.ir.loop import Loop as IrLoop
from deplodock.compiler.ir.loop import Write as IrWrite

_BLOCK = 256


# ---------------------------------------------------------------------------
# Per-kernel emission
# ---------------------------------------------------------------------------


def emit_kernel(node: Node, kernel_name: str, graph: Graph) -> tuple[GpuKernel, list[str]]:
    """Emit one ``GpuKernel`` for a single ``LoopOp`` node."""
    params, arg_order = _build_params(node)
    body, block_size = _emit_body(node, graph)
    kd = GpuKernel(name=kernel_name, params=params, body=body, block_size=block_size)
    return kd, arg_order


def kernel_name_for(loop: LoopOp, node_id: str) -> str:
    if any(isinstance(s, Accum) for s in loop):
        return f"k_{node_id}_reduce"
    return f"k_{node_id}_pointwise"


def launch_config(node: Node) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    loop: LoopOp = node.op
    reduce_names = loop.reduce_axis_names
    free_extents = [int(a.extent) for a in loop.axes if a.name not in reduce_names]
    if free_extents:
        n_output = _numel(tuple(free_extents))
    else:
        out_shape = tuple(node.output.shape)
        n_output = _numel(out_shape) if out_shape else 1
    n_blocks = (n_output + _BLOCK - 1) // _BLOCK
    return (max(n_blocks, 1), 1, 1), (_BLOCK, 1, 1)


def emit_kernel_source(gpu_kernel: GpuKernel) -> str:
    from deplodock.compiler.ir.cuda.emit import emit_kernel as _emit

    return _emit(gpu_kernel)


@dataclass
class _Ctx:
    """Emission state threaded through recursive body walk."""

    graph: Graph
    node: Node
    env: dict[str, Expr] = field(default_factory=dict)  # axis name → Expr
    values: dict[str, Expr] = field(default_factory=dict)  # SSA name → Var
    name_seq: list[int] = field(default_factory=lambda: [0])
    acc_seq: list[int] = field(default_factory=lambda: [0])
    loop_seq: list[int] = field(default_factory=lambda: [0])

    def fresh(self) -> str:
        n = self.name_seq[0]
        self.name_seq[0] += 1
        return f"t{n}"

    def input_name(self, source: int) -> str:
        return self.node.inputs[source]

    def output_name(self) -> str:
        return self.node.id

    def buffer_shape(self, name: str) -> tuple:
        n = self.graph.nodes.get(name)
        return tuple(n.output.shape) if n is not None else ()


def _emit_body(node: Node, graph: Graph) -> tuple[list[Stmt], tuple[int, int, int]]:
    loop: LoopOp = node.op
    reduce_names = loop.reduce_axis_names
    free_axes = tuple(a for a in loop.axes if a.name not in reduce_names)
    n_threads = _numel(tuple(a.extent for a in free_axes)) if free_axes else 1

    tid = Var("tid")
    ctx = _Ctx(graph=graph, node=node, env=_axis_env_for_flat(free_axes, tid))

    guarded = _emit_stmts(loop.body, ctx)
    stmts: list[Stmt] = [
        VarDecl(
            dtype="long long",
            name="tid",
            init=BinOp("+", BinOp("*", Var("blockIdx.x"), Var("blockDim.x")), Var("threadIdx.x")),
        ),
        IfStmt(cond=BinOp("<", tid, Literal(n_threads, "int")), body=guarded),
    ]
    return stmts, (_BLOCK, 1, 1)


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
        buf_name = ctx.output_name()
        buf_shape = ctx.buffer_shape(buf_name)
        coords = [substitute(e, ctx.env) for e in s.index]
        flat = _flatten_coords(coords, buf_shape)
        out.append(IrAssign(target=ArrayAccess(array=buf_name, index=flat), value=ctx.values[s.value]))
        return

    if isinstance(s, Accum):
        acc_var = ctx.values[s.name].name
        out.append(_emit_reduce_accum(acc_var, s.op.fn, ctx.values[s.value]))
        return

    if isinstance(s, IrLoop):
        _emit_loop(s, ctx, out)
        return

    raise NotImplementedError(f"unhandled Loop IR stmt: {type(s).__name__}")


def _emit_loop(loop: IrLoop, ctx: _Ctx, out: list[Stmt]) -> None:
    accums = [x for x in loop.body if isinstance(x, Accum)]
    is_reduce = bool(accums)

    if not is_reduce and loop.axis.name in ctx.env:
        out.extend(_emit_stmts(loop.body, ctx))
        return

    k_name = f"k{ctx.loop_seq[0]}"
    ctx.loop_seq[0] += 1
    k_var = Var(k_name)

    acc_names: list[str] = []
    for a in accums:
        if a.name in acc_names:
            continue
        acc_names.append(a.name)
        acc_var = f"acc{ctx.acc_seq[0]}"
        ctx.acc_seq[0] += 1
        identity = ACCUM_IDENTITY.get(a.op.fn, 0.0)
        out.append(VarDecl(dtype="float", name=acc_var, init=Literal(identity, "float")))
        ctx.values[a.name] = Var(acc_var)

    saved_env = dict(ctx.env)
    saved_values = dict(ctx.values)
    ctx.env[loop.axis.name] = k_var
    inner = _emit_stmts(loop.body, ctx)
    ctx.env = saved_env
    ctx.values = saved_values

    out.append(ForLoop(var=k_name, start=Literal(0, "int"), end=Literal(int(loop.axis.extent), "int"), body=inner))


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
        return VarAssign(name=acc_name, value=FuncCall("fmaxf", [Var(acc_name), value]))
    if fn == "min":
        return VarAssign(name=acc_name, value=FuncCall("fminf", [Var(acc_name), value]))
    op = {"add": "+=", "sum": "+=", "mul": "*=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


_SUPPORTED_UNARY = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "abs": "fabsf",
}


def _apply_elementwise(fn: str, inputs: list[Expr]) -> Expr:
    if fn in {"add", "sub", "mul", "div", "mod"}:
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[fn]
        return BinOp(op, inputs[0], inputs[1])
    if fn == "max":
        return FuncCall("fmaxf", list(inputs))
    if fn == "min":
        return FuncCall("fminf", list(inputs))
    if fn == "pow":
        return FuncCall("powf", list(inputs))
    if fn == "neg":
        return BinOp("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "recip":
        return BinOp("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCall("fmaxf", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        neg_x = BinOp("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCall("expf", [neg_x])
        return BinOp("/", Literal(1.0, "float"), BinOp("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY:
        return FuncCall(_SUPPORTED_UNARY[fn], list(inputs))
    raise NotImplementedError(f"elementwise fn={fn} not yet supported by emit")


def _numel(shape: tuple) -> int:
    return int(math.prod(int(d) for d in shape if isinstance(d, int)) or 1)


def _build_params(node: Node) -> tuple[list[GpuKernelParam], list[str]]:
    output_name = node.id
    seen: list[str] = []
    for buf_name in node.inputs:
        if buf_name not in seen and buf_name != output_name:
            seen.append(buf_name)
    params = [GpuKernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(GpuKernelParam(dtype="float*", name=output_name))
    return params, seen + [output_name]
