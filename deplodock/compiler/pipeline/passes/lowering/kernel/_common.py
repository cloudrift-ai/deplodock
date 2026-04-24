"""Shared helpers for the unified kernel emitter.

Hosts the post-fusion stmt walker (``emit_stmt`` / ``emit_stmts``), the
elementwise / accumulator op spelling tables, the flat-tid axis decoder,
and small utilities (``build_params``, ``kernel_name_for``, ``numel_axes``)
shared between the unified emitter and the legacy fallback emitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, FuncCallExpr, Literal, TernaryExpr, Var, substitute
from deplodock.compiler.ir.kernel.ir import (
    ArrayAccess,
    AugAssign,
    Stmt,
    VarAssign,
    VarDecl,
)
from deplodock.compiler.ir.kernel.ir import (
    Assign as IrAssign,
)
from deplodock.compiler.ir.loop import Accum, Axis, Load, LoopOp, Select, SelectBranch
from deplodock.compiler.ir.loop import Assign as IrAssignStmt
from deplodock.compiler.ir.loop import Write as IrWrite

BLOCK = 256


# ---------------------------------------------------------------------------
# Naming + parameter binding
# ---------------------------------------------------------------------------


def kernel_name_for(loop: LoopOp, node_id: str) -> str:
    if any(isinstance(s, Accum) for s in loop):
        return f"k_{node_id}_reduce"
    return f"k_{node_id}_pointwise"


def build_params(node: Node):
    """Return ``(params, arg_order)`` for a LoopOp node.

    Inputs are deduped (a buffer read twice still binds one kernel arg);
    output is appended last.
    """
    from deplodock.compiler.ir.kernel.ir import GpuKernelParam

    output_name = node.id
    seen: list[str] = []
    for buf_name in node.inputs:
        if buf_name not in seen and buf_name != output_name:
            seen.append(buf_name)
    params = [GpuKernelParam(dtype="const float*", name=bid) for bid in seen]
    params.append(GpuKernelParam(dtype="float*", name=output_name))
    return params, seen + [output_name]


# ---------------------------------------------------------------------------
# Emission context
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    """Per-kernel emission state threaded through the body walk."""

    graph: Graph
    node: Node
    env: dict[str, Expr] = field(default_factory=dict)  # axis name → Expr
    values: dict[str, Expr] = field(default_factory=dict)  # SSA name → Var
    name_seq: list[int] = field(default_factory=lambda: [0])
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


# ---------------------------------------------------------------------------
# Body-stmt walker
# ---------------------------------------------------------------------------


def emit_stmts(stmts: tuple, ctx: Ctx) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        emit_stmt(s, ctx, out)
    return out


def emit_stmt(s, ctx: Ctx, out: list[Stmt]) -> None:
    """Render one Loop-IR stmt into the GpuKernel AST.

    Handles Load / Assign / Select / Write / Accum. Loop walking is the
    caller's responsibility (the unified emitter dispatches reduce/output
    Loops directly so this walker doesn't need a strategy hook).
    """
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
        out.append(VarDecl(dtype="float", name=tname, init=apply_elementwise(s.op.name, args)))
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
        out.append(emit_reduce_accum(acc_var, s.op.name, ctx.values[s.value]))
        return

    raise NotImplementedError(f"emit_stmt: unhandled {type(s).__name__}")


def _emit_write(s: IrWrite, ctx: Ctx, out: list[Stmt]) -> None:
    buf_name = ctx.output_name()
    buf_shape = ctx.buffer_shape(buf_name)
    coords = [substitute(e, ctx.env) for e in s.index]
    flat = _flatten_coords(coords, buf_shape)
    out.append(IrAssign(target=ArrayAccess(array=buf_name, index=flat), value=ctx.values[s.value]))


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _env_with_values(ctx: Ctx) -> dict[str, Expr]:
    return {**ctx.env, **ctx.values}


def axis_env_for_flat(axes: tuple[Axis, ...], flat_idx: Expr) -> dict[str, Expr]:
    """Decompose a flat (row-major) thread index into per-axis Exprs."""
    env: dict[str, Expr] = {}
    if not axes:
        return env
    remainder = flat_idx
    for i in range(len(axes) - 1, -1, -1):
        dim = int(axes[i].extent)
        if i == 0:
            env[axes[i].name] = remainder
        else:
            env[axes[i].name] = BinaryExpr("%", remainder, Literal(dim, "int"))
            remainder = BinaryExpr("/", remainder, Literal(dim, "int"))
    return env


def _flatten_coords(coords: list[Expr], shape: tuple) -> Expr:
    if not coords:
        return Literal(0, "int")
    flat: Expr = Literal(0, "int")
    stride = 1
    dims = [int(d) if isinstance(d, int) else 1 for d in shape]
    for d in range(len(coords) - 1, -1, -1):
        coord = coords[d]
        term = coord if stride == 1 else BinaryExpr("*", coord, Literal(stride, "int"))
        if isinstance(flat, Literal) and flat.value == 0:
            flat = term
        else:
            flat = BinaryExpr("+", term, flat)
        if d > 0 and d < len(dims):
            stride *= dims[d]
    return flat


def _emit_select(stmt: Select, values: dict[str, Expr], axis_env: dict[str, Expr]) -> Expr:
    branches: list[SelectBranch] = list(stmt.branches)
    result: Expr = values[branches[-1].value]
    for branch in reversed(branches[:-1]):
        cond = substitute(branch.select, axis_env)
        result = TernaryExpr(cond=cond, if_true=values[branch.value], if_false=result)
    return result


def emit_reduce_accum(acc_name: str, fn: str, value: Expr) -> Stmt:
    if fn == "maximum":
        return VarAssign(name=acc_name, value=FuncCallExpr("fmax", [Var(acc_name), value]))
    if fn == "minimum":
        return VarAssign(name=acc_name, value=FuncCallExpr("fmin", [Var(acc_name), value]))
    op = {"add": "+=", "sum": "+=", "multiply": "*=", "prod": "*="}.get(fn, "+=")
    return AugAssign(target=acc_name, op=op, value=value)


_SUPPORTED_UNARY = {
    "exp": "exp",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "abs": "fabs",
}


def apply_elementwise(fn: str, inputs: list[Expr]) -> Expr:
    if fn in {"add", "subtract", "multiply", "divide", "mod"}:
        op = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "mod": "%"}[fn]
        return BinaryExpr(op, inputs[0], inputs[1])
    if fn == "maximum":
        return FuncCallExpr("fmax", list(inputs))
    if fn == "minimum":
        return FuncCallExpr("fmin", list(inputs))
    if fn == "pow":
        return FuncCallExpr("pow", list(inputs))
    if fn == "negative":
        return BinaryExpr("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "reciprocal":
        return BinaryExpr("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCallExpr("fmax", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        neg_x = BinaryExpr("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCallExpr("exp", [neg_x])
        return BinaryExpr("/", Literal(1.0, "float"), BinaryExpr("+", Literal(1.0, "float"), exp_neg))
    if fn in _SUPPORTED_UNARY:
        return FuncCallExpr(_SUPPORTED_UNARY[fn], list(inputs))
    raise NotImplementedError(f"elementwise fn={fn} not yet supported by emit")


def numel_axes(axes: tuple[Axis, ...]) -> int:
    return int(math.prod(int(a.extent) for a in axes) or 1)


__all__ = [
    "BLOCK",
    "Ctx",
    "apply_elementwise",
    "axis_env_for_flat",
    "build_params",
    "emit_reduce_accum",
    "emit_stmt",
    "emit_stmts",
    "kernel_name_for",
    "numel_axes",
]
