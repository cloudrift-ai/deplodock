"""Loop IR → Tile IR (single-thread, naive).

Mechanical translation:

- Each Loop IR ``Loop`` becomes a Tile IR ``FreeLoop`` (no Accum in body)
  or ``Reduce`` (with Accums collected into ``Reduce.accs``).
- Each leaf ``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum``
  translates 1:1 to ``Let`` / ``Store`` / ``AccumFold``.
- Top-level non-Loop stmts (typically scalar Loads with empty index)
  become ``Kernel.prologue``.

The output Kernel has ``thread_axes == ()`` and ``block_axes == ()`` —
it's a fully-serial single-thread program. ``ExtractGlobalSchedule``
(step 4) strips the outer FreeLoop chain into ``thread_axes``.

Reduce ops use ``ElementwiseImpl`` directly — same vocabulary as Loop IR's
``Accum.op`` (``"add"`` / ``"multiply"`` / ``"maximum"`` / ``"minimum"`` plus
``"sum"`` / ``"prod"`` aliases). The ``AccumFold.op`` and ``Acc.op`` carry
the same ``ElementwiseImpl`` instance that came in on the source ``Accum`` —
no name remapping. Identity is read from ``ElementwiseImpl.identity`` at
render time. Elementwise op → Expr translation (``_op_to_expr``) mirrors
``_common.apply_elementwise`` so the rendered CUDA matches today's output
for the same Loop IR input.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Load,
    Loop,
    LoopOp,
    Select,
    Write,
)
from deplodock.compiler.ir.loop import (
    Stmt as LoopStmt,
)
from deplodock.compiler.ir.tile.ir import (
    Acc,
    AccumFold,
    FreeLoop,
    Index,
    Kernel,
    Let,
    Param,
    Reduce,
    Stmt,
    Store,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_naive(loop_op: LoopOp, kernel_name: str, inputs: tuple[Param, ...], output: Param) -> Kernel:
    """Translate a ``LoopOp`` into a single-thread serial ``Kernel``.

    ``inputs`` and ``output`` populate ``Kernel.params`` (with ``Param.shape``
    used by the renderer to row-major-flatten multi-dim ``Index`` accesses).
    Buffer identity is carried inline on Loop IR's ``Load.source`` and
    ``Write.output`` (now strings), so lowering only needs to translate
    statement types — no positional source lookup against ``inputs``.
    """
    ctx = _Ctx()

    pre: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        pre.append(rest[0])
        rest = rest[1:]
    prologue = _lower_body(tuple(pre), ctx)
    body = _lower_body(rest, ctx)

    return Kernel(
        name=kernel_name,
        params=(*inputs, output),
        body=tuple(body),
        prologue=tuple(prologue),
    )


# ---------------------------------------------------------------------------
# Recursive walk
# ---------------------------------------------------------------------------


class _Ctx:
    """Lowering context — currently empty since ``Load.source`` and
    ``Write.output`` carry buf names directly. Retained as a struct so the
    recursive walk has a single hand-off point if more state is needed
    later (e.g. fresh-name counters for tiled rewrites)."""

    __slots__ = ()


def _lower_body(stmts: tuple[LoopStmt, ...], ctx: _Ctx) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        out.extend(_lower_stmt(s, ctx))
    return out


def _lower_stmt(s: LoopStmt, ctx: _Ctx) -> list[Stmt]:
    if isinstance(s, Loop):
        return [_lower_loop(s, ctx)]
    if isinstance(s, Load):
        return [Let(s.name, Index(s.input, tuple(s.index)))]
    if isinstance(s, Assign):
        args = [Var(a) for a in s.args]
        return [Let(s.name, _op_to_expr(s.op.name, args))]
    if isinstance(s, Select):
        return [Let(s.name, _select_to_ternary(s))]
    if isinstance(s, Write):
        return [Store(s.output, tuple(s.index), Var(s.value))]
    if isinstance(s, Accum):
        return [AccumFold(s.name, s.op, Var(s.value))]
    raise NotImplementedError(f"lower_naive: unhandled Loop IR stmt {type(s).__name__}")


def _lower_loop(loop: Loop, ctx: _Ctx) -> Stmt:
    """A reduce Loop becomes ``Reduce(axis, accs, body)``; otherwise ``FreeLoop``.

    For reduce, walk the body and pick out every ``Accum`` to populate
    ``Reduce.accs`` (one Acc per distinct accumulator name); the AccumFold
    stmts stay in the body in their original positions.
    """
    accums = [s for s in loop.body if isinstance(s, Accum)]
    if accums:
        seen: dict[str, Acc] = {}
        for a in accums:
            if a.name in seen:
                continue
            seen[a.name] = Acc(name=a.name, op=a.op)
        body = _lower_body(loop.body, ctx)
        return Reduce(axis=loop.axis, accs=tuple(seen.values()), body=tuple(body))
    body = _lower_body(loop.body, ctx)
    return FreeLoop(axis=loop.axis, body=tuple(body))


# ---------------------------------------------------------------------------
# Op-name → Expr translation (mirrors _common.apply_elementwise)
# ---------------------------------------------------------------------------


_BINARY_OP: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "mod": "%",
}

_SUPPORTED_UNARY_INTRINSIC: dict[str, str] = {
    "exp": "exp",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "abs": "fabs",
}


def _op_to_expr(fn: str, inputs: list[Expr]) -> Expr:
    if fn in _BINARY_OP:
        return BinaryExpr(_BINARY_OP[fn], inputs[0], inputs[1])
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
        return BinaryExpr(
            "/",
            Literal(1.0, "float"),
            BinaryExpr("+", Literal(1.0, "float"), exp_neg),
        )
    if fn in _SUPPORTED_UNARY_INTRINSIC:
        return FuncCallExpr(_SUPPORTED_UNARY_INTRINSIC[fn], list(inputs))
    raise NotImplementedError(f"lower_naive: elementwise fn={fn!r} not yet supported")


def _select_to_ternary(s: Select) -> Expr:
    """Build a chained ternary from a Loop IR ``Select``.

    Last branch is the catch-all (its predicate is the deepest else).
    """
    branches = list(s.branches)
    result: Expr = Var(branches[-1].value)
    for b in reversed(branches[:-1]):
        result = TernaryExpr(cond=b.select, if_true=Var(b.value), if_false=result)
    return result


__all__ = ["lower_naive"]
