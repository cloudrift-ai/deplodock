"""Leaf ``Stmt`` subclasses — pure compute primitives (no nested bodies).

``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select`` — each
produces / writes a single SSA value. Block-structured stmts (Loop /
Tile / Cond) live in ``blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr, Literal, Var, _float_lit
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, _pad, op_to_expr, render_index, select_to_ternary


@dataclass(frozen=True)
class Load(Stmt):
    """Read a value from an external input buffer into an SSA name.

    Each external-buffer read is an explicit body statement. ``input`` is
    the source buffer's name (matches the producing graph node's id);
    ``index`` is the dim-wise access pattern over the enclosing axes.
    The produced SSA ``name`` is a regular value that downstream stmts
    read.

    A Load is rendered as a literal binding (``float name = <value>;``)
    when ``ctx.literal_constants`` carries a value for ``input`` — the
    scalar-constant-inlining path populates that map at the cuda
    lowering boundary so kernels can embed ``ConstantOp`` values
    directly instead of taking them as ``float*`` parameters.
    """

    name: str
    input: str
    index: tuple[Expr, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def is_literal(self, literal_constants: dict[str, float]) -> bool:
        return self.input in literal_constants

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}{self.name} = load {self.input}[{idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        lit = ctx.literal_constants.get(self.input) if ctx.literal_constants else None
        if lit is not None:
            return [f"{_pad(ctx.indent)}float {self.name} = {_float_lit(lit)};"]
        flat = render_index(self.input, self.index, ctx)
        # When the source buffer is fp16, convert at the load boundary so
        # all SSA locals stay in ``float`` (CUDA's ``__half`` is a struct
        # with no implicit conversion to ``float``).
        if ctx.buffer_dtypes.get(self.input) == "f16":
            return [f"{_pad(ctx.indent)}float {self.name} = __half2float({self.input}[{flat}]);"]
        return [f"{_pad(ctx.indent)}float {self.name} = {self.input}[{flat}];"]


@dataclass(frozen=True)
class Assign(Stmt):
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is an ``ElementwiseImpl`` — the elementwise combine (add /
    mul / exp / ...). Reductions live in ``Accum``. ``args`` reference
    ``$N`` ports, an ``Accum.name`` (reads current / finalized acc
    value), or prior SSA names.
    """

    name: str
    op: ElementwiseImpl
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    def deps(self) -> tuple[str, ...]:
        return self.args

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} = {self.op.name}({', '.join(self.args)})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        args: list[Expr] = [Var(a) for a in self.args]
        expr = op_to_expr(self.op.name, args)
        return [f"{_pad(ctx.indent)}float {self.name} = {expr.render(ctx)};"]


@dataclass(frozen=True)
class Accum(Stmt):
    """Reduce accumulator — declares-and-folds in one statement.

    Semantics: ``name = op(name, value)`` inside the enclosing reduce
    ``Loop``. Before the first iteration ``name`` is initialized to
    ``op.identity`` (the combine's neutral element). After the Loop
    completes, ``name`` is an SSA binding visible in the enclosing scope,
    carrying the finalized reduced value.

    ``op`` is an ``ElementwiseImpl`` — typically one of ``ADD`` / ``MAX`` /
    ``MIN`` / ``MUL``. It defines both the combine operation and the
    accumulator's identity value. Multiple ``Accum`` stmts targeting the
    same ``name`` in one reduce Loop must agree on ``op``.

    Default op is ``add`` — fixtures that sum values can omit ``op=``;
    ``max`` / ``min`` / ``mul`` must be passed explicitly.
    """

    name: str
    value: str
    op: ElementwiseImpl = field(default_factory=lambda: ElementwiseImpl("add"))

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    @property
    def init(self) -> Expr:
        """Identity value for the accumulator (from the op's metadata)."""
        identity = self.op.identity
        return Literal(identity if identity is not None else 0.0)

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} <- {self.op.name}({self.name}, {self.value})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        op_name = self.op.name
        if op_name in ("maximum", "amax"):
            spelling = ctx.intrinsics.get("fmax", "fmax")
            return [f"{pad}{self.name} = {spelling}({self.name}, {self.value});"]
        if op_name == "minimum":
            spelling = ctx.intrinsics.get("fmin", "fmin")
            return [f"{pad}{self.name} = {spelling}({self.name}, {self.value});"]
        op = {"add": "+=", "sum": "+=", "multiply": "*=", "prod": "*="}.get(op_name, "+=")
        return [f"{pad}{self.name} {op} {self.value};"]


@dataclass(frozen=True)
class Init(Stmt):
    """Explicit accumulator initialization at this scope.

    By default, the renderer emits ``float <name> = <identity>;`` above
    a ``Loop`` whose immediate body contains a matching ``Accum``. That
    semantics is wrong when the same ``Accum`` is reduced across multiple
    nested ``Loop``s (e.g. matmul chunked-K: ``Loop(k_o) > Loop(k_i) >
    Accum(acc)`` should not reset ``acc`` per ``k_o`` iteration).

    Placing an ``Init(name, op)`` Stmt at the desired enclosing scope
    declares the accumulator there. The renderer emits the init at this
    point, and suppresses the default Loop-immediate init for any
    ``Accum`` whose name has a matching ``Init`` in an enclosing scope.

    The ``op`` is redundant with the matching ``Accum.op`` (the
    accumulator carries its own combine), but is kept here so the
    renderer can pick the identity without scanning ahead.
    """

    name: str
    op: ElementwiseImpl

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Init({self.name}, op={self.op.name})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        identity = self.op.identity
        if identity is None:
            raise ValueError(f"Init {self.name!r} op {self.op.name!r} has no identity")
        ctx.explicit_inits.add(self.name)
        return [f"{_pad(ctx.indent)}float {self.name} = {_float_lit(float(identity))};"]


# Map ``ElementwiseImpl`` op names to compound-assignment operator symbols
# used by ``Write.pretty()`` for reduce-writes (split-K partial accumulation).
_REDUCE_OP_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


@dataclass(frozen=True)
class Write(Stmt):
    """Write an SSA value to output buffer ``output`` at position ``index``.

    ``output`` is the destination buffer's name (matches the owning graph
    node's id, or — for multi-output kernels — one of its output buffer
    names). ``index`` uses axis Vars to compute the per-dim offset.
    ``value`` references an SSA name available at this point in the body
    (Assign, Accum, or a Load).

    ``reduce_op`` (optional): when set, the write becomes an atomic
    reduction (``atomicAdd`` for ``ElementwiseImpl('add')``) instead of
    a plain store. Used by cross-CTA split-K so multiple CTAs can
    contribute partial sums to the same output cell. Output buffer must
    be zero-initialized by the caller.
    """

    output: str
    index: tuple[Expr, ...]
    value: str
    reduce_op: ElementwiseImpl | None = None

    def __post_init__(self) -> None:
        if self.reduce_op is not None and self.reduce_op.name != "add":
            raise NotImplementedError(f"Write.reduce_op={self.reduce_op.name!r} not lowered yet (only 'add')")

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def has_side_effects(self) -> bool:
        return True

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        if self.reduce_op is not None:
            op = _REDUCE_OP_SYMBOL.get(self.reduce_op.name, self.reduce_op.name)
            return [f"{indent}{self.output}[{idx}] {op}= {self.value}"]
        return [f"{indent}{self.output}[{idx}] = {self.value}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        flat = render_index(self.output, self.index, ctx)
        if self.reduce_op is not None:
            return [f"{_pad(ctx.indent)}atomicAdd(&{self.output}[{flat}], {self.value});"]
        # When the destination buffer is fp16, cast the local float SSA
        # value back to ``__half`` at the store boundary.
        if ctx.buffer_dtypes.get(self.output) == "f16":
            return [f"{_pad(ctx.indent)}{self.output}[{flat}] = __float2half({self.value});"]
        return [f"{_pad(ctx.indent)}{self.output}[{flat}] = {self.value};"]


@dataclass(frozen=True)
class SelectBranch:
    """One branch of a ``Select`` body statement."""

    value: str  # SSA name when predicate holds
    select: Expr  # predicate over axis Vars


@dataclass(frozen=True)
class Select(Stmt):
    """Coord-predicated value binding — replaces Mux.

    At each iteration coord, exactly one branch's ``select`` predicate
    should be True; its ``value`` is bound to ``name`` in the SSA scope.
    Branches are expected to be disjoint; later branches act as
    catch-alls when no earlier predicate matches.
    """

    name: str
    branches: tuple[SelectBranch, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Select.branches must be non-empty")

    def deps(self) -> tuple[str, ...]:
        return tuple(b.value for b in self.branches)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def exprs(self) -> tuple[Expr, ...]:
        return tuple(b.select for b in self.branches)

    def pretty(self, indent: str = "") -> list[str]:
        lines: list[str] = []
        for bi, br in enumerate(self.branches):
            prefix = f"{self.name} =" if bi == 0 else f"{' ' * len(self.name)}  "
            lines.append(f"{indent}{prefix} {br.value} when ({br.select.pretty()})")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        expr = select_to_ternary(self)
        return [f"{_pad(ctx.indent)}float {self.name} = {expr.render(ctx)};"]
