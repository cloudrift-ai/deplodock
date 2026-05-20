"""Leaf ``Stmt`` subclasses — pure compute primitives (no nested bodies).

``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select`` — each
produces / writes a single SSA value. Block-structured stmts (Loop /
Tile / Cond) live in ``blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr, Literal, Var, _float_lit
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, _pad, op_to_expr, render_index, select_to_ternary


def _promote_args_to_f32(target, args: tuple[str, ...], arg_dtypes: list[str]) -> list[Expr]:
    """Wrap each non-f32 SSA name with ``target.convert(name, dt, "f32")``
    so the resulting Expr tree composes in fp32. Used by ``Assign.render``
    for the f32 result path and the f16-fallback path.

    Returns an ``Expr`` list, with conversions threaded through a
    ``FuncCallExpr``-style ``Cast``: we synthesize a no-op
    :class:`Var` for f32 args and an inline-rendered cast for others."""
    from deplodock.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

    out: list[Expr] = []
    for a, dt in zip(args, arg_dtypes, strict=True):
        if dt == "f32":
            out.append(Var(a))
        else:
            # ``target.convert`` returns a fully-rendered string; wrap as
            # a synthetic FuncCallExpr so it slots into ``op_to_expr``'s
            # Expr-tree output without round-tripping through Var lookups.
            converted = target.convert(a, dt, "f32")
            # Strip the synthesized prefix to reuse FuncCallExpr's
            # "name(args)" rendering. ``target.convert("x", "f16", "f32")``
            # returns ``"__half2float(x)"`` — we split on the open paren.
            paren = converted.index("(") if "(" in converted else -1
            if paren > 0 and converted.endswith(")"):
                out.append(FuncCallExpr(converted[:paren], [Var(a)]))
            else:
                out.append(Var(a))
    return out


def _dtype_intrinsics(target, result_dt: str, expr: Expr) -> dict[str, str]:
    """Per-dtype intrinsic overrides for the abstract op names that
    appear inside ``expr``. The Expr renderer reads ``ctx.intrinsics``
    when resolving ``FuncCallExpr.name`` to a target spelling; we patch
    in the dtype-specific spellings while rendering the fp16-native
    path."""
    from deplodock.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

    overrides: dict[str, str] = {}

    def collect(node):
        if isinstance(node, FuncCallExpr):
            overrides[node.name] = target.intrinsic(node.name, result_dt)
            for a in node.args:
                collect(a)
        else:
            # Generic Expr walker — the public API is limited; rely on
            # the common subclasses' field names.
            for field_name in ("left", "right", "cond", "if_true", "if_false", "expr"):
                child = getattr(node, field_name, None)
                if child is not None:
                    collect(child)
            args = getattr(node, "args", None)
            if isinstance(args, list):
                for a in args:
                    collect(a)

    collect(expr)
    return overrides


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
        # Inlined scalar constants stay as ``float`` locals — the
        # consumer's ``Assign.render`` will demote / convert if needed.
        lit = ctx.literal_constants.get(self.input) if ctx.literal_constants else None
        if lit is not None:
            ctx.ssa_dtypes[self.name] = "f32"
            return [f"{_pad(ctx.indent)}{ctx.type_name('f32')} {self.name} = {_float_lit(lit)};"]
        flat = render_index(self.input, self.index, ctx)
        # Declare the local in the source buffer's element type so
        # downstream ``Assign``s can pick native ops without an
        # immediate promote. Conversion is handled at the use site
        # when an Assign / Write needs a different dtype.
        src_dt = ctx.buffer_dtypes.get(self.input, "f32")
        ctx.ssa_dtypes[self.name] = src_dt
        return [f"{_pad(ctx.indent)}{ctx.type_name(src_dt)} {self.name} = {self.input}[{flat}];"]


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
        pad = _pad(ctx.indent)
        op_name = self.op.name
        arg_dtypes = [ctx.ssa_dtypes.get(a, "f32") for a in self.args]
        # Promotion rule for the elementwise scope: result matches the
        # input dtype when all inputs agree; any non-default fp32 input
        # promotes the whole expression to fp32 (with conversion at the
        # use site).
        result_dt = "f16" if arg_dtypes and all(d == "f16" for d in arg_dtypes) else "f32"

        if result_dt != "f32" and ctx.target.has_native_op(op_name, result_dt):
            # Native non-f32 path: swap the f32 intrinsic table for the
            # target's dtype-specific table around this render, and tell
            # ``Literal.render`` to wrap embedded float literals in the
            # target's dtype-cast intrinsic so they compose with the
            # non-default-dtype operands.
            args: list[Expr] = [Var(a) for a in self.args]
            expr = op_to_expr(op_name, args)
            saved_intr = ctx.intrinsics
            saved_lit = ctx.literal_default_dtype
            ctx.intrinsics = {**saved_intr, **_dtype_intrinsics(ctx.target, result_dt, expr)}
            ctx.literal_default_dtype = result_dt
            try:
                body = expr.render(ctx)
            finally:
                ctx.intrinsics = saved_intr
                ctx.literal_default_dtype = saved_lit
            ctx.ssa_dtypes[self.name] = result_dt
            return [f"{pad}{ctx.type_name(result_dt)} {self.name} = {body};"]

        if result_dt != "f32":
            # Fallback: no native form for this op at result_dt.
            # Promote each non-f32 arg to f32 at use, render in f32, then
            # convert the result back to the declared dtype.
            promoted = _promote_args_to_f32(ctx.target, self.args, arg_dtypes)
            expr = op_to_expr(op_name, promoted)
            ctx.ssa_dtypes[self.name] = result_dt
            body = ctx.target.convert(expr.render(ctx), "f32", result_dt)
            return [f"{pad}{ctx.type_name(result_dt)} {self.name} = {body};"]

        # f32 result: any non-f32 inputs get a per-arg conversion wrap.
        args = _promote_args_to_f32(ctx.target, self.args, arg_dtypes)
        expr = op_to_expr(op_name, args)
        ctx.ssa_dtypes[self.name] = "f32"
        return [f"{pad}{ctx.type_name('f32')} {self.name} = {expr.render(ctx)};"]


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
    # Optional accumulator dtype. ``None`` in Loop IR — derived at lowering
    # time. The Init-placement pass freezes this to a concrete
    # :class:`DataType` (typically ``F32`` for fp16 reductions). When set,
    # ``Accum.render`` declares the local in that dtype and inserts a
    # ``__half2float`` / ``__float2half`` on ``value`` only when ``value``'s
    # dtype disagrees with the accumulator's. ``None`` preserves the legacy
    # f32 rendering.
    dtype: DataType | None = None

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
        dt = f" :{self.dtype.name}" if self.dtype is not None else ""
        return [f"{indent}{self.name}{dt} <- {self.op.name}({self.name}, {self.value})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        op_name = self.op.name
        # Accumulator dtype — explicit on Accum once the Init-placement pass
        # has frozen it; otherwise default to fp32 (legacy behavior).
        acc_dt = (self.dtype or F32).name
        ctx.ssa_dtypes[self.name] = acc_dt
        value_dt = ctx.ssa_dtypes.get(self.value, "f32")
        rhs = ctx.target.convert(self.value, value_dt, acc_dt)
        if op_name in ("maximum", "amax"):
            spelling = ctx.target.intrinsic("fmax", acc_dt)
            return [f"{pad}{self.name} = {spelling}({self.name}, {rhs});"]
        if op_name == "minimum":
            spelling = ctx.target.intrinsic("fmin", acc_dt)
            return [f"{pad}{self.name} = {spelling}({self.name}, {rhs});"]
        op = {"add": "+=", "sum": "+=", "multiply": "*=", "prod": "*="}.get(op_name, "+=")
        return [f"{pad}{self.name} {op} {rhs};"]


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
    # Accumulator dtype — required. Placing an ``Init`` is the freeze
    # point; the pass that emits it must commit to a concrete dtype. The
    # same pass stamps the matching ``Accum``'s ``dtype`` to this value
    # so the IR stays self-consistent.
    dtype: DataType = field(kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))
        if isinstance(self.dtype, str):
            from deplodock.compiler.dtype import get as _get  # noqa: PLC0415

            object.__setattr__(self, "dtype", _get(self.dtype))

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Init({self.name} :{self.dtype.name}, op={self.op.name})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        identity = self.op.identity
        if identity is None:
            raise ValueError(f"Init {self.name!r} op {self.op.name!r} has no identity")
        ctx.explicit_inits.add(self.name)
        ctx.ssa_dtypes[self.name] = self.dtype.name
        return [f"{_pad(ctx.indent)}{ctx.type_name(self.dtype)} {self.name} = {ctx.identity_literal(identity, self.dtype)};"]


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
        # Convert at the store boundary only when the value's SSA dtype
        # disagrees with the destination buffer's dtype — native chains
        # write through with no conversion. Applies to both plain stores
        # and atomic reduce-writes (split-K matmul fans an f32
        # accumulator into an f16 output; without conversion the
        # ``atomicAdd(__half*, float)`` call is silently broken).
        value_dt = ctx.ssa_dtypes.get(self.value, "f32")
        out_dt = ctx.buffer_dtypes.get(self.output, "f32")
        rhs = ctx.target.convert(self.value, value_dt, out_dt)
        if self.reduce_op is not None:
            return [f"{_pad(ctx.indent)}atomicAdd(&{self.output}[{flat}], {rhs});"]
        return [f"{_pad(ctx.indent)}{self.output}[{flat}] = {rhs};"]


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
