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

_CUDA_TYPE_BY_DTYPE_NAME: dict[str, str] = {"f32": "float", "f16": "__half"}


def _cuda_type(dtype: DataType | None) -> str:
    """C type spelling for an accumulator local. Falls back to ``float``."""
    return _CUDA_TYPE_BY_DTYPE_NAME.get((dtype or F32).name, "float")


def _identity_literal(identity: float, dtype: DataType | None) -> str:
    """Render the per-op identity (0, 1, -inf, ...) as a C literal in ``dtype``.
    Wraps in ``__float2half`` for fp16 so the declaration compiles."""
    txt = _float_lit(float(identity))
    if dtype is not None and dtype.name == "f16":
        return f"__float2half({txt})"
    return txt


def _convert_to(value: str, src_dt: str, dst_dt: str) -> str:
    """Insert a fp16/f32 conversion intrinsic when ``src_dt`` differs from
    ``dst_dt``; bare value otherwise."""
    if src_dt == dst_dt:
        return value
    if dst_dt == "f16" and src_dt == "f32":
        return f"__float2half({value})"
    if dst_dt == "f32" and src_dt == "f16":
        return f"__half2float({value})"
    return value


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
            return [f"{_pad(ctx.indent)}float {self.name} = {_float_lit(lit)};"]
        flat = render_index(self.input, self.index, ctx)
        # Declare the local in the source buffer's element type so
        # downstream ``Assign``s can pick native fp16 ops without an
        # immediate promote-back-to-float. Conversion at load is only
        # needed when the source is fp16 and the chain decides to
        # promote — handled at the use site, not here.
        src_dt = ctx.buffer_dtypes.get(self.input, "f32")
        ctx.ssa_dtypes[self.name] = src_dt
        if src_dt == "f16":
            return [f"{_pad(ctx.indent)}__half {self.name} = {self.input}[{flat}];"]
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
        from deplodock.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

        pad = _pad(ctx.indent)
        op_name = self.op.name
        arg_dtypes = [ctx.ssa_dtypes.get(a, "f32") for a in self.args]
        # Promotion rule for the elementwise scope: result is fp16 only
        # when every input is fp16; any fp32 input promotes the whole
        # expression to fp32 (with __half2float at the use site).
        result_dt = "f16" if arg_dtypes and all(d == "f16" for d in arg_dtypes) else "f32"

        if result_dt == "f16" and op_name in ctx.native_fp16_ops:
            # Native fp16: pick fp16 intrinsic spellings (via a temporary
            # ctx.intrinsics swap) and wrap float literals in
            # ``__float2half`` so they compose with __half operands.
            args: list[Expr] = [Var(a) for a in self.args]
            expr = op_to_expr(op_name, args)
            saved_intr = ctx.intrinsics
            saved_lit = ctx.literal_default_dtype
            ctx.intrinsics = {**saved_intr, **ctx.intrinsics_fp16}
            ctx.literal_default_dtype = "f16"
            try:
                body = expr.render(ctx)
            finally:
                ctx.intrinsics = saved_intr
                ctx.literal_default_dtype = saved_lit
            ctx.ssa_dtypes[self.name] = "f16"
            return [f"{pad}__half {self.name} = {body};"]

        if result_dt == "f16":
            # Fallback: no native fp16 form for this op (or mixed dtypes).
            # Promote each fp16 arg to float at use, render in f32, demote
            # the result back to fp16.
            promoted: list[Expr] = [
                FuncCallExpr("__half2float", [Var(a)]) if dt == "f16" else Var(a) for a, dt in zip(self.args, arg_dtypes, strict=True)
            ]
            expr = op_to_expr(op_name, promoted)
            ctx.ssa_dtypes[self.name] = "f16"
            return [f"{pad}__half {self.name} = __float2half({expr.render(ctx)});"]

        # f32 result: any fp16 inputs get a per-arg ``__half2float`` wrap;
        # everything else stays as today.
        args = [FuncCallExpr("__half2float", [Var(a)]) if dt == "f16" else Var(a) for a, dt in zip(self.args, arg_dtypes, strict=True)]
        expr = op_to_expr(op_name, args)
        ctx.ssa_dtypes[self.name] = "f32"
        return [f"{pad}float {self.name} = {expr.render(ctx)};"]


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
        rhs = _convert_to(self.value, value_dt, acc_dt)
        if op_name in ("maximum", "amax"):
            spelling = ctx.intrinsics.get("fmax", "fmax") if acc_dt == "f32" else ctx.intrinsics_fp16.get("fmax", "__hmax")
            return [f"{pad}{self.name} = {spelling}({self.name}, {rhs});"]
        if op_name == "minimum":
            spelling = ctx.intrinsics.get("fmin", "fmin") if acc_dt == "f32" else ctx.intrinsics_fp16.get("fmin", "__hmin")
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
        return [f"{_pad(ctx.indent)}{_cuda_type(self.dtype)} {self.name} = {_identity_literal(identity, self.dtype)};"]


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
        # Convert at the store boundary only when the value's SSA dtype
        # disagrees with the destination buffer's dtype — native fp16
        # chains write through with no conversion at all.
        value_dt = ctx.ssa_dtypes.get(self.value, "f32")
        out_dt = ctx.buffer_dtypes.get(self.output, "f32")
        rhs = self.value
        if value_dt != out_dt:
            if out_dt == "f16":
                rhs = f"__float2half({self.value})"
            elif out_dt == "f32" and value_dt == "f16":
                rhs = f"__half2float({self.value})"
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
