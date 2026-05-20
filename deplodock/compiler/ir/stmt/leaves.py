"""Leaf ``Stmt`` subclasses — pure compute primitives (no nested bodies).

``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select`` — each
produces / writes a single SSA value. Block-structured stmts (Loop /
Tile / Cond) live in ``blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var, _float_lit
from deplodock.compiler.ir.stmt.base import RenderCtx, Stmt, _pad, op_to_expr, render_index, select_to_ternary


def _args_at_dtype(target, args: tuple[str, ...], arg_dtypes: list[str], dst_dt: str) -> list[Expr]:
    """Convert each SSA arg to ``dst_dt`` via ``target.convert``, returning
    Exprs ready to drop into ``op_to_expr``. Args already at ``dst_dt``
    pass through as bare ``Var``s. Mismatched args are wrapped in the
    target's conversion intrinsic (e.g. ``__half2float(name)``) by
    parsing ``target.convert``'s output back into a ``FuncCallExpr`` so
    it composes with the Expr renderer."""
    from deplodock.compiler.ir.expr import FuncCallExpr  # noqa: PLC0415

    out: list[Expr] = []
    for a, dt in zip(args, arg_dtypes, strict=True):
        if dt == dst_dt:
            out.append(Var(a))
            continue
        converted = target.convert(a, dt, dst_dt)
        paren = converted.index("(") if "(" in converted else -1
        if paren > 0 and converted.endswith(")"):
            out.append(FuncCallExpr(converted[:paren], [Var(a)]))
        else:
            out.append(Var(a))
    return out


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
class Pack(Stmt):
    """Pack two scalar values into one ``__half2`` (or future short-vec).

    ``name`` is the new packed SSA local; ``low`` / ``high`` are the
    two source SSA names. The pack emits ``__halves2half2(low, high)``
    via the target's ``convert``; if ``low`` and ``high`` are already
    fp16, no conversion happens — just the pair-bundle intrinsic. If
    they are fp32, the target inserts ``__float2half`` first.

    Used by the ``__half2``-packing pass to bundle the two scalar
    operands of a paired ``Accum`` (one per lane of the F16x2 pair)
    into a single SSA value the paired Accum can consume.
    """

    name: str
    low: str
    high: str
    dtype: DataType = field(default_factory=lambda: F32)  # F16x2 in real use

    def deps(self) -> tuple[str, ...]:
        return (self.low, self.high)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.dtype.name} {self.name} = pack({self.low}, {self.high})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        low_dt = ctx.ssa_dtypes.get(self.low, "f32")
        high_dt = ctx.ssa_dtypes.get(self.high, "f32")
        # ``__halves2half2`` takes two ``__half``; convert each arg if
        # it isn't already fp16 (the target's convert handles f32→f16).
        low_expr = ctx.target.convert(self.low, low_dt, "f16")
        high_expr = ctx.target.convert(self.high, high_dt, "f16")
        ctx.ssa_dtypes[self.name] = self.dtype.name
        return [f"{_pad(ctx.indent)}{ctx.type_name(self.dtype)} {self.name} = __halves2half2({low_expr}, {high_expr});"]


@dataclass(frozen=True)
class Unpack(Stmt):
    """Split one ``__half2`` SSA value into two scalar ``__half`` names.

    Inverse of :class:`Pack`. Emits one decl line per lane via
    ``__low2half`` / ``__high2half``. Used by the packing pass to
    restore the scalar names that downstream stmts (e.g. WarpShuffle)
    still reference.
    """

    low_name: str
    high_name: str
    value: str  # the f16x2 SSA name being split
    lane_dtype: DataType = field(default_factory=lambda: F32)  # F16 in real use

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.low_name, self.high_name)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.lane_dtype.name} {self.low_name}, {self.high_name} = unpack({self.value})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        ty = ctx.type_name(self.lane_dtype)
        ctx.ssa_dtypes[self.low_name] = self.lane_dtype.name
        ctx.ssa_dtypes[self.high_name] = self.lane_dtype.name
        pad = _pad(ctx.indent)
        return [
            f"{pad}{ty} {self.low_name} = __low2half({self.value});",
            f"{pad}{ty} {self.high_name} = __high2half({self.value});",
        ]


@dataclass(frozen=True)
class VecLoad(Stmt):
    """N consecutive ``Load``s packed into a single vector read.

    Emits ``<vec_type> _v_<n0> = *reinterpret_cast<const <vec_type>*>(...);``
    followed by ``<elem_type> <name_k> = _v_<n0>.<x|y|z|w>;`` unpacks. The
    vector + element C type names come from the target's
    :meth:`RenderTarget.vector_type` for ``(elem_dtype, n)``.

    ``base_index`` is the index of lane 0; lane k's source position is
    ``base_index[:-1] + (base_index[-1] + k,)``. The pass that introduces
    ``VecLoad`` (``003_vectorize_loads``) verifies the consecutive-load
    pattern structurally; ``VecLoad.render`` only needs to format the
    output.
    """

    names: tuple[str, ...]
    input: str
    base_index: tuple[Expr, ...]
    elem_dtype: str  # canonical token: "f32" / "f16"

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return self.names

    def exprs(self) -> tuple[Expr, ...]:
        return self.base_index

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.base_index)
        names = ", ".join(self.names)
        return [f"{indent}{self.elem_dtype} {names} = vec_load[{len(self.names)}] {self.input}[{idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        n = len(self.names)
        vec_pair = ctx.target.vector_type(self.elem_dtype, n)
        pad = _pad(ctx.indent)
        if vec_pair is None:
            # Target doesn't support this width — fall back to scalar
            # Loads. The vectorize pass should have avoided this, but
            # render's job is to always produce valid code.
            out: list[str] = []
            for k, nm in enumerate(self.names):
                idx_k = tuple(self.base_index[:-1]) + (BinaryExpr("+", self.base_index[-1], Literal(k, "int")),)
                flat = render_index(self.input, idx_k, ctx)
                ctx.ssa_dtypes[nm] = self.elem_dtype
                out.append(f"{pad}{ctx.type_name(self.elem_dtype)} {nm} = {self.input}[{flat}];")
            return out
        vec_type, elem_type = vec_pair
        flat = render_index(self.input, self.base_index, ctx)
        vname = f"_v_{self.names[0]}"
        components = ("x", "y", "z", "w")[:n]
        out = [f"{pad}{vec_type} {vname} = *reinterpret_cast<const {vec_type}*>(&{self.input}[{flat}]);"]
        out.extend(f"{pad}{elem_type} {nm} = {vname}.{c};" for nm, c in zip(self.names, components, strict=True))
        for nm in self.names:
            ctx.ssa_dtypes[nm] = self.elem_dtype
        return out


@dataclass(frozen=True)
class Assign(Stmt):
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is an ``ElementwiseImpl`` — the elementwise combine (add /
    mul / exp / ...). Reductions live in ``Accum``. ``args`` reference
    ``$N`` ports, an ``Accum.name`` (reads current / finalized acc
    value), or prior SSA names.

    ``dtype`` is optional and overrides the default ``promote(args)``
    rule when set. The ``demote_to_write_dtype`` Kernel-IR pass stamps
    it on Assigns whose results only feed dtype-narrower consumers, so
    a softmax / RMSNorm epilogue feeding an ``__half*`` Write computes
    in ``__half`` end to end without spurious promotions.
    """

    name: str
    op: ElementwiseImpl
    args: tuple[str, ...]
    dtype: DataType | None = None

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    def deps(self) -> tuple[str, ...]:
        return self.args

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        prefix = f"{self.dtype.name} " if self.dtype is not None else ""
        return [f"{indent}{prefix}{self.name} = {self.op.name}({', '.join(self.args)})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        op_name = self.op.name
        arg_dtypes = [ctx.ssa_dtypes.get(a, "f32") for a in self.args]
        # Default rule: result matches the input dtype when all inputs
        # agree; any fp32 input promotes the whole expression to fp32.
        # Explicit ``self.dtype`` overrides — the demote pass uses this
        # to force a narrower dtype on the result.
        if self.dtype is not None:
            result_dt = self.dtype.name
        else:
            result_dt = "f16" if arg_dtypes and all(d == "f16" for d in arg_dtypes) else "f32"

        all_args_at_result = bool(arg_dtypes) and all(d == result_dt for d in arg_dtypes)
        if result_dt != "f32" and ctx.target.has_native_op(op_name, result_dt) and all_args_at_result:
            # Native non-f32 path: all args already at ``result_dt`` so
            # no per-arg conversion is needed. Render with the target's
            # dtype-specific intrinsics. ``Literal.render`` wraps embedded
            # float literals via the target so they compose with the
            # non-default-dtype operands. When args are *not* all at
            # ``result_dt`` (mixed dtypes from an explicit ``self.dtype``
            # demotion), fall through to the promote-then-demote path
            # below — computing in fp32 and converting once preserves
            # precision better than converting each arg to fp16 first.
            args = _args_at_dtype(ctx.target, self.args, arg_dtypes, result_dt)
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

        # f32 / no-native path: promote args to f32, render in f32, then
        # convert the result back to ``result_dt`` when narrower.
        promoted = _promote_args_to_f32(ctx.target, self.args, arg_dtypes)
        expr = op_to_expr(op_name, promoted)
        body_str = expr.render(ctx)
        if result_dt != "f32":
            body_str = ctx.target.convert(body_str, "f32", result_dt)
        ctx.ssa_dtypes[self.name] = result_dt
        return [f"{pad}{ctx.type_name(result_dt)} {self.name} = {body_str};"]


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
        prefix = f"{self.dtype.name} " if self.dtype is not None else ""
        return [f"{indent}{prefix}{self.name} <- {self.op.name}({self.name}, {self.value})"]

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
        return [f"{indent}Init({self.dtype.name} {self.name}, op={self.op.name})"]

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
