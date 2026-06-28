"""Leaf ``Stmt`` subclasses — pure compute primitives (no nested bodies).

``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select`` — each
produces / writes a single SSA value. Block-structured stmts (Loop /
Tile / Cond) live in ``blocks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.elementwise import ElementwiseImpl, reduce_spelling
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var, _float_lit
from deplodock.compiler.ir.stmt.base import (
    RenderCtx,
    Stmt,
    _pad,
    dtype_promote,
    op_to_expr,
    render_index,
    select_to_ternary,
)

if TYPE_CHECKING:  # annotation only — algebra imports leaves (Accum.as_monoid does the runtime import)
    from deplodock.compiler.ir.stmt.algebra import Monoid


def _resolve_value(name: str, ctx: RenderCtx) -> str:
    """If ``name`` is an SSA name bound by a skipped literal-constant Load,
    return its rendered float-literal form; otherwise return ``name``
    unchanged. ``render_body`` drops the defining Load of every
    literal-constant SSA (its value gets inlined at use sites), so
    Stmts that carry raw SSA-name strings (``Write.values`` / ``Pack.low`` /
    ``Accum.value``) must substitute here instead of emitting an
    undefined identifier."""
    lit = ctx.literal_ssa.get(name) if ctx.literal_ssa else None
    return _float_lit(lit) if lit is not None else name


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
                out.append(FuncCallExpr(converted[:paren], (Var(a),)))
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


@dataclass(frozen=True, init=False)
class Load(Stmt):
    """Read a value (or N consecutive values) from an external input buffer.

    Scalar (``width == 1``): one SSA binding ``names[0]``. Vector
    (``width > 1``): N SSA bindings; lane k reads
    ``index[:-1] + (index[-1] + k,)`` into ``names[k]``. The
    ``050_vectorize_loads`` pass widens a run of consecutive scalar
    Loads into one Load with ``len(names) > 1``. Every pass before that
    produces scalar Loads and can keep using ``s.name`` / ``s.index``.

    The constructor accepts either ``name="x"`` (scalar shorthand,
    normalized to ``names=("x",)``) or ``names=(...)`` directly. Use
    ``.is_vector`` / ``.is_scalar`` / ``.width`` / ``.name`` (asserts
    scalar) / ``.names`` to test shape.

    A scalar Load is rendered as a literal binding (``float name =
    <value>;``) when ``ctx.literal_constants`` carries a value for
    ``input`` — the scalar-constant-inlining path populates that map at
    the cuda lowering boundary so kernels can embed ``ConstantOp``
    values directly instead of taking them as ``float*`` parameters.

    ``dtype`` is optional and, when set, names the source-buffer element
    dtype that this Load produces. The ``030_stamp_types`` Kernel-IR pass
    populates it once before any analytical pass runs; downstream passes
    (vectorize_loads, demote, etc.) read it instead of reaching for the
    matcher-populated ``KernelOp.inputs``/``outputs`` side channels.
    ``None`` keeps the legacy render-time inference path working for
    tests that construct Loads by hand without dtype.
    """

    names: tuple[str, ...]
    input: str
    index: tuple[Expr, ...]
    dtype: DataType | None

    def __init__(
        self,
        name: str | None = None,
        input: str | None = None,
        index: tuple[Expr, ...] | None = None,
        *,
        names: tuple[str, ...] | None = None,
        dtype: DataType | None = None,
    ) -> None:
        if names is None:
            if name is None:
                raise TypeError("Load requires either `name=` (scalar) or `names=` (vector)")
            names = (name,)
        elif name is not None:
            raise TypeError("Load: pass `name=` xor `names=`, not both")
        if input is None:
            raise TypeError("Load requires `input=`")
        if index is None:
            raise TypeError("Load requires `index=`")
        if not names:
            raise ValueError("Load.names must be non-empty")
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "index", tuple(index))
        object.__setattr__(self, "dtype", dtype)

    @property
    def name(self) -> str:
        """Scalar-only shorthand for ``names[0]``. Asserts ``is_scalar`` —
        use ``.names`` (or guard with ``.is_scalar``) for vector Loads."""
        assert self.is_scalar, f"Load has {self.width} names — use .names for vector Loads"
        return self.names[0]

    @property
    def width(self) -> int:
        return len(self.names)

    @property
    def is_vector(self) -> bool:
        return len(self.names) > 1

    @property
    def is_scalar(self) -> bool:
        return len(self.names) == 1

    def deps(self) -> tuple[str, ...]:
        # SSA names this Load reads through its *index* — a data-dependent
        # (gather) index like ``weight[(int)in0, a]`` reads ``in0``. Returns
        # the index Exprs' free Vars (first-use order, de-duped). Axis-name
        # Vars (``a0``, ``a1``) appear too, but every consumer resolves deps
        # against the body's ``definitions``, where axes are absent — so they
        # pass through as no-ops, exactly as ``_rename_ssa_vars_in_expr`` and
        # the dataflow helpers already assume.
        return tuple(dict.fromkeys(v for e in self.index for v in e.free_vars()))

    def defines(self) -> tuple[str, ...]:
        return self.names

    def external_reads(self) -> tuple[str, ...]:
        return (self.input,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def is_literal(self, literal_constants: dict[str, float]) -> bool:
        return self.input in literal_constants

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        names = ", ".join(self.names)
        return [f"{indent}{names} = load {self.input}[{idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        # Inlined scalar constants stay as ``float`` locals — the
        # consumer's ``Assign.render`` will demote / convert if needed.
        # (Only valid for scalar Loads; the vectorize pass excludes
        # literal-const buffers.)
        lit = ctx.literal_constants.get(self.input) if ctx.literal_constants else None
        if lit is not None and self.is_scalar:
            ctx.ssa_dtypes[self.names[0]] = "f32"
            return [f"{pad}{ctx.type_name('f32')} {self.names[0]} = {_float_lit(lit)};"]
        # Prefer the stamped ``self.dtype`` (set by ``030_stamp_types``);
        # fall back to ``ctx.buffer_dtypes`` so handwritten test fixtures
        # without a stamped dtype still render correctly.
        src_dt = self.dtype.name if self.dtype is not None else ctx.buffer_dtypes.get(self.input, "f32")
        if self.is_scalar:
            # Scalar path. Declare the local in the source buffer's
            # element type so downstream ``Assign``s can pick native ops
            # without an immediate promote.
            flat = render_index(self.input, self.index, ctx)
            ctx.ssa_dtypes[self.names[0]] = src_dt
            return [f"{pad}{ctx.type_name(src_dt)} {self.names[0]} = {self.input}[{flat}];"]
        # Vector path: one ``<vec_type>`` reinterpret-cast read + N
        # ``.x/.y/.z/.w`` (or indexed) unpacks.
        n = self.width
        vec_pair = ctx.target.vector_type(src_dt, n)
        if vec_pair is None:
            # Target doesn't support this width — fall back to scalar
            # Loads. The vectorize pass should have avoided this, but
            # render's job is to always produce valid code.
            out: list[str] = []
            for k, nm in enumerate(self.names):
                idx_k = tuple(self.index[:-1]) + (BinaryExpr("+", self.index[-1], Literal(k, "int")),)
                flat = render_index(self.input, idx_k, ctx)
                ctx.ssa_dtypes[nm] = src_dt
                out.append(f"{pad}{ctx.type_name(src_dt)} {nm} = {self.input}[{flat}];")
            return out
        vec_type, elem_type = vec_pair
        flat = render_index(self.input, self.index, ctx)
        vname = f"_v_{self.names[0]}"
        # ``.x/.y/.z/.w`` accessors only work when ``vec_type``'s native
        # components match ``elem_type`` 1:1 (``float2``→``float``,
        # ``__half2``→``__half``). For wider packed vectors that
        # reinterpret a multi-element pack (``uint2``→4 halves,
        # ``uint4``→8 halves), each ``.x``/``.y`` slot holds multiple
        # elements, so we fall back to array-style indexing through a
        # reinterpret-cast.
        native_n = {"float2": 2, "float4": 4, "__half2": 2}.get(vec_type)
        use_array_index = native_n != n
        out_lines = [f"{pad}{vec_type} {vname} = *reinterpret_cast<const {vec_type}*>(&{self.input}[{flat}]);"]
        if use_array_index:
            arr_name = f"{vname}_h"
            out_lines.append(f"{pad}const {elem_type}* {arr_name} = reinterpret_cast<const {elem_type}*>(&{vname});")
            out_lines.extend(f"{pad}{elem_type} {nm} = {arr_name}[{k}];" for k, nm in enumerate(self.names))
        else:
            components = ("x", "y", "z", "w")[:n]
            out_lines.extend(f"{pad}{elem_type} {nm} = {vname}.{c};" for nm, c in zip(self.names, components, strict=True))
        for nm in self.names:
            ctx.ssa_dtypes[nm] = src_dt
        return out_lines


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
            result_dt = dtype_promote(op_name, arg_dtypes)

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

    ``axes`` is the tuple of axis names this Accum reduces over —
    populated by the lifting pass (one axis name from the wrapping reduce
    ``Loop``) and propagated by every rewrite that renames axes via
    Sigma. Used by the escape-analysis helper to derive cross-thread
    cooperativity (``Accum.axes ∩ enclosing ThreadTile.axes``); empty
    tuple = legacy/handwritten path with no reduction-axis info.
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
    axes: tuple[str, ...] = ()
    # Optional rescaled base — the value the fold reads as its left operand instead
    # of ``name`` (``name = op(base, value)``). ``None`` = the ordinary self-fold
    # ``name = op(name, value)``. A twisted carrier's streaming merge lowers each
    # state component to a ``base``-``Accum``: the ψ rescale is a preceding ``Assign``
    # binding ``base`` (e.g. ``lm = l·alpha``), so the fold itself stays an ``Accum``
    # whose seed is still ``op.identity``. The seed (and ``Loop.render``'s per-Accum
    # init) is unchanged — ``base`` only redirects the in-loop left operand.
    base: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    @property
    def init(self) -> Expr:
        """Identity value for the accumulator (from the op's metadata)."""
        identity = self.op.identity
        return Literal(identity if identity is not None else 0.0)

    def deps(self) -> tuple[str, ...]:
        # ``base`` (when it redirects the left operand) is a same-scope read; the
        # carried ``name`` read is implicit (loop-carried), like the default fold.
        if self.base is not None and self.base != self.name:
            return (self.value, self.base)
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def carried_names(self) -> tuple[str, ...]:
        return (self.name,)

    def combine_operands(self) -> tuple[str, ...]:
        return (f"{self.name}__o",)

    def combine_partials(self) -> tuple[Assign, ...]:
        """The scalar op-fold of two partials: ``name = op(name, name__o)`` — the
        same combine the cooperative / split-K realizations apply, reified as a
        one-``Assign`` program so the decomposition move reads it uniformly with
        ``Monoid.combine_states``."""
        return (Assign(name=self.name, op=self.op, args=(self.name, f"{self.name}__o"), dtype=self.dtype),)

    def as_monoid(self) -> Monoid:
        """This additive/associative ``Accum`` AS the degenerate 1-component ``Monoid`` it already is —
        state ``(name,)``, partial ``(value,)``, ``merge`` = ``name = op(name, value)``, identity the
        op's. The carrier-algebra fact that a SEMIRING / scalar reduce is the trivial monoid: it lets an
        ``Accum`` lower through the **same** ``Monoid.render`` / cross-partition path as a general
        ``Monoid``, with no additive special-case. The auto-derived ``combine_states``
        (``name = op(name, name__o)``) equals :meth:`combine_partials`, so the ``⊙`` realization is identical."""
        from deplodock.compiler.ir.stmt.algebra import Monoid, State, Twist  # local: algebra imports leaves

        # A loop-IR carrier — ``partial=()``; the folded ``value`` is a sibling whose name
        # lives in the degenerate ``merge`` (``name = op(name, value)``).
        return Monoid(
            state=State(names=(self.name,), identity=(self.init,)),
            partial=(),
            twist=Twist.degenerate((self.name,), (self.value,), (self.op,), self.dtype),
        )

    # Algebraic traits forward to the scalar combine op — a ``max`` Accum and a
    # ``sum`` Accum differ, and ``self.op`` is the source of truth.
    @property
    def associative(self) -> bool:
        return self.op.associative

    @property
    def commutative(self) -> bool:
        return self.op.commutative

    @property
    def has_identity(self) -> bool:
        return self.op.has_identity

    def pretty(self, indent: str = "") -> list[str]:
        prefix = f"{self.dtype.name} " if self.dtype is not None else ""
        base = self.base if self.base is not None else self.name
        return [f"{indent}{prefix}{self.name} <- {self.op.name}({base}, {self.value})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        # Accumulator dtype — explicit on Accum once the dtype policy
        # has frozen it; otherwise default to fp32 (legacy behavior).
        acc_dt = (self.dtype or F32).name
        ctx.ssa_dtypes[self.name] = acc_dt
        value_dt = ctx.ssa_dtypes.get(self.value, "f32")
        rhs = ctx.target.convert(self.value, value_dt, acc_dt)
        # Left operand of the fold: ``base`` (a rescaled state, already at acc_dt)
        # when set, else the carried ``name`` itself.
        base = self.base if self.base is not None else self.name
        # Spelling (``+=`` / ``*=`` / ``fmax`` / ``fmin``) from the shared reduce
        # registry; defaults to additive for non-reduce ops.
        spelling = reduce_spelling(self.op)
        if spelling.intrinsic is not None:
            fn = ctx.target.intrinsic(spelling.intrinsic, acc_dt)
            return [f"{pad}{self.name} = {fn}({base}, {rhs});"]
        if base == self.name:
            return [f"{pad}{self.name} {spelling.compound} {rhs};"]
        return [f"{pad}{self.name} = {base} {spelling.infix} {rhs};"]


@dataclass(frozen=True)
class Mma(Stmt):
    """Tensor-core multiply-accumulate over one atom cell — ``c += a @ b``.

    The fused replacement for the scalar ``Assign(multiply) + Accum`` matmul
    cell on the tensor-core path. Emitted by ``tile/enumeration/050_warp_build``
    alongside its two operand ``Load``s — which stay **plain** (no tensor-core
    tag): the ``Mma`` is the sole carrier of the cell's :class:`Atom` spec +
    operand identity, naming its A/B operands by SSA value, so
    ``kernel/005_lower_atom_tile`` reads the spec straight off the ``Mma`` and
    recovers each operand Load's role from it. Carried through the staging
    passes (it makes its reduce loop ``is_reduce`` just like an ``Accum``), and
    lowered to a kernel-IR ``MmaSyncPtx``.

    - ``c`` — the accumulator SSA name (declared + zero-init'd as the fp32 c
      fragment at lowering); read-and-written, like ``Accum.name``.
    - ``a`` / ``b`` — the SSA names of the two operand ``Load``s (A = M×K,
      B = K×N); the lowering matches each Load by these names.
    - ``atom`` — the ``Atom`` spec itself (cell shape + per-operand dtypes +
      group size); a hashable frozen record, so it rides on this frozen ``Mma``
      Stmt. NOTE: the tile-IR ``Atom`` type was demolished — the annotation is a
      bare ``object`` placeholder pending the tile-IR rebuild.
    - ``axes`` — the reduction axes (mirrors ``Accum.axes``; carries the
      cooperative-K info the escape analysis reads). Threaded through
      ``rewrite`` like ``Accum.axes``.
    - ``b_trans`` — the B operand is stored N×K (K in its last dim), i.e. a
      transposed-B ``Q @ K^T`` cell. This is the native ``mma.row.col`` B layout
      (col-major K×N), so ``kernel/005_lower_atom_tile`` loads it via ``ldmatrix``
      WITHOUT ``.trans`` (the default canonical B[k,n] uses ``.trans``). Set by
      ``tile/enumeration/050_warp_build`` from the classified B Load's K position.
    """

    c: str
    a: str
    b: str
    atom: object
    axes: tuple[str, ...] = ()
    b_trans: bool = False
    # Explicit masked-tile guards for a HAND-BUILT cell (the symbolic warp-chain flash),
    # where ``kernel/005_lower_atom_tile`` can't derive them from a Write boundary ``Cond``
    # (a fragment-output / fragment-A cell has no Write) or the operand tensor shape (the
    # flash uses flat single-index Loads). Each is ``(base Expr, bound Expr)`` on the named
    # axis; ``005`` routes them to the operand ``LdmatrixLoad``s — ``m_guard`` clamps the A
    # rows (masked query), ``n_guard`` clamps the B cols (masked key, transposed-B), ``k_zero``
    # zero-fills the B reduce rows past ``bound`` (masked-K P@V). ``None`` = the enumeration
    # σ-split path, where ``005`` derives guards as before.
    m_guard: tuple[Expr, Expr] | None = None
    n_guard: tuple[Expr, Expr] | None = None
    k_zero: tuple[Expr, Expr] | None = None

    def deps(self) -> tuple[str, ...]:
        # Mirror ``Accum``: the accumulator read is implicit (loop-carried),
        # so only the operands are listed — keeps sibling-def analyses (topo
        # sort, reg-pipeline) from treating ``c`` as a same-scope read.
        return (self.a, self.b)

    def defines(self) -> tuple[str, ...]:
        return (self.c,)

    def carried_names(self) -> tuple[str, ...]:
        return (self.c,)

    def combine_operands(self) -> tuple[str, ...]:
        return (f"{self.c}__o",)

    def combine_partials(self) -> tuple[Assign, ...]:
        """The fragment add of two partial accumulators: ``c = c + c__o`` — the
        cross-CTA split-K combine, reified as a one-``Assign`` program (the
        accumulation is additive, so the fold op is ``add`` regardless of the
        original scalar cell)."""
        return (Assign(name=self.c, op=ElementwiseImpl("add"), args=(self.c, f"{self.c}__o")),)

    # The tensor-core accumulation ``c += a @ b`` is an additive fold —
    # associative + commutative with identity 0. That is what tells
    # reassociation gates split-K over the matmul's K axis is legal, exactly as
    # for a scalar ``sum`` Accum (there is no scalar op to point at, so the
    # traits are reported as constants).
    @property
    def associative(self) -> bool:
        return True

    @property
    def commutative(self) -> bool:
        return True

    @property
    def has_identity(self) -> bool:
        return True

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.c} <- mma[{self.atom.name}]({self.a} @ {self.b})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            f"Mma must be consumed by kernel/005_lower_atom_tile before render — reached render with atom={self.atom.name!r}"
        )


@dataclass(frozen=True)
class Init(Stmt):
    """Explicit accumulator / carried-state seed at this scope:
    ``<dtype> <name> = <identity>;`` — a scope-local declaration.

    Currently UNPRODUCED — a carrier's seed now rides on its fold and is derived by
    ``Loop.render`` (an ``Accum`` from ``op.identity``, a ``Monoid`` carrier from
    ``State.identity``), so no pass emits an explicit ``Init``. Kept as a primitive
    (with its render / rewrite / validation handlers) for an explicit cross-scope seed
    the cooperative / split-K reduce tier may want — e.g. a chunked-K accumulator that
    must seed above the outer loop and NOT reset per chunk. ``identity`` is the neutral
    element (one scalar — 0 / 1 / -inf), held directly.
    """

    name: str
    identity: float
    # Accumulator / state dtype — required. Placing an ``Init`` is the freeze
    # point; the pass that emits it must commit to a concrete dtype. The
    # same pass stamps the matching ``Accum``'s ``dtype`` to this value
    # so the IR stays self-consistent.
    dtype: DataType = field(kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            from deplodock.compiler.dtype import get as _get  # noqa: PLC0415

            object.__setattr__(self, "dtype", _get(self.dtype))

    def deps(self) -> tuple[str, ...]:
        return ()

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Init({self.dtype.name} {self.name} = {self.identity})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        ctx.ssa_dtypes[self.name] = self.dtype.name
        return [f"{_pad(ctx.indent)}{ctx.type_name(self.dtype)} {self.name} = {ctx.identity_literal(self.identity, self.dtype)};"]


# Map ``ElementwiseImpl`` op names to compound-assignment operator symbols
# used by ``Write.pretty()`` for reduce-writes (split-K partial accumulation).
_REDUCE_OP_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


@dataclass(frozen=True, init=False)
class Write(Stmt):
    """Write an SSA value (or N consecutive values) to ``output``.

    Scalar (``width == 1``): stores one SSA value at ``index``. Vector
    (``width > 1``): stores N values; ``values[0]`` goes to ``index``,
    ``values[k]`` goes to ``index[:-1] + (index[-1] + k,)``. The
    ``080_vectorize_stores`` pass widens a run of consecutive scalar
    Writes into one Write with ``len(values) > 1``. Every pass before
    that produces scalar Writes and can keep using ``s.value`` /
    ``s.index``.

    The constructor accepts either ``value="v"`` (scalar shorthand,
    normalized to ``values=("v",)``) or ``values=(...)`` directly. Use
    ``.is_vector`` / ``.is_scalar`` / ``.width`` / ``.value`` (asserts
    scalar) / ``.values`` to test shape.

    ``output`` is the destination buffer's name (matches the owning graph
    node's id, or — for multi-output kernels — one of its output buffer
    names). ``index`` uses axis Vars to compute the per-dim offset.

    ``value_dtype`` is optional; when set, names the SSA-value dtype being
    stored. Stamped by ``030_stamp_types``; downstream passes read it
    instead of querying ``ctx.ssa_dtypes`` at render time.
    """

    output: str
    index: tuple[Expr, ...]
    values: tuple[str, ...]
    value_dtype: DataType | None

    def __init__(
        self,
        output: str | None = None,
        index: tuple[Expr, ...] | None = None,
        value: str | None = None,
        *,
        values: tuple[str, ...] | None = None,
        value_dtype: DataType | None = None,
    ) -> None:
        if values is None:
            if value is None:
                raise TypeError("Write requires either `value=` (scalar) or `values=` (vector)")
            values = (value,)
        elif value is not None:
            raise TypeError("Write: pass `value=` xor `values=`, not both")
        if output is None:
            raise TypeError("Write requires `output=`")
        if index is None:
            raise TypeError("Write requires `index=`")
        if not values:
            raise ValueError("Write.values must be non-empty")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "index", tuple(index))
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "value_dtype", value_dtype)

    @property
    def value(self) -> str:
        """Scalar-only shorthand for ``values[0]``. Asserts ``is_scalar`` —
        use ``.values`` (or guard with ``.is_scalar``) for vector Writes."""
        assert self.is_scalar, f"Write has {self.width} values — use .values for vector Writes"
        return self.values[0]

    @property
    def width(self) -> int:
        return len(self.values)

    @property
    def is_vector(self) -> bool:
        return len(self.values) > 1

    @property
    def is_scalar(self) -> bool:
        return len(self.values) == 1

    def deps(self) -> tuple[str, ...]:
        return self.values

    def external_writes(self) -> tuple[str, ...]:
        return (self.output,)

    def exprs(self) -> tuple[Expr, ...]:
        return self.index

    def has_side_effects(self) -> bool:
        return True

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        if self.is_vector:
            return [f"{indent}{self.output}[{idx}] = ({', '.join(self.values)})"]
        return [f"{indent}{self.output}[{idx}] = {self.values[0]}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        out_dt = ctx.buffer_dtypes.get(self.output, "f32")
        # Prefer the stamped ``self.value_dtype`` (set by ``030_stamp_types``);
        # fall back to ``ctx.ssa_dtypes`` for legacy/handwritten paths.
        stamped_value_dt = self.value_dtype.name if self.value_dtype is not None else None
        if self.is_scalar:
            # Scalar path. Convert at the store boundary only when the
            # value's SSA dtype disagrees with the destination buffer's
            # dtype — native chains write through with no conversion.
            flat = render_index(self.output, self.index, ctx)
            value_dt = stamped_value_dt or ctx.ssa_dtypes.get(self.value, "f32")
            rhs = ctx.target.convert(_resolve_value(self.value, ctx), value_dt, out_dt)
            return [f"{pad}{self.output}[{flat}] = {rhs};"]
        # Vectorized path. Per-value dtype conversion: every SSA arg
        # must be at ``out_dt`` before packing.
        n = self.width
        converted: list[str] = []
        for nm in self.values:
            src_dt = stamped_value_dt or ctx.ssa_dtypes.get(nm, "f32")
            resolved = _resolve_value(nm, ctx)
            converted.append(resolved if src_dt == out_dt else ctx.target.convert(resolved, src_dt, out_dt))
        vec_pair = ctx.target.vector_type(out_dt, n)
        if vec_pair is None:
            # Target doesn't support this width — fall back to scalar
            # writes. The vectorize pass should have avoided this, but
            # render's job is to always produce valid code.
            lines: list[str] = []
            for k in range(n):
                idx_k = tuple(self.index[:-1]) + (BinaryExpr("+", self.index[-1], Literal(k, "int")),)
                flat = render_index(self.output, idx_k, ctx)
                lines.append(f"{pad}{self.output}[{flat}] = {converted[k]};")
            return lines
        vec_type, _elem_type = vec_pair
        flat = render_index(self.output, self.index, ctx)
        # Native-width vectors (``float2`` / ``float4`` / ``__half2``) take
        # a positional constructor — ``make_float2(a, b)`` for fp32 paths,
        # ``__halves2half2(a, b)`` for the fp16 pair. Wider packed vectors
        # (``uint2`` / ``uint4``) re-interpret arrays of elements — we
        # stage the elements into a local array, then store the array's
        # uint{2,4} view in one transaction.
        # ``id(self)`` disambiguates the temp name across sibling Writes
        # that read from the same SSA — e.g. broadcasting a shared
        # literal-constant value (where every Write's ``values[0]`` is the
        # same SSA name) used to collide on ``_vs_<name>``.
        temp = f"_vs_{self.values[0]}_{id(self) & 0xFFFF:04x}"
        if vec_type == "__half2":
            return [
                f"{pad}{vec_type} {temp} = __halves2half2({converted[0]}, {converted[1]});",
                f"{pad}*reinterpret_cast<{vec_type}*>(&{self.output}[{flat}]) = {temp};",
            ]
        if vec_type in ("float2", "float4"):
            args = ", ".join(converted)
            return [
                f"{pad}{vec_type} {temp} = make_{vec_type}({args});",
                f"{pad}*reinterpret_cast<{vec_type}*>(&{self.output}[{flat}]) = {temp};",
            ]
        # Packed widths (uint2 / uint4) over fp16: stage through a local
        # array of ``__half`` so we never need to construct a uint{2,4}
        # literal directly.
        elem_type = ctx.type_name(out_dt)
        arr = temp
        init = ", ".join(converted)
        return [
            f"{pad}{elem_type} {arr}[{n}] = {{ {init} }};",
            f"{pad}*reinterpret_cast<{vec_type}*>(&{self.output}[{flat}]) = *reinterpret_cast<const {vec_type}*>({arr});",
        ]


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
