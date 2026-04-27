"""Shared statement primitives — the leaves and control flow used across
every IR layer.

Defined here rather than under any one IR package because all three IRs
(Loop, Tile, Kernel) consume the same leaf vocabulary:

- ``Stmt`` — abstract base for every body statement.
- Leaves: ``Load``, ``Assign``, ``Accum``, ``Write``, ``Select``,
  ``SelectBranch`` — pure compute primitives that read/write SSA names
  and external buffers.
- Control flow: ``Cond`` (if/else), ``Loop`` (iterate over an axis).
- Tree walks: ``iter_body`` (pre-order traversal driven by
  ``Stmt.nested``), ``map_body`` (flat transformer).
- Pretty printing: ``Stmt.pretty(indent)`` returns indented lines for
  the stmt; block-structured stmts recurse into their bodies. Used by
  every Op's ``pretty_body`` for kernel dumps.

Each IR layer adds its own scheduling-specific Stmts on top:

- Loop IR: nothing extra — its bodies are exactly Loop / leaves.
- Tile IR: ``Stage``, ``Combine``, plus the shared ``Tile`` /
  ``Loop`` / ``StridedLoop`` constructs from this module.
- Kernel IR: ``Smem``, ``Sync``, ``TreeHalve``, plus the shared
  constructs.

Loop-IR's ``LoopOp``, ``LoopMeta``, validation, and normalization stay
in ``ir/loop/`` because they're Loop-IR-internal — they enforce
Loop-IR's invariants (SSA scoping rules, axis uniqueness) and produce
Loop-IR's canonical form.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    _PRECEDENCE,
    BinaryExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
    _float_lit,
)
from deplodock.compiler.ir.sigma import Sigma

INDENT = "    "


# ---------------------------------------------------------------------------
# RenderCtx — target-tuned tables + walk state for ``Stmt.render`` / ``Expr.render``
# ---------------------------------------------------------------------------


@dataclass
class RenderCtx:
    """Per-render state. Targets pre-fill ``intrinsics`` / ``builtins`` with
    target-specific spellings (``"exp" → "expf"``, ``"thread_idx.x" →
    "threadIdx.x"``, ...). ``shapes`` maps every buffer to its declared
    shape so multi-dim ``Load`` / ``Write`` indices can be flattened
    row-major. ``explicit_inits`` carries the set of accumulator names
    whose init has been emitted by an enclosing ``Init`` Stmt — Loop's
    default per-Loop init is suppressed for those names.
    """

    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    explicit_inits: set[str] = field(default_factory=set)

    def child(self) -> RenderCtx:
        """Return a new ctx one indent level deeper, sharing all tables."""
        return RenderCtx(
            shapes=self.shapes,
            indent=self.indent + 1,
            intrinsics=self.intrinsics,
            builtins=self.builtins,
            explicit_inits=self.explicit_inits,
        )


def _pad(n: int) -> str:
    return "    " * n


def _axis_identity(a: Axis) -> Axis:
    """Default ``axis_fn`` for ``Loop.rewrite`` / ``StridedLoop.rewrite``."""
    return a


# ---------------------------------------------------------------------------
# Render helpers — translate elementwise op names to Expr trees, and flatten
# multi-dim coord tuples into row-major flat-index strings.
# ---------------------------------------------------------------------------


_BINARY_OP: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "mod": "%",
}


def op_to_expr(fn: str, inputs: list[Expr]) -> Expr:
    """Translate an elementwise op name to an ``Expr`` tree.

    Emits abstract intrinsic names (``"exp"``, ``"fmax"``, ``"fabs"``, ...)
    that targets translate to libm / CUDA spellings via
    ``RenderCtx.intrinsics`` at ``FuncCallExpr.render`` time.
    """
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
        return BinaryExpr("/", Literal(1.0, "float"), BinaryExpr("+", Literal(1.0, "float"), exp_neg))
    if fn in ("exp", "rsqrt", "tanh", "sqrt"):
        return FuncCallExpr(fn, list(inputs))
    if fn == "abs":
        return FuncCallExpr("fabs", list(inputs))
    raise NotImplementedError(f"render: elementwise fn={fn!r} not supported")


def select_to_ternary(s: Select) -> Expr:
    """Build a chained ternary from a ``Select``'s branch list."""
    branches = list(s.branches)
    result: Expr = Var(branches[-1].value)
    for b in reversed(branches[:-1]):
        result = TernaryExpr(cond=b.select, if_true=Var(b.value), if_false=result)
    return result


def render_index(buf: str, indices: tuple, ctx: RenderCtx) -> str:
    """Row-major flatten ``buf[i0][i1]...`` to a single C/CUDA expression."""
    if len(indices) == 0:
        return "0"
    if len(indices) == 1:
        return indices[0].render(ctx)
    shape = ctx.shapes.get(buf)
    if shape is None or len(shape) != len(indices):
        return " + ".join(i.render(ctx) for i in indices)
    parts: list[str] = []
    for d, idx in enumerate(indices):
        stride = 1
        for k in range(d + 1, len(shape)):
            stride *= int(shape[k])
        idx_str = idx.render(ctx, _PRECEDENCE["*"])
        parts.append(idx_str if stride == 1 else f"{idx_str} * {stride}")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Stmt — abstract base
# ---------------------------------------------------------------------------


class Stmt:
    """Base class for IR body statements.

    Every concrete Stmt implements:

    - ``deps()`` — SSA names this stmt reads.
    - ``rewrite(rename_ssa, sigma)`` — return a copy with SSA names mapped
      through ``rename_ssa`` and Expr subterms σ-substituted.
    - ``nested()`` — child statement bodies for tree traversal (default:
      no children; block-structured stmts override).
    """

    def deps(self) -> tuple[str, ...]:
        """SSA names this stmt reads — its 'requirements'."""
        raise NotImplementedError

    def rewrite(
        self,
        rename_ssa: Callable[[str], str],
        sigma: Sigma = Sigma.IDENTITY,
        axis_fn: Callable[[Axis], Axis] = _axis_identity,
    ) -> Stmt:
        """Return a copy with every SSA name (binding + dep refs) mapped
        through ``rename_ssa``, every Expr subterm σ-substituted, and
        every axis on a ``Loop`` / ``StridedLoop`` mapped through
        ``axis_fn``. Subclasses without axes accept and ignore ``axis_fn``;
        Loop-like subclasses thread it through their bodies.

        ``rename_ssa`` is applied uniformly to the stmt's own name (if any)
        and to each name it reads. Callers typically provide a callable
        that defaults to identity (``lambda n: mapping.get(n, n)``) so only
        the names they care about are changed.
        """
        raise NotImplementedError

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        """Child statement bodies for tree traversal.

        Default: no children (leaf stmt). Block-structured stmts override
        to return their body tuple(s) — ``Loop`` returns ``(self.body,)``;
        ``Cond`` returns ``(self.body, self.else_body)``; ``Tile`` returns
        ``(self.body,)``.

        ``iter_body`` walks all IR layers via this single method — every
        node knows its own children, so the walker doesn't need to
        switch on type.
        """
        return ()

    def pretty(self, indent: str = "") -> list[str]:
        """Render this stmt as a list of indented lines.

        Block-structured stmts recurse into their bodies via
        ``child.pretty(indent + INDENT)``; leaves return a single line.
        Subclasses override to control formatting; default surfaces the
        class name as a placeholder for any stmt that forgot to override.
        """
        return [f"{indent}<unrecognized {type(self).__name__}>"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit indented C / CUDA source lines for this stmt.

        Block-structured stmts recurse via ``child.render(ctx.child())``;
        leaves return a single line. The ``ctx`` carries target-specific
        intrinsic / builtin tables, current indent, and per-buf shapes
        for index flattening. Subclasses override.
        """
        raise NotImplementedError(f"{type(self).__name__}.render not implemented")


def pretty_body(body: tuple[Stmt, ...], indent: str = "") -> list[str]:
    """Flatten ``stmt.pretty(indent)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.pretty(indent))
    return out


def render_body(body: tuple[Stmt, ...], ctx: RenderCtx) -> list[str]:
    """Flatten ``stmt.render(ctx)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.render(ctx))
    return out


# ---------------------------------------------------------------------------
# Leaves — pure compute primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Load(Stmt):
    """Read a value from an external input buffer into an SSA name.

    Each external-buffer read is an explicit body statement. ``input`` is
    the source buffer's name (matches the producing graph node's id);
    ``index`` is the dim-wise access pattern over the enclosing axes.
    The produced SSA ``name`` is a regular value that downstream stmts
    read.
    """

    name: str
    input: str
    index: tuple[Expr, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Load(name=rename_ssa(self.name), input=self.input, index=tuple(sigma.apply(e) for e in self.index))

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}{self.name} = load {self.input}[{idx}]"]

    def render(self, ctx: RenderCtx) -> list[str]:
        flat = render_index(self.input, self.index, ctx)
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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Assign(name=rename_ssa(self.name), op=self.op, args=tuple(rename_ssa(a) for a in self.args))

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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Accum(name=rename_ssa(self.name), value=rename_ssa(self.value), op=self.op)

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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Init(name=rename_ssa(self.name), op=self.op)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Init({self.name}, op={self.op.name})"]

    def render(self, ctx: RenderCtx) -> list[str]:
        identity = self.op.identity
        if identity is None:
            raise ValueError(f"Init {self.name!r} op {self.op.name!r} has no identity")
        ctx.explicit_inits.add(self.name)
        return [f"{_pad(ctx.indent)}float {self.name} = {_float_lit(float(identity))};"]


@dataclass(frozen=True)
class Write(Stmt):
    """Write an SSA value to output buffer ``output`` at position ``index``.

    ``output`` is the destination buffer's name (matches the owning graph
    node's id, or — for multi-output kernels — one of its output buffer
    names). ``index`` uses axis Vars to compute the per-dim offset.
    ``value`` references an SSA name available at this point in the body
    (Assign, Accum, or a Load).
    """

    output: str
    index: tuple[Expr, ...]
    value: str

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Write(output=self.output, index=tuple(sigma.apply(e) for e in self.index), value=rename_ssa(self.value))

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}{self.output}[{idx}] = {self.value}"]

    def render(self, ctx: RenderCtx) -> list[str]:
        flat = render_index(self.output, self.index, ctx)
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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Select(
            name=rename_ssa(self.name),
            branches=tuple(SelectBranch(value=rename_ssa(b.value), select=sigma.apply(b.select)) for b in self.branches),
        )

    def pretty(self, indent: str = "") -> list[str]:
        lines: list[str] = []
        for bi, br in enumerate(self.branches):
            prefix = f"{self.name} =" if bi == 0 else f"{' ' * len(self.name)}  "
            lines.append(f"{indent}{prefix} {br.value} when ({br.select.pretty()})")
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        expr = select_to_ternary(self)
        return [f"{_pad(ctx.indent)}float {self.name} = {expr.render(ctx)};"]


# ---------------------------------------------------------------------------
# Control flow — Loop, Cond
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Loop(Stmt):
    """Explicit iteration block — one loop over an axis.

    ``body`` executes ``axis.extent`` times, once per axis value. Reduce-
    kind Loops fold any ``Accum`` statements in their body into the named
    accumulator (one sweep over the axis per accumulator). Free-kind
    Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``Accum`` targets cross the Loop boundary, carrying the finalized
    reduced value.

    Used by Loop IR for general iteration; reused by Kernel IR for
    serial (post-materialization) loops inside cooperative blocks.
    """

    axis: Axis
    body: tuple[Stmt, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    @property
    def is_reduce(self) -> bool:
        """A loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

    @property
    def loops(self) -> tuple[Loop, ...]:
        """Top-level ``Loop`` stmts in the body."""
        return tuple(s for s in self.body if isinstance(s, Loop))

    @property
    def loads(self) -> tuple[Load, ...]:
        """Top-level ``Load`` stmts in the body."""
        return tuple(s for s in self.body if isinstance(s, Load))

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        """Recursive rewrite: rebuild ``body`` with each child's ``rewrite``.
        ``axis`` is mapped through ``axis_fn`` (default identity)."""
        return Loop(axis=axis_fn(self.axis), body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body))

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}:  # {kind}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        out: list[str] = []
        # Per-Loop ``float <acc> = identity;`` for each distinct Accum in the
        # immediate body — suppressed when an enclosing Init already declared it.
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}float {s.name} = {_float_lit(float(identity))};")
        var = self.axis.name
        extent = int(self.axis.extent)
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        for s in self.body:
            out.extend(s.render(inner))
        out.append(f"{pad}}}")
        return out


@dataclass
class Tile(Stmt):
    """Axis-bound scope wrapper — one CUDA-kernel scope.

    Carries ``axes: tuple[BoundAxis, ...]`` (launch geometry —
    ``BIND_THREAD`` and ``BIND_BLOCK`` axes) plus a body of statements.
    Used at both Tile IR (with Tile-IR-specific stmts like ``Stage`` /
    ``Combine`` in the body) and Kernel IR (with hardware primitives
    like ``Smem`` / ``Sync`` / ``TreeHalve`` after materialization).

    Materialization rewrites the body content but preserves the
    wrapper — same axes, same type, just different body shape.

    ``thread_axes`` / ``block_axes`` are convenience properties that
    project ``axes`` by binding kind — render and launch geometry use
    them.
    """

    axes: tuple[BoundAxis, ...]
    body: tuple[Stmt, ...]

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        new_axes = tuple(BoundAxis(axis=axis_fn(ba.axis), bind=ba.bind) for ba in self.axes)
        return Tile(axes=new_axes, body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body))

    @property
    def thread_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_THREAD)

    @property
    def block_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_BLOCK)

    @property
    def all_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes)

    @property
    def loops(self) -> tuple[Loop, ...]:
        """Top-level ``Loop`` stmts in the body. Mirrors ``Loop.loops`` so
        scope-walking passes can iterate without re-filtering."""
        return tuple(s for s in self.body if isinstance(s, Loop))

    def pretty(self, indent: str = "") -> list[str]:
        axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in self.axes) or "-"
        return [f"{indent}Tile(axes=({axes})):", *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """CUDA block / thread axis decode + body emission.

        Two forms:

        - **Cooperative (``block_axes`` populated):** one CUDA block per
          ``block_axes`` slot, ``thread_axes`` index threads inside the
          block. Decodes ``blockIdx.x`` and ``threadIdx.x`` directly.
        - **Linear (``block_axes`` empty):** flatten all ``thread_axes``
          into one linear ``tid``; bounds-guard against the product of
          extents.
        """
        pad = _pad(ctx.indent)
        inner = ctx.child()
        if self.block_axes:
            out = [f"{pad}{{"]
            out.extend(_render_grid_axis_decode(self.block_axes, "blockIdx.x", inner))
            out.extend(_render_grid_axis_decode(self.thread_axes, "threadIdx.x", inner))
            for s in self.body:
                out.extend(s.render(inner))
            out.append(f"{pad}}}")
            return out

        n_threads = 1
        for ax in self.thread_axes:
            n_threads *= int(ax.extent)
        out = [
            f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
            f"{pad}if (tid < {n_threads}) {{",
        ]
        out.extend(_render_thread_axis_decode(self.thread_axes, inner))
        for s in self.body:
            out.extend(s.render(inner))
        out.append(f"{pad}}}")
        return out


def _render_grid_axis_decode(axes: tuple[Axis, ...], idx_expr: str, ctx: RenderCtx) -> list[str]:
    """Decode ``idx_expr`` (``blockIdx.x`` or ``threadIdx.x``) into per-axis ints."""
    pad = _pad(ctx.indent)
    if not axes:
        return []
    if len(axes) == 1:
        return [f"{pad}int {axes[0].name} = {idx_expr};"]
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = {idx_expr} % {extent};")
        else:
            decoded.append(f"int {ax.name} = ({idx_expr} / {stride}) % {extent};")
        stride *= extent
    outer = axes[0]
    outer_stride = 1
    for ax in axes[1:]:
        outer_stride *= int(ax.extent)
    decoded[-1] = f"int {outer.name} = {idx_expr} / {outer_stride};"
    return [pad + line for line in reversed(decoded)]


def _render_thread_axis_decode(axes: tuple[Axis, ...], ctx: RenderCtx) -> list[str]:
    """Emit ``int <axis> = (tid / stride) % extent;`` per axis."""
    pad = _pad(ctx.indent)
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = tid % {extent};")
        else:
            decoded.append(f"int {ax.name} = (tid / {stride}) % {extent};")
        stride *= extent
    if len(axes) == 1:
        decoded = [f"int {axes[0].name} = tid;"]
    else:
        outer = axes[0]
        outer_stride = 1
        for ax in axes[1:]:
            outer_stride *= int(ax.extent)
        decoded[-1] = f"int {outer.name} = tid / {outer_stride};"
    return [pad + line for line in reversed(decoded)]


@dataclass(frozen=True)
class StridedLoop(Stmt):
    """Strided iteration: ``for (axis = start; axis < axis.extent; axis += step)``.

    Cooperative variant of ``Loop`` — used at Tile IR to express "threads
    of the CUDA block stride through this axis" (typical
    ``start = Var('t'), step = BLOCK_SIZE``). The body uses the original
    axis Var directly; the strided iteration shape is encoded by the
    loop construct itself rather than via affine indexing in the body.

    Reduction detection mirrors ``Loop``: a ``StridedLoop`` is a
    reduce-loop iff its body contains an ``Accum``."""

    axis: Axis
    start: Expr
    step: Expr
    body: tuple[Stmt, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body,)

    @property
    def is_reduce(self) -> bool:
        """A strided loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return StridedLoop(
            axis=axis_fn(self.axis),
            start=sigma.apply(self.start),
            step=sigma.apply(self.step) if isinstance(self.step, Expr) else self.step,
            body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body),
        )

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}StridedLoop({self.axis.name} = {start}; < {self.axis.extent}; += {step}):  # {kind}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``for (int axis = start; axis < extent; axis += step)`` with the
        same per-Loop accumulator-init prelude as ``Loop.render``."""
        pad = _pad(ctx.indent)
        out: list[str] = []
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}float {s.name} = {_float_lit(float(identity))};")
        var = self.axis.name
        start_str = self.start.render(ctx)
        step_str = self.step.render(ctx) if isinstance(self.step, Expr) else str(self.step)
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {int(self.axis.extent)}; {var} += {step_str}) {{")
        inner = ctx.child()
        for s in self.body:
            out.extend(s.render(inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class Cond(Stmt):
    """Conditional block — ``if (cond) { body } [else { else_body }]``.

    ``cond`` is an ``Expr`` over axis Vars and previously-defined SSA
    names; ``body`` and ``else_body`` are stmt sequences executed when
    the predicate evaluates true / false respectively. ``else_body``
    empty means a bare ``if``.

    SSA scoping mirrors ``Loop``: names defined inside either body are
    scoped to that body, except ``Accum`` targets which cross the boundary
    with their finalized value (matching Loop semantics).

    ``deps`` are the SSA names referenced inside ``cond`` — the splicer /
    dataflow analyses need them to thread the predicate's reads through.
    Names referenced inside ``body`` / ``else_body`` are the body stmts'
    own deps; the recursive walker picks them up.
    """

    cond: Expr
    body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...] = ()

    def deps(self) -> tuple[str, ...]:
        return tuple(self.cond.free_vars())

    def nested(self) -> tuple[tuple[Stmt, ...], ...]:
        return (self.body, self.else_body)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Cond(
            cond=sigma.apply(self.cond),
            body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body),
            else_body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.else_body),
        )

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}if ({self.cond.pretty()}):", *pretty_body(self.body, indent + INDENT)]
        if self.else_body:
            lines.append(f"{indent}else:")
            lines.extend(pretty_body(self.else_body, indent + INDENT))
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        cond = self.cond.render(ctx)
        inner = ctx.child()
        body: list[str] = []
        for s in self.body:
            body.extend(s.render(inner))
        out = [f"{pad}if ({cond}) {{", *body, f"{pad}}}"]
        if self.else_body:
            out[-1] = f"{pad}}} else {{"
            for s in self.else_body:
                out.extend(s.render(inner))
            out.append(f"{pad}}}")
        return out


# ---------------------------------------------------------------------------
# Tree walks
# ---------------------------------------------------------------------------


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    """Yield every ``Stmt`` in ``body`` in pre-order, recursing into each
    stmt's ``nested()`` bodies.

    Works across all IRs (Loop, Tile, Kernel) without type-switching:
    every block-structured Stmt subclass overrides ``Stmt.nested`` to
    return its child body tuples, and this walker drives off that
    method. Callers that want only leaves can filter with ``isinstance``.
    """
    for s in body:
        yield s
        for child_body in s.nested():
            yield from iter_body(child_body)


def map_body(
    body: tuple[Stmt, ...],
    fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]],
) -> tuple[Stmt, ...]:
    """Flat body transformer: apply ``fn`` to each stmt, splice its result
    into the output. ``fn`` may return:

    - a single ``Stmt`` (kept in place of the input),
    - ``None`` (drop the input), or
    - an iterable of ``Stmt`` (inline all of them — useful for 1:N
      expansions like loop unrolling or size-1 Loop inlining).

    ``fn`` is called on *every* stmt including ``Loop`` wrappers; recursion
    into a Loop's body is the caller's responsibility (typically by writing
    a self-recursive ``fn`` that returns ``Loop(axis=..., body=map_body(s.body, fn))``
    for Loop cases). Lets callers pick their own policy for axis renames,
    Loop skipping, or selective recursion.
    """
    out: list[Stmt] = []
    for s in body:
        r = fn(s)
        if r is None:
            continue
        if isinstance(r, Stmt):
            out.append(r)
        else:
            out.extend(r)
    return tuple(out)


# ---------------------------------------------------------------------------
# Normalization passes
# ---------------------------------------------------------------------------
#
# Pure ``body → body`` transforms applied via ``normalize_body`` from
# ``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so every constructed
# Op lands in canonical form. The passes operate on the shared Stmt
# vocabulary (``Loop``, ``Load``, ``Assign``, ``Accum``, ``Select``,
# ``Write``) and recurse through every block-structured Stmt
# (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) so they apply uniformly
# to Loop IR and Tile IR bodies.

from deplodock.compiler.ir.expr import Interval, SimplifyCtx  # noqa: E402


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def _recurse_through_block(s: Stmt, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Stmt | None:
    """If ``s`` is a non-Loop block stmt, return a copy with its body / bodies
    re-walked through ``fn`` via ``map_body``. Else return ``None``."""
    if isinstance(s, StridedLoop):
        return StridedLoop(axis=s.axis, start=s.start, step=s.step, body=map_body(s.body, fn))
    if isinstance(s, Tile):
        return Tile(axes=s.axes, body=map_body(s.body, fn))
    if isinstance(s, Cond):
        return Cond(cond=s.cond, body=map_body(s.body, fn), else_body=map_body(s.else_body, fn))
    return None


def normalize_body(stmts: tuple[Stmt, ...], *, hoist: bool = True) -> tuple[Stmt, ...]:
    """Apply the structural and cosmetic normalization passes in order.

    Used by both ``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so
    bodies — Loop-IR and Tile-IR — land in a canonical shape before
    validation.

    ``hoist=False`` skips :func:`hoist_loop_invariants`. TileOp bodies turn
    it off because a Stage binding is scoped to the Loop where it's
    declared — hoisting Loads that read from a staged buffer above the
    Stage decl would leave the read referencing an undeclared name."""
    stmts = drop_size_one_free_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    if hoist:
        stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = rename_ssa_sequential(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator.
    Recurses through StridedLoop / Tile / Cond bodies without rewriting
    those wrappers (their iteration semantics aren't a free Loop)."""

    def fn(s: Stmt) -> Stmt | tuple[Stmt, ...]:
        if isinstance(s, Loop):
            candidate = Loop(axis=s.axis, body=map_body(s.body, fn))
            if int(candidate.axis.extent) == 1 and not candidate.is_reduce:
                sub = Sigma({candidate.axis.name: Literal(0, "int")})
                return tuple(c.rewrite(_identity_rename, sub) for c in candidate.body)
            return candidate
        rec = _recurse_through_block(s, fn)
        return rec if rec is not None else s

    return map_body(stmts, fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def _recurse_canonicalize(s: Stmt) -> Stmt:
    if isinstance(s, Loop):
        return Loop(axis=s.axis, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, StridedLoop):
        return StridedLoop(axis=s.axis, start=s.start, step=s.step, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, Tile):
        return Tile(axes=s.axes, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, Cond):
        return Cond(
            cond=s.cond,
            body=canonicalize_free_axis_order(s.body),
            else_body=canonicalize_free_axis_order(s.else_body),
        )
    return s


def canonicalize_free_axis_order(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Sort the outer chain of free ``Loop`` blocks alphabetically by axis
    name. The chain is the sequence of single-child free Loops at the top of
    ``stmts``; it terminates at a reduce Loop or a branching body. Recursion
    continues into terminal block bodies (Loop / StridedLoop / Tile / Cond)."""
    chain_axes: list[Axis] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain_axes.append(loop.axis)
        current = loop.body

    terminal = tuple(_recurse_canonicalize(s) for s in current)

    chain_axes_sorted = sorted(chain_axes, key=lambda a: a.name)
    result: tuple[Stmt, ...] = terminal
    for axis in reversed(chain_axes_sorted):
        result = (Loop(axis=axis, body=result),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them. Every such Assign is dropped and downstream
    references to ``y`` are rewired to the alias root. Pure IR hygiene."""
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def fn(s: Stmt) -> Stmt | None:
        if isinstance(s, Loop):
            return Loop(axis=s.axis, body=map_body(s.body, fn))
        rec = _recurse_through_block(s, fn)
        if rec is not None:
            return rec
        if isinstance(s, Assign) and s.op.name == "copy" and len(s.args) == 1:
            alias[s.name] = s.args[0]
            return None
        return s.rewrite(resolve)

    return map_body(stmts, fn)


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes index
    the same ``(Load.source, dim)`` position and rename them to a single
    canonical axis name. Recurses through every block-structured Stmt
    (Loop / StridedLoop / Tile / Cond) to find nested scopes."""

    def walk(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, Loop):
                new_body.append(Loop(axis=s.axis, body=walk(s.body)))
            elif isinstance(s, StridedLoop):
                new_body.append(StridedLoop(axis=s.axis, start=s.start, step=s.step, body=walk(s.body)))
            elif isinstance(s, Tile):
                new_body.append(Tile(axes=s.axes, body=walk(s.body)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=walk(s.body), else_body=walk(s.else_body)))
            else:
                new_body.append(s)

        groups: dict[frozenset[tuple[str, int]], list[int]] = {}
        for i, s in enumerate(new_body):
            if isinstance(s, Loop) and s.is_reduce:
                positions = _reduce_axis_source_positions(s.body, s.axis.name)
                if positions:
                    groups.setdefault(frozenset(positions), []).append(i)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            first = new_body[indices[0]]
            assert isinstance(first, Loop)
            canonical = first.axis.name
            canonical_extent = int(first.axis.extent)
            for idx in indices[1:]:
                loop = new_body[idx]
                assert isinstance(loop, Loop)
                if int(loop.axis.extent) != canonical_extent or loop.axis.name == canonical:
                    continue
                new_axis = Axis(name=canonical, extent=canonical_extent)
                sub = Sigma({loop.axis.name: Var(canonical)})
                rename_axis = _make_axis_renamer(loop.axis.name, new_axis)
                renamed = tuple(s.rewrite(_identity_rename, sub, rename_axis) for s in loop.body)
                new_body[idx] = Loop(axis=new_axis, body=renamed)

        return tuple(new_body)

    return walk(stmts)


def _reduce_axis_source_positions(body: tuple[Stmt, ...], reduce_axis_name: str) -> set[tuple[str, int]]:
    """Collect ``(source, dim)`` positions where ``Var(reduce_axis_name)``
    appears bare in a Load index within ``body`` (recursing into nested
    blocks)."""
    return {
        (s.input, dim)
        for s in iter_body(body)
        if isinstance(s, Load)
        for dim, e in enumerate(s.index)
        if isinstance(e, Var) and e.name == reduce_axis_name
    }


# ---------------------------------------------------------------------------
# Pass 5: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Move ``Load`` / ``Assign`` / ``Select`` stmts out of ``Loop``s whose
    axis their value doesn't depend on. Hoisting only crosses ``Loop``
    boundaries; ``StridedLoop`` / ``Tile`` / ``Cond`` are barriers but are
    recursed into so inner Loops still get the optimization."""
    axes_of: dict[str, frozenset[str]] = {}
    ssa_names: set[str] = set()

    def _load_index_parts(index: tuple[Expr, ...]) -> tuple[set[str], set[str]]:
        deps: set[str] = set()
        axes: set[str] = set()
        for e in index:
            for v in e.free_vars():
                (deps if v in ssa_names else axes).add(v)
        return deps, axes

    def _stmt_live(s: Stmt) -> frozenset[str]:
        if isinstance(s, Load):
            deps, axes = _load_index_parts(s.index)
            result = set(axes)
            for d in deps:
                result |= axes_of.get(d, frozenset())
            return frozenset(result)
        if isinstance(s, Assign):
            result = set()
            for name in s.args:
                result |= axes_of.get(name, frozenset())
            return frozenset(result)
        if isinstance(s, Select):
            result = set()
            for b in s.branches:
                result |= b.select.free_vars()
                result |= axes_of.get(b.value, frozenset())
            return frozenset(result)
        return frozenset()

    def _deps(s: Stmt) -> tuple[str, ...]:
        if isinstance(s, Load):
            deps, _ = _load_index_parts(s.index)
            return tuple(deps)
        if isinstance(s, Assign):
            return s.args
        if isinstance(s, Select):
            return tuple(b.value for b in s.branches)
        return ()

    def _record(c: Stmt, bindings: set[str]) -> None:
        name = getattr(c, "name", None)
        if name is None:
            return
        if isinstance(c, Accum):
            axes_of[name] = axes_of.get(c.value, frozenset())
        else:
            axes_of[name] = _stmt_live(c)
        ssa_names.add(name)
        bindings.add(name)

    def walk(body: tuple[Stmt, ...], outer_bindings: set[str]) -> list[Stmt]:
        new_body: list[Stmt] = []
        bindings = set(outer_bindings)

        for s in body:
            if isinstance(s, Loop):
                inner = walk(s.body, bindings)
                axis = s.axis.name
                hoisted_names: set[str] = set()
                hoisted: list[Stmt] = []
                stay: list[Stmt] = []
                for c in inner:
                    if isinstance(c, (Load, Assign, Select)):
                        live = axes_of.get(c.name, frozenset())
                        if axis not in live and all(d in bindings or d in hoisted_names for d in _deps(c)):
                            hoisted.append(c)
                            hoisted_names.add(c.name)
                            continue
                    stay.append(c)

                for h in hoisted:
                    new_body.append(h)
                    bindings.add(h.name)
                new_body.append(Loop(axis=s.axis, body=tuple(stay)))

                for c in inner:
                    if isinstance(c, Accum):
                        axes_of[c.name] = axes_of.get(c.value, frozenset()) - {axis}
                        ssa_names.add(c.name)
                        bindings.add(c.name)
            elif isinstance(s, StridedLoop):
                inner = walk(s.body, bindings)
                new_body.append(StridedLoop(axis=s.axis, start=s.start, step=s.step, body=tuple(inner)))
                for c in inner:
                    if isinstance(c, Accum):
                        ssa_names.add(c.name)
                        bindings.add(c.name)
            elif isinstance(s, Tile):
                inner = walk(s.body, bindings)
                new_body.append(Tile(axes=s.axes, body=tuple(inner)))
            elif isinstance(s, Cond):
                inner_b = walk(s.body, bindings)
                inner_e = walk(s.else_body, bindings)
                new_body.append(Cond(cond=s.cond, body=tuple(inner_b), else_body=tuple(inner_e)))
            else:
                new_body.append(s)
                _record(s, bindings)

        return new_body

    return tuple(walk(stmts, set()))


# ---------------------------------------------------------------------------
# Pass 6: simplify Exprs inside body Stmts (constant folding, identity collapse,
# range-based comparison folding). The per-Expr rewrite logic lives on each
# ``Expr`` subclass as ``simplify(ctx)``; this pass just walks the body and
# threads ``SimplifyCtx`` ranges as it descends.
# ---------------------------------------------------------------------------


def _simplify_expr_tuple(xs: tuple[Expr, ...], ctx: SimplifyCtx) -> tuple[Expr, ...]:
    return tuple(e.simplify(ctx) for e in xs)


def _simplify_stmt(stmt: Stmt, ctx: SimplifyCtx) -> Stmt:
    if isinstance(stmt, Loop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return Loop(stmt.axis, tuple(_simplify_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, StridedLoop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return StridedLoop(
            axis=stmt.axis,
            start=stmt.start.simplify(ctx),
            step=stmt.step.simplify(ctx) if isinstance(stmt.step, Expr) else stmt.step,
            body=tuple(_simplify_stmt(s, inner) for s in stmt.body),
        )
    if isinstance(stmt, Tile):
        inner = ctx
        for ba in stmt.axes:
            inner = inner.extend(ba.axis.name, Interval(0, ba.axis.extent - 1))
        return Tile(axes=stmt.axes, body=tuple(_simplify_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond.simplify(ctx),
            body=tuple(_simplify_stmt(s, ctx) for s in stmt.body),
            else_body=tuple(_simplify_stmt(s, ctx) for s in stmt.else_body),
        )
    if isinstance(stmt, Select):
        return Select(stmt.name, tuple(SelectBranch(b.value, b.select.simplify(ctx)) for b in stmt.branches))
    if isinstance(stmt, Write):
        return Write(stmt.output, _simplify_expr_tuple(stmt.index, ctx), stmt.value)
    if isinstance(stmt, Load):
        return Load(stmt.name, stmt.input, _simplify_expr_tuple(stmt.index, ctx))
    # Tile-IR-only ``Stage`` carries Exprs in ``origin`` / ``source_index_template``.
    # Lazy import to avoid a tile→stmt circular at module load time.
    from deplodock.compiler.ir.tile.ir import Stage  # noqa: PLC0415

    if isinstance(stmt, Stage):
        new_template = _simplify_expr_tuple(stmt.source_index_template, ctx) if stmt.source_index_template is not None else None
        return Stage(
            name=stmt.name,
            buf=stmt.buf,
            origin=_simplify_expr_tuple(stmt.origin, ctx),
            axes=stmt.axes,
            slab_dims=stmt.slab_dims,
            source_index_template=new_template,
        )
    # Assign / Accum / Combine carry only SSA names — no Expr field to simplify.
    return stmt


def simplify_body(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Simplify every Expr inside a body. Seeds ``SimplifyCtx`` from
    ``Loop`` / ``StridedLoop`` / ``Tile`` axis extents as the walker descends."""
    ctx = SimplifyCtx.empty()
    return tuple(_simplify_stmt(s, ctx) for s in body)


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Canonicalize names in a fused body:

    - Axes from every axis-bearing scope (``Loop`` / ``StridedLoop`` /
      ``Tile.axes``) renamed to ``a0, a1, ...`` in pre-order of first
      declaration. All three share one numbering namespace so Tile.axes
      ``a0_o`` and a Loop axis ``a2_o`` don't collide on rename.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    ssa_rename: dict[str, str] = {}
    axis_rename: dict[str, str] = {}
    expr_sub: dict[str, Expr] = {}
    counters = {"v": 0, "in": 0, "acc": 0}

    def _rename(name: str, prefix: str) -> str:
        new = f"{prefix}{counters[prefix]}"
        ssa_rename[name] = new
        counters[prefix] += 1
        return new

    def _record_axis(name: str) -> None:
        if name in axis_rename:
            return
        new = f"a{len(axis_rename)}"
        axis_rename[name] = new
        if name != new:
            expr_sub[name] = Var(new)

    for stmt in iter_body(stmts):
        if isinstance(stmt, Load) and stmt.name not in ssa_rename:
            new = _rename(stmt.name, "in")
            if stmt.name != new:
                expr_sub[stmt.name] = Var(new)
        elif isinstance(stmt, Accum) and stmt.name not in ssa_rename:
            _rename(stmt.name, "acc")
        elif isinstance(stmt, (Assign, Select)) and stmt.name not in ssa_rename:
            _rename(stmt.name, "v")
        elif isinstance(stmt, Tile):
            for ba in stmt.axes:
                _record_axis(ba.axis.name)
        elif isinstance(stmt, (Loop, StridedLoop)):
            _record_axis(stmt.axis.name)

    if all(o == n for o, n in ssa_rename.items()) and all(o == n for o, n in axis_rename.items()):
        return stmts

    sigma = Sigma(expr_sub)

    def rename_ssa(name: str) -> str:
        return ssa_rename.get(name, name)

    def axis_fn(a: Axis) -> Axis:
        new = axis_rename.get(a.name, a.name)
        return Axis(name=new, extent=a.extent) if new != a.name else a

    return tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in stmts)


__all__ = [
    "INDENT",
    "Stmt",
    "Load",
    "Assign",
    "Accum",
    "Init",
    "Write",
    "StridedLoop",
    "Tile",
    "Select",
    "SelectBranch",
    "Loop",
    "Cond",
    "RenderCtx",
    "iter_body",
    "map_body",
    "pretty_body",
    "render_body",
    "render_index",
    "op_to_expr",
    "select_to_ternary",
    "normalize_body",
    "drop_size_one_free_axes",
    "canonicalize_free_axis_order",
    "eliminate_copy_aliases",
    "unify_sibling_reduce_axes",
    "hoist_loop_invariants",
    "simplify_body",
    "rename_ssa_sequential",
]
