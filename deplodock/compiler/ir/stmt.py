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
from deplodock.compiler.ir.expr import Expr, Literal
from deplodock.compiler.ir.sigma import Sigma

INDENT = "    "

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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        """Return a copy with every SSA name (binding + dep refs) mapped
        through ``rename_ssa`` and every Expr subterm σ-substituted.

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


def pretty_body(body: tuple[Stmt, ...], indent: str = "") -> list[str]:
    """Flatten ``stmt.pretty(indent)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.pretty(indent))
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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Load(name=rename_ssa(self.name), input=self.input, index=tuple(sigma.apply(e) for e in self.index))

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}{self.name} = load {self.input}[{idx}]"]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Assign(name=rename_ssa(self.name), op=self.op, args=tuple(rename_ssa(a) for a in self.args))

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} = {self.op.name}({', '.join(self.args)})"]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Accum(name=rename_ssa(self.name), value=rename_ssa(self.value), op=self.op)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} <- {self.op.name}({self.name}, {self.value})"]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Init(name=rename_ssa(self.name), op=self.op)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Init({self.name}, op={self.op.name})"]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Write(output=self.output, index=tuple(sigma.apply(e) for e in self.index), value=rename_ssa(self.value))

    def pretty(self, indent: str = "") -> list[str]:
        idx = ", ".join(e.pretty() for e in self.index)
        return [f"{indent}{self.output}[{idx}] = {self.value}"]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        """Recursive rewrite: rebuild ``body`` with each child's ``rewrite``.

        ``axis`` is left alone — strategies that need axis renaming
        (``loop.normalize``) special-case ``Loop`` and bypass this method.
        """
        return Loop(axis=self.axis, body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body))

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}:  # {kind}"
        return [head, *pretty_body(self.body, indent + INDENT)]


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

    @property
    def thread_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_THREAD)

    @property
    def block_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_BLOCK)

    def pretty(self, indent: str = "") -> list[str]:
        axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in self.axes) or "-"
        return [f"{indent}Tile(axes=({axes})):", *pretty_body(self.body, indent + INDENT)]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return StridedLoop(
            axis=self.axis,
            start=sigma.apply(self.start),
            step=sigma.apply(self.step) if isinstance(self.step, Expr) else self.step,
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
        )

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}StridedLoop({self.axis.name} = {start}; < {self.axis.extent}; += {step}):  # {kind}"
        return [head, *pretty_body(self.body, indent + INDENT)]


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY) -> Stmt:
        return Cond(
            cond=sigma.apply(self.cond),
            body=tuple(s.rewrite(rename_ssa, sigma) for s in self.body),
            else_body=tuple(s.rewrite(rename_ssa, sigma) for s in self.else_body),
        )

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}if ({self.cond.pretty()}):", *pretty_body(self.body, indent + INDENT)]
        if self.else_body:
            lines.append(f"{indent}else:")
            lines.extend(pretty_body(self.else_body, indent + INDENT))
        return lines


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
    "iter_body",
    "map_body",
    "pretty_body",
]
