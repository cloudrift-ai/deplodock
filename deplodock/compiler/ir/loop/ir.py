"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    body              : tuple[Stmt, ...]   — SSA body: Assign | Accum | Write | Select | Loop | Load
    axes              : computed property  — iteration space walked from body's Loop tree
    reduce_axis_names : computed property  — names of axes whose Loop wraps an Accum
    loads             : computed property  — body-form Load stmts (external reads)
    accums            : computed property  — unique Accum stmts, first-use order
    input_bufs        : computed property  — distinct Load.source names (in first-use order)
    num_inputs        : computed property  — len(input_bufs)
    analyze()         : LoopMeta           — precomputed name → def / scope / reduce-axis / live-axes
    map(fn)           : LoopOp             — transform body via ``map_body`` and rewrap
    __iter__          : Iterator[Stmt]     — pre-order walk (via ``iter_body``)

Iteration is explicit via ``Loop(axis, body)`` statements. Each ``Loop``
is one iteration dimension; ``Loop.body`` runs ``axis.extent`` times.
Reduce-kind Loops fold ``Accum`` statements into the accumulator named
by each ``Accum.name``. Free-kind Loops run in parallel with no
accumulator folding. Reading top-to-bottom matches execution order.

SSA names defined by ``Assign`` / ``Select`` / ``Load`` inside a
``Loop.body`` are scoped to that body — only accumulator state crosses
Loop boundaries (via ``Accum``). Invariants (unique names, defined-
before-use, accumulator liveness) are enforced by ``LoopOp.__post_init__``
(which also runs ``normalize_body`` to canonicalize the body).

Free-function companions (used by passes that work on raw
``tuple[Stmt, ...]``): ``iter_body`` (pre-order generator),
``map_body`` (transformer supporting ``Stmt | None | Iterable[Stmt]``),
``Stmt.rewrite(rename_ssa, sigma)`` (per-stmt SSA / Expr rewrite).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr, Literal, free_vars
from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.loop.sigma import Sigma

# ---------------------------------------------------------------------------
# Axis — named iteration variable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """One named iteration variable at the loop level.

    Referenced from ``Expr`` subtrees (inside ``Load.index`` etc.) by
    ``Var(name)``. Free vs reduce is inferred structurally from the body:
    a ``Loop`` over this axis is a reduce loop iff its body (recursively)
    contains an ``Accum``. See ``LoopOp.reduce_axis_names``.

    ``extent`` is a static integer in v1; future revisions may allow an
    ``Expr`` for dynamic batch/seq dims.
    """

    name: str
    extent: int


# ---------------------------------------------------------------------------
# Scope — a path of enclosing axes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """Enclosing loop nest from outermost to innermost.

    A ``Scope`` identifies a location in a ``LoopOp`` body: the sequence
    of ``Loop`` axes one descends to reach that point. Empty = body root.
    Used by analysis passes that need to know where a named SSA value was
    defined or where a new stmt should be emitted.
    """

    enclosing: tuple[Axis, ...] = ()

    def nest(self, axis: Axis) -> Scope:
        return Scope(enclosing=self.enclosing + (axis,))


# ---------------------------------------------------------------------------
# Body statements
# ---------------------------------------------------------------------------


class Stmt:
    """Base class for loop-IR body statements.

    Concrete stmts (``Load``, ``Assign``, ``Accum``, ``Select``, ``Write``,
    ``Loop``) are frozen dataclasses that inherit from this base. Each
    implements :meth:`deps` — the SSA names the stmt reads — so consumers
    (splicer, validators) can query dependencies uniformly.
    """

    def deps(self) -> tuple[str, ...]:
        """SSA names this stmt reads — its 'requirements'."""
        raise NotImplementedError

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        """Return a copy with every SSA name (binding + dep refs) mapped
        through ``rename_ssa`` and every Expr subterm σ-substituted.

        ``rename_ssa`` is applied uniformly to the stmt's own name (if any)
        and to each name it reads. Callers typically provide a callable
        that defaults to identity (``lambda n: mapping.get(n, n)``) so only
        the names they care about are changed.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Load(Stmt):
    """Read a value from an external input buffer into an SSA name.

    Each external-buffer read is an explicit body statement. ``source`` is
    the source buffer's name (matches the producing graph node's id);
    ``index`` is the dim-wise access pattern over the enclosing axes. The
    produced SSA ``name`` is a regular value that downstream stmts read.
    """

    name: str
    input: str
    index: tuple[Expr, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        return Load(name=rename_ssa(self.name), input=self.input, index=tuple(sigma.apply(e) for e in self.index))


# ---------------------------------------------------------------------------
# Accumulator identity table
# ---------------------------------------------------------------------------
#
# ``Accum`` stmts carry their own reduce op — ``add`` / ``max`` / ``min`` /
# ``mul``. The init value for the accumulator is derived from the op's
# ``identity`` metadata on ``ir.expr.ElementwiseImpl`` — no separate dispatch table.
# An ``Accum(target, value, op=...)`` inside a reduce ``Loop`` implicitly
# initializes the accumulator to ``op.op.identity`` before the sweep and
# folds via ``op`` at each iteration. The target SSA name is bound in the
# enclosing scope after the Loop.


@dataclass(frozen=True)
class Assign(Stmt):
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is an ``ElementwiseImpl`` — the elementwise combine (add / mul / exp /
    ...). Reductions live in ``Accum``. ``args`` reference ``$N`` ports,
    an ``Accum.name`` (reads current / finalized acc value), or prior SSA
    names.
    """

    name: str
    op: ElementwiseImpl
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.op, str):
            object.__setattr__(self, "op", ElementwiseImpl(self.op))

    def deps(self) -> tuple[str, ...]:
        return self.args

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        return Assign(name=rename_ssa(self.name), op=self.op, args=tuple(rename_ssa(a) for a in self.args))


@dataclass(frozen=True)
class Accum(Stmt):
    """Reduce accumulator — declares-and-folds in one statement.

    Semantics: ``name = op(name, value)`` inside the enclosing reduce
    ``Loop``. Before the first iteration ``name`` is initialized to
    ``op.identity`` (the combine's neutral element). After the Loop
    completes, ``name`` is an SSA binding visible in the enclosing scope,
    carrying the finalized reduced value.

    ``op`` is an ``ElementwiseImpl`` — typically one of ``ADD`` / ``MAX`` / ``MIN``
    / ``MUL``. It defines both the combine operation and the accumulator's
    identity value. Multiple ``Accum`` stmts targeting the same ``name`` in
    one reduce Loop must agree on ``op``.

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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        return Accum(name=rename_ssa(self.name), value=rename_ssa(self.value), op=self.op)


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        return Write(output=self.output, index=tuple(sigma.apply(e) for e in self.index), value=rename_ssa(self.value))


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

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        return Select(
            name=rename_ssa(self.name),
            branches=tuple(SelectBranch(value=rename_ssa(b.value), select=sigma.apply(b.select)) for b in self.branches),
        )


@dataclass(frozen=True)
class Loop(Stmt):
    """Explicit iteration block — one loop over an axis.

    ``body`` executes ``axis.extent`` times, once per axis value. Reduce-kind
    Loops fold any ``Accum`` statements in their body into the named
    accumulator (one sweep over the axis per accumulator). Free-kind
    Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``Accum`` targets cross the Loop boundary, carrying the finalized
    reduced value.
    """

    axis: Axis
    body: tuple[Stmt, ...]

    def deps(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class Cond(Stmt):
    """Conditional block — ``if (cond) { body } [else { else_body }]``.

    ``cond`` is an ``Expr`` over axis Vars and previously-defined SSA
    names; ``body`` and ``else_body`` are stmt sequences executed
    when the predicate evaluates true / false respectively. ``else_body``
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
        return tuple(free_vars(self.cond))

    def rewrite(self, rename_ssa: Callable[[str], str], sigma: Sigma) -> Stmt:
        # Don't recurse into ``body`` / ``else_body`` here — that's the caller's
        # responsibility (see ``normalize._rewrite_body``), matching ``Loop``.
        return Cond(cond=sigma.apply(self.cond), body=self.body, else_body=self.else_body)


# ---------------------------------------------------------------------------
# LoopOp
# ---------------------------------------------------------------------------


@dataclass
class LoopOp(Op):
    """One kernel's worth of computation as an SSA program over named axes.

    ``body`` is the sole stored field: a tuple of nested ``Loop`` blocks
    with leaf ``Load`` / ``Assign`` / ``Accum`` / ``Write`` / ``Select``
    statements. ``axes``, ``loads``, ``accums``, and ``num_inputs`` are
    computed properties derived from that tree.
    """

    body: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.loop.normalize import normalize_body

        new_body = normalize_body(self.body)
        if new_body != self.body:
            self.body = new_body
        _validate(self)

    @property
    def axes(self) -> tuple[Axis, ...]:
        """Iteration axes collected from the body's ``Loop`` tree.

        Pre-order traversal: outer ``Loop`` blocks (the kernel's grid) come
        first, nested inner Loops follow. Sibling Loops with the same axis
        name (softmax's two K-sweeps) contribute one entry — axes are
        deduplicated by name, keeping the first occurrence.
        """
        seen: dict[str, Axis] = {}
        for s in self:
            if isinstance(s, Loop) and s.axis.name not in seen:
                seen[s.axis.name] = s.axis
        return tuple(seen.values())

    @property
    def reduce_axis_names(self) -> frozenset[str]:
        """Names of axes whose Loop directly wraps an ``Accum``.

        Derived from body structure, not stored metadata: a Loop is a
        'reduce Loop' iff its immediate body contains an Accum.
        """
        names: set[str] = set()

        def walk(stmts: tuple[Stmt, ...], innermost: str | None) -> None:
            for s in stmts:
                if isinstance(s, Accum) and innermost is not None:
                    names.add(innermost)
                elif isinstance(s, Loop):
                    walk(s.body, s.axis.name)

        walk(self.body, None)
        return frozenset(names)

    @property
    def loads(self) -> tuple[Load, ...]:
        """All ``Load`` statements in the body, pre-order.

        New-form successor to ``inputs``: Loads carry an explicit SSA name,
        source-buffer index, and access pattern — unlike ``$N`` refs, they
        live at the scope they're needed and can share a source with other
        Loads that use different indices.
        """
        return tuple(s for s in self if isinstance(s, Load))

    @property
    def inputs(self) -> tuple[str, ...]:
        """Distinct buffer names referenced by body Loads, in first-use order.

        Order matters: ``LoopOp.forward`` zips positional input arrays
        against this tuple, and the surrounding graph node's ``inputs`` list
        is set in the same order at lifting / fusion time. Using ``set(...)``
        here would scramble the mapping (set order is hash-based) and
        positional args would land on the wrong buffer names.
        """
        return tuple(dict.fromkeys(s.input for s in self.loads))

    @property
    def accums(self) -> tuple[Accum, ...]:
        """Unique reduce accumulators in the body, pre-order by first use.

        Walks the body; for each distinct ``Accum.name`` returns the first
        ``Accum`` stmt that targets it. Callers read ``accum.name``,
        ``accum.op`` (the combine), and ``accum.init`` (identity derived
        from op) directly without a separate summary type.
        """
        seen: dict[str, Accum] = {}
        for s in self:
            if isinstance(s, Accum) and s.name not in seen:
                seen[s.name] = s
        return tuple(seen.values())

    def analyze(self) -> LoopMeta:
        """One-pass summary of the body: name→def, name→scope, writes.

        Convenience for passes (e.g. the fusion splicer) that resolve SSA
        dependencies against a stable snapshot of the body tree.
        """
        defs: dict[str, Stmt] = {}
        scopes: dict[str, Scope] = {}
        reduce_axes: dict[str, Axis] = {}
        writes: list[tuple[Write, Scope]] = []

        def walk(stmts: tuple[Stmt, ...], scope: Scope) -> None:
            for s in stmts:
                if isinstance(s, Loop):
                    walk(s.body, scope.nest(s.axis))
                elif isinstance(s, Cond):
                    walk(s.body, scope)
                    walk(s.else_body, scope)
                elif isinstance(s, Accum):
                    defs[s.name] = s
                    # Binding scope excludes the reduce axis (the Accum is live
                    # after its reduce Loop completes).
                    if scope.enclosing:
                        reduce_axes[s.name] = scope.enclosing[-1]
                        scopes[s.name] = Scope(enclosing=scope.enclosing[:-1])
                    else:
                        scopes[s.name] = scope
                elif isinstance(s, (Load, Assign, Select)):
                    defs[s.name] = s
                    scopes[s.name] = scope
                elif isinstance(s, Write):
                    writes.append((s, scope))

        walk(self.body, Scope())
        return LoopMeta(
            op=self,
            defs=defs,
            scopes=scopes,
            reduce_axes=reduce_axes,
            writes=tuple(writes),
            live_axes=_compute_live_axes(defs, scopes, reduce_axes),
        )

    def map(self, fn: Callable[[Stmt], Stmt | None]) -> LoopOp:
        """Transform the body through ``map_body(self.body, fn)`` and wrap
        the result in a new ``LoopOp`` (which re-runs normalize + validate).

        Convenience for callers that already have a ``LoopOp`` in hand.
        Most internal passes work on raw ``tuple[Stmt, ...]`` bodies and
        use ``map_body`` directly.
        """
        return LoopOp(body=map_body(self.body, fn))

    def __iter__(self) -> Iterator[Stmt]:
        """Yield every ``Stmt`` in this op's body in pre-order (same as
        :func:`iter_body`). Enables ``for s in loop_op: ...`` and
        comprehensions like ``[s for s in loop_op if isinstance(s, Load)]``."""
        return iter_body(self.body)

    def forward(self, *inputs):
        """Evaluate the kernel body on numpy arrays — mirrors the other ``Op.forward`` methods.

        Enables graph-level execution via ``NumpyBackend`` after fusion has
        replaced tensor-IR ops with ``LoopOp`` nodes. Output shape is inferred
        from the kernel's sole ``Write`` statement; callers (``NumpyBackend``)
        reshape the result to the node's declared output shape, so keep-dim
        reductions and similar element-count-preserving variations work
        transparently.
        """
        import numpy as np

        from deplodock.compiler.ir.loop.interpret import execute_loop_op

        out_shape = self._infer_write_shape()
        bufs = self.inputs
        if len(inputs) != len(bufs):
            raise ValueError(f"LoopOp.forward: expected {len(bufs)} inputs (matching input_bufs={list(bufs)}), got {len(inputs)}")
        input_arrays = {name: np.asarray(x, dtype=np.float32) for name, x in zip(bufs, inputs, strict=True)}
        return execute_loop_op(self, input_arrays, out_shape)

    def _infer_write_shape(self) -> tuple[int, ...]:
        """Derive the output buffer shape from the kernel's ``Write`` index.

        Evaluates each dim's index Expr over the full iteration space; the
        per-dim extent is ``max(value) + 1``. Handles plain ``Var(axis)`` (→
        axis extent), ``Literal(c)`` (→ 1), and affine combinations uniformly.
        Falls back to the free-axis extents when no ``Write`` is present.
        """
        import numpy as np

        writes = [s for s in self if isinstance(s, Write)]
        if not writes:
            reduce_names = self.reduce_axis_names
            return tuple(a.extent for a in self.axes if a.name not in reduce_names)
        w = writes[0]
        env: dict[str, object] = {}
        for i, a in enumerate(self.axes):
            shape = [1] * len(self.axes)
            shape[i] = int(a.extent)
            env[a.name] = np.arange(int(a.extent)).reshape(shape)
        dims: list[int] = []
        for e in w.index:
            vals = e.eval(env)
            if isinstance(vals, np.ndarray):
                dims.append(int(vals.max()) + 1)
            else:
                dims.append(int(vals) + 1)
        return tuple(dims)


# ---------------------------------------------------------------------------
# LoopMeta — analysis summary produced by ``LoopOp.analyze``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopMeta:
    """Precomputed lookups over a ``LoopOp`` body.

    - ``op``: the source ``LoopOp`` this meta was built from.
    - ``defs``: SSA name → defining ``Stmt`` (``Load`` / ``Assign`` /
      ``Select`` / ``Accum``). A ``Write`` has no SSA name and is not here.
    - ``scopes``: SSA name → binding ``Scope`` (where the value is live
      after its def). For plain stmts this is the enclosing axis chain;
      for ``Accum`` the reduce axis is excluded — the Accum binds *after*
      the reduce Loop completes.
    - ``reduce_axes``: ``Accum`` name → its reduce ``Axis`` (the tail of
      the raw enclosing chain, stripped from ``scopes``). Only present
      for ``Accum`` defs.
    - ``writes``: every ``Write`` stmt paired with the ``Scope`` it sits
      in — one entry per output, in body order.
    - ``live_axes``: SSA name → axis names transitively reachable through
      Expr subtrees (``Load.index``, ``SelectBranch.select``) while resolving
      the stmt's dep chain. For an ``Accum``, the reduce axis is excluded
      since it gets freshened at emission time.
    """

    op: LoopOp
    defs: dict[str, Stmt]
    scopes: dict[str, Scope]
    reduce_axes: dict[str, Axis]
    writes: tuple[tuple[Write, Scope], ...]
    live_axes: dict[str, frozenset[str]]


def _compute_live_axes(
    defs: dict[str, Stmt],
    scopes: dict[str, Scope],  # noqa: ARG001 — kept for signature symmetry with analyze()
    reduce_axes: dict[str, Axis],
) -> dict[str, frozenset[str]]:
    """Axes reachable through Expr subtrees rooted at each SSA name.

    A dep's live axes are the union of its own Expr subtree's free vars plus
    the live axes of every SSA name it reads. ``Accum`` excludes its reduce
    axis, since that axis gets freshened per emission.
    """
    cache: dict[str, frozenset[str]] = {}
    in_progress: set[str] = set()

    def live(name: str) -> frozenset[str]:
        if name in cache:
            return cache[name]
        if name in in_progress:
            return frozenset()  # cycle: approximate as empty and let the caller union
        in_progress.add(name)
        stmt = defs.get(name)
        if stmt is None:
            in_progress.discard(name)
            cache[name] = frozenset()
            return cache[name]
        own: frozenset[str] = frozenset()
        if isinstance(stmt, Load):
            for e in stmt.index:
                own |= free_vars(e)
        elif isinstance(stmt, Select):
            for b in stmt.branches:
                own |= free_vars(b.select) | live(b.value)
        elif isinstance(stmt, Assign):
            for a in stmt.args:
                own |= live(a)
        elif isinstance(stmt, Accum):
            own |= live(stmt.value)
            ra = reduce_axes.get(name)
            if ra is not None:
                own -= {ra.name}
        in_progress.discard(name)
        cache[name] = own
        return own

    for n in defs:
        live(n)
    return cache


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def pretty_print(loop: LoopOp, port_buffers: list[str] | None = None, indent: str = "") -> str:
    """Render a ``LoopOp`` as an explicit nested-loop program.

    Each ``Loop`` block renders as ``for X in 0..N:  # kind``. Body Loads
    render their ``source`` name directly (a string identifying the source
    buffer); ``port_buffers`` is kept for back-compat but ignored.
    """
    del port_buffers  # source is the buf name, intrinsic to Load now
    lines: list[str] = []
    _render_body(loop.body, indent, lines)
    return "\n".join(lines)


def _render_body(
    stmts: tuple[Stmt, ...],
    indent: str,
    lines: list[str],
) -> None:
    """Render a body tuple (recursive for nested ``Loop``)."""
    for stmt in stmts:
        if isinstance(stmt, Assign):
            args = ", ".join(stmt.args)
            lines.append(f"{indent}{stmt.name} = {stmt.op.name}({args})")
        elif isinstance(stmt, Load):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}{stmt.name} = load {stmt.input}[{idx}]")
        elif isinstance(stmt, Accum):
            lines.append(f"{indent}{stmt.name} <- {stmt.op.name}({stmt.name}, {stmt.value})")
        elif isinstance(stmt, Write):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}{stmt.output}[{idx}] = {stmt.value}")
        elif isinstance(stmt, Select):
            for bi, br in enumerate(stmt.branches):
                prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                lines.append(f"{indent}{prefix} {br.value} when ({render_expr(br.select)})")
        elif isinstance(stmt, Loop):
            a = stmt.axis
            # Kind is structural: a Loop whose immediate body contains an Accum is a reduce Loop.
            kind = "reduce" if any(isinstance(s, Accum) for s in stmt.body) else "free"
            lines.append(f"{indent}for {a.name} in 0..{a.extent}:  # {kind}")
            _render_body(stmt.body, indent + "    ", lines)
        elif isinstance(stmt, Cond):
            lines.append(f"{indent}if ({render_expr(stmt.cond)}):")
            _render_body(stmt.body, indent + "    ", lines)
            if stmt.else_body:
                lines.append(f"{indent}else:")
                _render_body(stmt.else_body, indent + "    ", lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness, SSA invariants, accumulator pairing.

    Body validation recurses into ``Loop`` blocks. SSA names defined inside a
    ``Loop.body`` are scoped to that body — invisible outside — except for
    ``Accum.name`` names, which are bound post-Loop in the enclosing
    scope. Axis names are validated as a flat set for uniqueness across
    the kernel.
    """
    # LoopOp.axes uniqueness (stored axes).
    all_axis_names: set[str] = set()
    for a in loop.axes:
        if a.name in all_axis_names:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        all_axis_names.add(a.name)

    # Accumulator op-consistency: all Updates targeting the same name must
    # share the same combine op (they're folding into the same accumulator).
    seen_accums: dict[str, Accum] = {}
    for info in loop.accums:
        if info.name in seen_accums:
            raise ValueError(f"Accumulator: duplicate name {info.name!r}")
        seen_accums[info.name] = info

    # Op-consistency across repeated Updates to same target.
    target_ops: dict[str, ElementwiseImpl] = {}

    def _walk(stmts: tuple[Stmt, ...], defined: set[str]) -> set[str]:
        """Validate a body scope. Returns the set of ``Accum.name`` names
        that propagate to the enclosing scope after this body completes
        (they carry the accumulator's finalized value).

        Sibling Loops with the same axis name are fine — that's the softmax
        pattern (two sequential sweeps over the same axis). Nested Loops
        that re-use an outer axis name are left to backend codegen to
        disambiguate (the axis is only referenced as Var(name), and SSA
        scoping handles variable shadowing).
        """
        exported_accs: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, Assign):
                for arg in stmt.args:
                    if arg not in defined:
                        raise ValueError(f"Assign {stmt.name!r}: arg {arg!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Assign {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Load):
                # Load is a binding site — introduces its SSA name. ``source``
                # is the producing graph node's id.
                if not isinstance(stmt.input, str) or not stmt.input:
                    raise ValueError(f"Load {stmt.name!r}: source {stmt.input!r} must be a non-empty string")
                if stmt.name in defined:
                    raise ValueError(f"Load {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Accum):
                if stmt.value not in defined and stmt.name != stmt.value:
                    raise ValueError(f"Accum: value {stmt.value!r} not defined")
                # Each Accum implicitly declares / extends an accumulator.
                # Repeated Updates to same target must share op.
                prev_op = target_ops.get(stmt.name)
                if prev_op is not None and prev_op.name != stmt.op.name:
                    raise ValueError(f"Accum {stmt.name!r}: op {stmt.op.name!r} conflicts with earlier Accum's op {prev_op.name!r}")
                target_ops[stmt.name] = stmt.op
                defined.add(stmt.name)
                exported_accs.add(stmt.name)
            elif isinstance(stmt, Select):
                for b in stmt.branches:
                    if b.value not in defined:
                        raise ValueError(f"Select {stmt.name!r}: branch value {b.value!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Select {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Write):
                if stmt.value not in defined:
                    raise ValueError(f"Write: value {stmt.value!r} not defined")
            elif isinstance(stmt, Loop):
                # SSA names defined inside Loop.body are scoped to that body,
                # except Accum.names which carry the finalized reduced
                # value out to the enclosing scope.
                inner_exports = _walk(stmt.body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
            elif isinstance(stmt, Cond):
                # Same scoping rules as Loop: inner Assign / Select names are
                # local; Accum names export. Validate both branches.
                inner_exports = _walk(stmt.body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
                inner_exports = _walk(stmt.else_body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
        return exported_accs

    _walk(loop.body, set())


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def iter_body(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    """Yield every ``Stmt`` in ``body`` in pre-order, recursing through
    ``Loop`` wrappers. ``Loop`` nodes are yielded themselves *and* then
    their body is walked — callers that want only leaves can filter with
    ``isinstance`` or use :func:`flatten_body`.

    The default primitive for collection walks; composes naturally with
    comprehensions and ``any()`` / ``sum()`` without allocating a new
    body tuple.
    """
    for s in body:
        yield s
        if isinstance(s, Loop):
            yield from iter_body(s.body)
        elif isinstance(s, Cond):
            yield from iter_body(s.body)
            yield from iter_body(s.else_body)


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
