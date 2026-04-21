"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    body   : tuple[Stmt, ...]   — SSA: Assign | Accum | Write | Select | Loop | Load
    axes   : computed property  — iteration space walked from body's Loop tree
    loads  : computed property  — body-form Load stmts (external reads)
    accums : computed property  — Accum stmts seeded from the body

Iteration is explicit via ``Loop(axis, body)`` statements. Each ``Loop``
is one iteration dimension; ``Loop.body`` runs ``axis.extent`` times.
Reduce-kind Loops fold ``Accum`` statements into the accumulator named
by each ``Accum.name``. Free-kind Loops run in parallel with no
accumulator folding. Reading top-to-bottom matches execution order.

SSA names defined by ``Assign`` / ``Select`` / ``Load`` inside a
``Loop.body`` are scoped to that body — only accumulator state crosses
Loop boundaries (via ``Accum``). Invariants (unique names, defined-
before-use, accumulator liveness) are enforced by ``LoopOp.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr, Literal, Sigma, free_vars
from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.tensor_ir import ElementwiseOp

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

    def rewrite(self, new_name: str, resolved: dict[str, str], sigma: Sigma) -> Stmt:
        """Return a copy with ``new_name`` as the SSA binding, SSA refs
        remapped via ``resolved``, and ``sigma`` substituted into every
        ``Expr`` subterm. Only meaningful for SSA-binding stmts (``Load``,
        ``Assign``, ``Select``); other stmt kinds raise ``NotImplementedError``.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Load(Stmt):
    """Read a value from an external input buffer into an SSA name.

    Each external-buffer read is an explicit body statement. ``source`` is
    the index into ``LoopLaunch.input_names`` identifying which external
    buffer; ``index`` is the dim-wise access pattern over the enclosing
    axes. The produced SSA ``name`` is a regular value downstream stmts
    read.
    """

    name: str
    source: int
    index: tuple[Expr, ...]

    def deps(self) -> tuple[str, ...]:
        return ()

    def rewrite(self, new_name: str, resolved: dict[str, str], sigma: Sigma) -> Stmt:
        return Load(name=new_name, source=self.source, index=tuple(sigma.apply(e) for e in self.index))


# ---------------------------------------------------------------------------
# Accumulator identity table
# ---------------------------------------------------------------------------
#
# ``Accum`` stmts carry their own reduce op — ``add`` / ``max`` / ``min`` /
# ``mul``. The init value for the accumulator is derived from the op via
# this table. No separate declaration stmt is needed: an ``Accum(target,
# value, op=...)`` inside a reduce ``Loop`` implicitly initializes the
# accumulator to ``ACCUM_IDENTITY[op.fn]`` before the sweep and folds via
# ``op`` at each iteration. The target SSA name is bound in the enclosing
# scope after the Loop.

ACCUM_IDENTITY: dict[str, float] = {
    "add": 0.0,
    "sum": 0.0,
    "max": -1e30,
    "min": 1e30,
    "mul": 1.0,
    "prod": 1.0,
}


@dataclass(frozen=True)
class Assign(Stmt):
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is always an ``ElementwiseOp`` (reductions live in ``Accum``).
    ``args`` reference ``$N`` ports, an ``Accum.name`` (reads current /
    finalized acc value), or prior SSA names.
    """

    name: str
    op: ElementwiseOp
    args: tuple[str, ...]

    def deps(self) -> tuple[str, ...]:
        return self.args

    def rewrite(self, new_name: str, resolved: dict[str, str], sigma: Sigma) -> Stmt:
        return Assign(name=new_name, op=self.op, args=tuple(resolved[a] for a in self.args))


@dataclass(frozen=True)
class Accum(Stmt):
    """Reduce accumulator — declares-and-folds in one statement.

    Semantics: ``name = op(name, value)`` inside the enclosing reduce
    ``Loop``. Before the first iteration ``name`` is initialized to
    ``ACCUM_IDENTITY[op.fn]``. After the Loop completes, ``name`` is an
    SSA binding visible in the enclosing scope, carrying the finalized
    reduced value.

    ``op`` is one of ``add`` / ``max`` / ``min`` / ``mul`` — it defines
    both the combine operation and the accumulator's identity value.
    Multiple ``Accum`` stmts targeting the same ``name`` in one reduce
    Loop must agree on ``op`` (they're folding into the same accumulator).

    Default op is ``add`` — fixtures that sum values can omit ``op=``;
    ``max`` / ``min`` / ``mul`` must be passed explicitly.
    """

    name: str
    value: str
    op: ElementwiseOp = field(default_factory=lambda: ElementwiseOp("add"))

    @property
    def init(self) -> Expr:
        """Identity value for the accumulator (``ACCUM_IDENTITY[op.fn]``).

        Exposed as an attribute so readers that need an ``Expr`` init (loop
        backend, emit) don't have to re-implement the lookup.
        """
        return Literal(ACCUM_IDENTITY.get(self.op.fn, 0.0))

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def rewrite(self, new_name: str, resolved: dict[str, str], sigma: Sigma) -> Stmt:
        return Accum(name=new_name, value=resolved[self.value], op=self.op)


@dataclass(frozen=True)
class Write(Stmt):
    """Write an SSA value to output buffer ``output`` at position ``index``.

    ``output`` is an integer index into the program-level output-name
    tuple (``LoopLaunch.output_name`` is the sole output in v1, so
    ``output=0`` is common). ``index`` uses axis Vars to compute the
    per-dim offset. ``value`` references an SSA name available at this
    point in the body (Assign, Accum, or ``$N``).
    """

    output: int
    index: tuple[Expr, ...]
    value: str

    def deps(self) -> tuple[str, ...]:
        return (self.value,)


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

    def rewrite(self, new_name: str, resolved: dict[str, str], sigma: Sigma) -> Stmt:
        return Select(
            name=new_name,
            branches=tuple(SelectBranch(value=resolved[b.value], select=sigma.apply(b.select)) for b in self.branches),
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

        def walk(stmts: tuple[Stmt, ...]) -> None:
            for s in stmts:
                if isinstance(s, Loop):
                    if s.axis.name not in seen:
                        seen[s.axis.name] = s.axis
                    walk(s.body)

        walk(self.body)
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
        result: list[Load] = []

        def walk(stmts: tuple[Stmt, ...]) -> None:
            for s in stmts:
                if isinstance(s, Load):
                    result.append(s)
                elif isinstance(s, Loop):
                    walk(s.body)

        walk(self.body)
        return tuple(result)

    @property
    def num_inputs(self) -> int:
        """Number of external input buffers referenced by body Loads.

        Derived as ``max(load.source) + 1``, or 0 when no Loads are present.
        This is the count ``LoopLaunch.input_names`` must provide.
        """
        loads = self.loads
        return max((ld.source for ld in loads), default=-1) + 1

    @property
    def accums(self) -> tuple[Accum, ...]:
        """Unique reduce accumulators in the body, pre-order by first use.

        Walks the body; for each distinct ``Accum.name`` returns the first
        ``Accum`` stmt that targets it. Callers read ``accum.name``,
        ``accum.op`` (the combine), and ``accum.init`` (identity derived
        from op) directly without a separate summary type.
        """
        seen: dict[str, Accum] = {}

        def walk(stmts: tuple[Stmt, ...]) -> None:
            for s in stmts:
                if isinstance(s, Accum):
                    if s.name not in seen:
                        seen[s.name] = s
                elif isinstance(s, Loop):
                    walk(s.body)

        walk(self.body)
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

        from deplodock.compiler.backend.loop.backend import execute_loop_op

        out_shape = self._infer_write_shape()
        input_arrays = [np.asarray(x, dtype=np.float32) for x in inputs]
        return execute_loop_op(self, input_arrays, out_shape)

    def _infer_write_shape(self) -> tuple[int, ...]:
        """Derive the output buffer shape from the kernel's ``Write`` index.

        Evaluates each dim's index Expr over the full iteration space; the
        per-dim extent is ``max(value) + 1``. Handles plain ``Var(axis)`` (→
        axis extent), ``Literal(c)`` (→ 1), and affine combinations uniformly.
        Falls back to the free-axis extents when no ``Write`` is present.
        """
        import numpy as np

        writes = [s for s in flatten_body(self.body) if isinstance(s, Write)]
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
    render as ``name = load buf[index...]`` where ``buf`` is the matching
    entry in ``port_buffers`` (when supplied) or a ``$source`` fallback.
    """
    nbufs = loop.num_inputs
    buffers = list(port_buffers) if port_buffers else [f"${i}" for i in range(nbufs)]
    while len(buffers) < nbufs:
        buffers.append(f"${len(buffers)}")

    lines: list[str] = []
    _render_body(loop.body, indent, buffers, lines)
    return "\n".join(lines)


def _render_body(
    stmts: tuple[Stmt, ...],
    indent: str,
    buffers: list[str],
    lines: list[str],
) -> None:
    """Render a body tuple (recursive for nested ``Loop``)."""
    for stmt in stmts:
        if isinstance(stmt, Assign):
            args = ", ".join(stmt.args)
            lines.append(f"{indent}{stmt.name} = {stmt.op.fn}({args})")
        elif isinstance(stmt, Load):
            buf = buffers[stmt.source] if 0 <= stmt.source < len(buffers) else f"src{stmt.source}"
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}{stmt.name} = load {buf}[{idx}]")
        elif isinstance(stmt, Accum):
            lines.append(f"{indent}{stmt.name} <- {stmt.op.fn}({stmt.name}, {stmt.value})")
        elif isinstance(stmt, Write):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}out{stmt.output}[{idx}] = {stmt.value}")
        elif isinstance(stmt, Select):
            for bi, br in enumerate(stmt.branches):
                prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                lines.append(f"{indent}{prefix} {br.value} when ({render_expr(br.select)})")
        elif isinstance(stmt, Loop):
            a = stmt.axis
            # Kind is structural: a Loop whose immediate body contains an Accum is a reduce Loop.
            kind = "reduce" if any(isinstance(s, Accum) for s in stmt.body) else "free"
            lines.append(f"{indent}for {a.name} in 0..{a.extent}:  # {kind}")
            _render_body(stmt.body, indent + "    ", buffers, lines)


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
    target_ops: dict[str, ElementwiseOp] = {}

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
                # indexes into ``LoopLaunch.input_names`` at the program level,
                # which is checked against the launch's input count there.
                if stmt.source < 0:
                    raise ValueError(f"Load {stmt.name!r}: source {stmt.source} must be non-negative")
                if stmt.name in defined:
                    raise ValueError(f"Load {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Accum):
                if stmt.value not in defined and stmt.name != stmt.value:
                    raise ValueError(f"Accum: value {stmt.value!r} not defined")
                # Each Accum implicitly declares / extends an accumulator.
                # Repeated Updates to same target must share op.
                prev_op = target_ops.get(stmt.name)
                if prev_op is not None and prev_op.fn != stmt.op.fn:
                    raise ValueError(f"Accum {stmt.name!r}: op {stmt.op.fn!r} conflicts with earlier Accum's op {prev_op.fn!r}")
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
        return exported_accs

    _walk(loop.body, set())


# ---------------------------------------------------------------------------
# Tree walk helpers
# ---------------------------------------------------------------------------


def iter_loops(body: tuple[Stmt, ...]) -> list[Loop]:
    """Return every ``Loop`` in ``body`` in pre-order tree traversal."""
    out: list[Loop] = []

    def walk(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                out.append(s)
                walk(s.body)

    walk(body)
    return out


def flatten_body(body: tuple[Stmt, ...]) -> list[Stmt]:
    """Extract leaf statements (``Assign`` / ``Accum`` / ``Write`` / ``Select``)
    from ``body``, recursing through ``Loop`` blocks.

    Useful for consumers that operate on the flat statement sequence
    regardless of block structure (numpy whole-tensor interpreter,
    Accum-boundary segmentation in ``loop.plan.analyze_kernel``). Leaf
    statements come out in pre-order (parent before children); the Loop
    wrappers are transparent — their ``axis`` metadata is discarded at
    this level.
    """
    out: list[Stmt] = []

    def walk(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                walk(s.body)
            else:
                out.append(s)

    walk(body)
    return out


# ---------------------------------------------------------------------------
# Flat → nested normalization shim (used by merge and legacy test fixtures)
# ---------------------------------------------------------------------------


def flat_body_to_nested(
    axes: tuple[Axis, ...],
    body: tuple[Stmt, ...],
    reduce_axis_names: frozenset[str] | set[str] = frozenset(),
) -> tuple[Stmt, ...]:
    """Wrap a flat SSA body into the nested ``Loop``-block form.

    Pure body transform — takes the flat body + axes hint, returns a new
    body tuple. ``reduce_axis_names`` identifies which axes should become
    reduce Loops (wrapping ``Accum`` segments); the rest are free.

    Idempotent: if the body already contains any ``Loop``, returns it
    unchanged.

    Layout:

    - Outer free-axis ``Loop``s wrap everything, innermost = last free axis.
    - Inside the innermost free level, one ``Loop(reduce_axis, body=segment)``
      per segment produced by splitting the flat body at ``Accum`` boundaries.
    - Post-reduce ``Assign`` / ``Select`` / ``Write`` statements (everything
      after the last ``Accum``) sit after the reduce Loops.
    - Non-reduce flat bodies are wrapped only in free Loops.
    """
    if any(isinstance(s, Loop) for s in body):
        return tuple(body)

    reduce_axis_names = frozenset(reduce_axis_names)
    has_update = any(isinstance(s, Accum) for s in body)

    # If caller didn't specify reduce axis names but the body has Updates,
    # infer from body structure: axes that appear in the Write index are
    # free; axes that appear ONLY on the Accum's data flow (via some
    # Assign/Load reference) but NOT in any Write index are reduce axes.
    # This matches the historical signature where ``Axis.kind`` encoded
    # the distinction. Singleton axes (extent 1) that don't appear in the
    # Write are treated as free (degenerate) rather than reduce — the
    # Write typically uses Literal(0) for them.
    if has_update and not reduce_axis_names:
        from deplodock.compiler.ir.expr import BinOp as _BinOp
        from deplodock.compiler.ir.expr import Var as _Var

        write_axis_names: set[str] = set()

        def _walk_expr(e) -> None:
            if isinstance(e, _Var):
                write_axis_names.add(e.name)
            elif isinstance(e, _BinOp):
                _walk_expr(e.left)
                _walk_expr(e.right)

        for s in body:
            if isinstance(s, Write):
                for e in s.index:
                    _walk_expr(e)
        reduce_axis_names = frozenset(a.name for a in axes if a.name not in write_axis_names and int(a.extent) > 1)

    free_axes = [a for a in axes if a.name not in reduce_axis_names]

    if not has_update:
        # Pure pointwise: wrap in free Loops only.
        wrapped: tuple[Stmt, ...] = tuple(body)
        for a in reversed(free_axes):
            wrapped = (Loop(axis=a, body=wrapped),)
        return wrapped

    # Reduce body: split at Updates into segments. Each Accum terminates
    # a segment that gets wrapped in a reduce ``Loop``; the Accum implicitly
    # declares its accumulator (via its ``op`` field), which becomes visible
    # in the enclosing scope after the Loop completes.
    reduce_axes = [a for a in axes if a.name in reduce_axis_names]
    if len(reduce_axes) != 1:
        # Malformed or multi-reduce flat — leave unchanged; caller/validator flags.
        return tuple(body)
    reduce_axis = reduce_axes[0]

    segments: list[list[Stmt]] = []
    current: list[Stmt] = []
    for stmt in body:
        current.append(stmt)
        if isinstance(stmt, Accum):
            segments.append(current)
            current = []
    tail = current  # post-last-Accum stmts

    nested: list[Stmt] = []
    for seg in segments:
        nested.append(Loop(axis=reduce_axis, body=tuple(seg)))
    nested.extend(tail)

    wrapped = tuple(nested)
    for a in reversed(free_axes):
        wrapped = (Loop(axis=a, body=wrapped),)
    return wrapped
