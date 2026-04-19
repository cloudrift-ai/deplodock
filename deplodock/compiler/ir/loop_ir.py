"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    inputs : tuple[Port, ...]   — computed property derived from body Loads
    body   : tuple[Stmt, ...]   — SSA: Assign | Update | Write | Select | Loop | Load | AccumDecl
    axes   : computed property  — iteration space walked from body's Loop tree

Iteration is explicit via ``Loop(axis, body)`` statements. Each ``Loop``
is one iteration dimension; ``Loop.body`` runs ``axis.extent`` times.
Reduce-kind Loops fold ``Update`` statements into an ``AccumDecl``
declared inline in the body. Free-kind Loops run in parallel with no
accumulator folding. Reading top-to-bottom matches execution order.

SSA names defined by ``Assign`` / ``Select`` / ``Load`` inside a
``Loop.body`` are scoped to that body — only ``AccumDecl`` state crosses
Loop boundaries (via ``Update``). Invariants (unique names, defined-
before-use, accumulator liveness) are enforced by ``LoopOp.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.tensor_ir import ElementwiseOp

# ---------------------------------------------------------------------------
# Axis — named iteration variable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """One named iteration variable at the loop level.

    Referenced from ``Expr`` subtrees (inside ``Port.index`` etc.) by
    ``Var(name)``. Free vs reduce is inferred structurally from the body:
    a ``Loop`` over this axis is a reduce loop iff its body (recursively)
    contains an ``Update``. See ``LoopOp.reduce_axis_names``.

    ``extent`` is a static integer in v1; future revisions may allow an
    ``Expr`` for dynamic batch/seq dims.
    """

    name: str
    extent: int


# ---------------------------------------------------------------------------
# Port — external-buffer access pattern
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """Access pattern for one external input buffer.

    ``index`` is a tuple of ``Expr`` — one per dimension of the external
    buffer. Each Expr computes the offset into its buffer dim from the
    enclosing ``LoopOp.axes`` (via ``Var(axis_name)``), possibly combined
    with literals and arithmetic for transposes, broadcasts, slices.
    """

    index: tuple[Expr, ...] = ()


# ---------------------------------------------------------------------------
# Body statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Load:
    """Read a value from an external input buffer into an SSA name.

    Replaces the ``$N`` port-reference convention: instead of a side-channel
    ``LoopOp.inputs`` tuple whose entries are indexed by string sentinels,
    each read becomes an explicit statement in the body. ``source`` is the
    index into ``LoopLaunch.input_names`` identifying which external buffer;
    ``index`` is the dim-wise access pattern over the enclosing axes.
    The produced SSA ``name`` is a regular value downstream stmts read.
    """

    name: str
    source: int
    index: tuple[Expr, ...]


@dataclass(frozen=True)
class AccumDecl:
    """Declare a reduce accumulator at a body position.

    The accumulator is live from this statement through the end of the
    enclosing ``Loop.body`` scope. ``Update`` statements with matching
    ``target`` name fold values into it via ``combine``; stmts after the
    reduce sweep read the finalized value by SSA name.

    Semantics per ``Update``: ``acc = combine(acc, value)``, initialized
    to ``init`` before the reduce sweep. One ``AccumDecl`` per accumulator;
    duplicates are rejected by the validator.
    """

    name: str
    combine: ElementwiseOp
    init: Expr


# Backwards-compatible alias. ``Accumulator`` used to be a side-channel
# ``LoopOp.accumulators`` tuple entry; now it's just an ``AccumDecl`` living
# in the body. Readers / tests that still import ``Accumulator`` get the
# same dataclass — constructing one is interchangeable with constructing an
# ``AccumDecl``.
Accumulator = AccumDecl


@dataclass(frozen=True)
class Assign:
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is always an ``ElementwiseOp`` (reductions have moved to
    ``Accumulator`` + ``Update``). ``args`` reference ``$N`` ports,
    ``Accumulator.name`` (reads current / finalized acc value), or prior
    SSA names.
    """

    name: str
    op: ElementwiseOp
    args: tuple[str, ...]


@dataclass(frozen=True)
class Update:
    """Fold a value into an Accumulator.

    Semantics: ``acc = combine(acc, value)`` using the Accumulator's
    ``combine`` op. ``target`` must name an ``Accumulator``; ``value``
    references an SSA name (Assign, prior Update target finalized, or
    ``$N``).

    The positionally-last Update in ``LoopOp.body`` marks the end of the
    reduce sweep: statements after read finalized accumulators.
    """

    target: str
    value: str


@dataclass(frozen=True)
class Write:
    """Write an SSA value to output buffer ``output`` at position ``index``.

    ``output`` is an integer index into the program-level output-name
    tuple (``LoopLaunch.output_name`` is the sole output in v1, so
    ``output=0`` is common). ``index`` uses axis Vars to compute the
    per-dim offset. ``value`` references an SSA name available at this
    point in the body (Assign, Accumulator, or ``$N``).
    """

    output: int
    index: tuple[Expr, ...]
    value: str


@dataclass(frozen=True)
class SelectBranch:
    """One branch of a ``Select`` body statement."""

    value: str  # SSA name when predicate holds
    select: Expr  # predicate over axis Vars


@dataclass(frozen=True)
class Select:
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


@dataclass(frozen=True)
class Loop:
    """Explicit iteration block — one loop over an axis.

    ``body`` executes ``axis.extent`` times, once per axis value. Reduce-kind
    Loops fold any ``Update`` statements in their body into the enclosing
    ``Accumulator`` (one sweep over the axis per accumulator). Free-kind
    Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``Accumulator`` state (written via ``Update``) crosses the Loop
    boundary, carrying the finalized reduced value.
    """

    axis: Axis
    body: tuple[Stmt, ...]


Stmt = Assign | Update | Write | Select | Loop | Load | AccumDecl


# ---------------------------------------------------------------------------
# LoopOp
# ---------------------------------------------------------------------------


@dataclass
class LoopOp(Op):
    """One kernel's worth of computation as an SSA program over named axes.

    ``axes`` is a computed property over the body's ``Loop`` tree — the tree
    is the single source of truth. See the :attr:`axes` docstring for the
    collection order. ``inputs`` and ``accumulators`` are computed properties
    derived from body-form ``Load`` and ``AccumDecl`` statements.
    """

    inputs: tuple[Port, ...] = ()
    body: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
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
        """Names of axes whose Loop directly wraps an ``Update``.

        Derived from body structure, not stored metadata: a Loop is a
        'reduce Loop' iff its immediate body contains an Update.
        """
        names: set[str] = set()

        def walk(stmts: tuple[Stmt, ...], innermost: str | None) -> None:
            for s in stmts:
                if isinstance(s, Update) and innermost is not None:
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
    def accum_decls(self) -> tuple[AccumDecl, ...]:
        """All ``AccumDecl`` statements in the body, pre-order.

        Scope is lexical — each AccumDecl is live from its position through
        the end of the enclosing ``Loop.body`` (or the kernel body, at top
        level).
        """
        result: list[AccumDecl] = []

        def walk(stmts: tuple[Stmt, ...]) -> None:
            for s in stmts:
                if isinstance(s, AccumDecl):
                    result.append(s)
                elif isinstance(s, Loop):
                    walk(s.body)

        walk(self.body)
        return tuple(result)

    @property
    def accumulators(self) -> tuple[AccumDecl, ...]:
        """Alias for :attr:`accum_decls` — kept for readers that still use
        ``loop.accumulators``. New code should use ``accum_decls`` directly.
        """
        return self.accum_decls

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
# Pretty-printing
# ---------------------------------------------------------------------------


def pretty_print(loop: LoopOp, port_buffers: list[str] | None = None, indent: str = "") -> str:
    """Render a ``LoopOp`` as an explicit nested-loop program.

    Each ``Loop`` block renders as ``for X in 0..N:  # kind``. ``$N`` port
    references (from legacy test fixtures that still emit them) are
    rendered as ``buf[...]`` when ``port_buffers`` is supplied, otherwise
    as ``$N[...]`` with the port's index exprs inlined.
    """
    ports = loop.inputs
    # Buffer names used when rendering Load stmts and $N port refs.
    nbufs = max(len(ports), loop.num_inputs)
    buffers = list(port_buffers) if port_buffers else [f"${i}" for i in range(nbufs)]
    while len(buffers) < nbufs:
        buffers.append(f"${len(buffers)}")
    accs_by_name: dict[str, AccumDecl] = {d.name: d for d in loop.accum_decls}

    def render_arg(name: str) -> str:
        if name.startswith("$"):
            try:
                pi = int(name[1:])
            except ValueError:
                return name
            if 0 <= pi < len(ports):
                idx = ", ".join(render_expr(e) for e in ports[pi].index)
                return f"{buffers[pi]}[{idx}]" if idx else buffers[pi]
        return name

    lines: list[str] = []
    _render_body(loop.body, indent, accs_by_name, render_arg, buffers, lines)
    return "\n".join(lines)


def _render_body(
    stmts: tuple[Stmt, ...],
    indent: str,
    accs_by_name: dict[str, AccumDecl],
    render_arg,
    buffers: list[str],
    lines: list[str],
) -> None:
    """Render a body tuple (recursive for nested ``Loop``)."""
    for stmt in stmts:
        if isinstance(stmt, Assign):
            args = ", ".join(render_arg(a) for a in stmt.args)
            lines.append(f"{indent}{stmt.name} = {stmt.op.fn}({args})")
        elif isinstance(stmt, Load):
            buf = buffers[stmt.source] if 0 <= stmt.source < len(buffers) else f"src{stmt.source}"
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}{stmt.name} = load {buf}[{idx}]")
        elif isinstance(stmt, AccumDecl):
            lines.append(f"{indent}{stmt.name} = {render_expr(stmt.init)}  # accum, combine={stmt.combine.fn}")
        elif isinstance(stmt, Update):
            acc = accs_by_name.get(stmt.target)
            fn = acc.combine.fn if acc is not None else "?"
            lines.append(f"{indent}{stmt.target} <- {fn}({stmt.target}, {render_arg(stmt.value)})")
        elif isinstance(stmt, Write):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}out{stmt.output}[{idx}] = {render_arg(stmt.value)}")
        elif isinstance(stmt, Select):
            for bi, br in enumerate(stmt.branches):
                prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                lines.append(f"{indent}{prefix} {render_arg(br.value)} when ({render_expr(br.select)})")
        elif isinstance(stmt, Loop):
            a = stmt.axis
            # Kind is structural: a Loop whose immediate body contains an Update is a reduce Loop.
            kind = "reduce" if any(isinstance(s, Update) for s in stmt.body) else "free"
            lines.append(f"{indent}for {a.name} in 0..{a.extent}:  # {kind}")
            _render_body(stmt.body, indent + "    ", accs_by_name, render_arg, buffers, lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness, SSA invariants, accumulator pairing.

    Body validation recurses into ``Loop`` blocks. SSA names defined inside a
    ``Loop.body`` are scoped to that body — invisible outside. Only
    ``AccumDecl`` state crosses Loop boundaries (via ``Update``).
    Axis names (from the stored ``LoopOp.axes`` and from any nested Loops) are
    validated as a flat set for uniqueness across the kernel.
    """
    # LoopOp.axes uniqueness (stored axes).
    all_axis_names: set[str] = set()
    for a in loop.axes:
        if a.name in all_axis_names:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        all_axis_names.add(a.name)

    # Accumulator uniqueness — AccumDecls walked from the body tree.
    accumulators: dict[str, AccumDecl] = {}
    for decl in loop.accum_decls:
        if decl.name in accumulators:
            raise ValueError(f"Accumulator: duplicate name {decl.name!r}")
        accumulators[decl.name] = decl

    # Kernel-scope names (visible everywhere): $N ports (legacy form, used by
    # a few test fixtures that still spell args as ``$0``). Body-level Load /
    # AccumDecl bindings get added as the walk encounters them.
    kernel_scope: set[str] = {f"${i}" for i in range(len(loop.inputs))}

    update_targets: set[str] = set()

    def _walk(stmts: tuple[Stmt, ...], defined: set[str]) -> None:
        """Validate a body scope. ``defined`` contains in-scope SSA names.
        Sibling Loops with the same axis name are fine — that's the softmax
        pattern (two sequential sweeps over the same axis). Nested Loops
        that re-use an outer axis name are left to backend codegen to
        disambiguate (the axis is only referenced as Var(name), and SSA
        scoping handles variable shadowing).
        """
        for stmt in stmts:
            if isinstance(stmt, Assign):
                for arg in stmt.args:
                    if arg not in defined:
                        raise ValueError(f"Assign {stmt.name!r}: arg {arg!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Assign {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Load):
                # Load is a binding site — introduces its SSA name. Source-index
                # bounds check happens here only when the legacy inputs field is
                # populated; otherwise num_inputs is self-derived from Loads.
                if loop.inputs and (stmt.source < 0 or stmt.source >= len(loop.inputs)):
                    raise ValueError(f"Load {stmt.name!r}: source {stmt.source} out of range 0..{len(loop.inputs)}")
                if stmt.name in defined:
                    raise ValueError(f"Load {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, AccumDecl):
                # AccumDecl uniqueness was checked above (across field + body).
                # Here just confirm the name becomes an in-scope readable value.
                defined.add(stmt.name)
            elif isinstance(stmt, Update):
                if stmt.target not in accumulators:
                    raise ValueError(f"Update.target {stmt.target!r} does not name an Accumulator")
                if stmt.value not in defined:
                    raise ValueError(f"Update: value {stmt.value!r} not defined")
                update_targets.add(stmt.target)
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
                # SSA names defined inside Loop.body are scoped to that body.
                _walk(stmt.body, set(defined))

    _walk(loop.body, set(kernel_scope))

    # Accumulator liveness: every acc must be updated at least once (anywhere in the tree).
    for acc_name in accumulators:
        if acc_name not in update_targets:
            raise ValueError(f"Accumulator {acc_name!r}: declared but never Updated")


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
    """Extract leaf statements (``Assign`` / ``Update`` / ``Write`` / ``Select``)
    from ``body``, recursing through ``Loop`` blocks.

    Useful for consumers that operate on the flat statement sequence
    regardless of block structure (numpy whole-tensor interpreter,
    Update-boundary segmentation in ``loop_plan.analyze_kernel``). Leaf
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
    reduce Loops (wrapping ``Update`` segments); the rest are free.

    Idempotent: if the body already contains any ``Loop``, returns it
    unchanged.

    Layout:

    - Outer free-axis ``Loop``s wrap everything, innermost = last free axis.
    - Inside the innermost free level, one ``Loop(reduce_axis, body=segment)``
      per segment produced by splitting the flat body at ``Update`` boundaries.
    - Post-reduce ``Assign`` / ``Select`` / ``Write`` statements (everything
      after the last ``Update``) sit after the reduce Loops.
    - Non-reduce flat bodies are wrapped only in free Loops.
    """
    if any(isinstance(s, Loop) for s in body):
        return tuple(body)

    reduce_axis_names = frozenset(reduce_axis_names)
    has_update = any(isinstance(s, Update) for s in body)

    # If caller didn't specify reduce axis names but the body has Updates,
    # infer from body structure: axes that appear in the Write index are
    # free; axes that appear ONLY on the Update's data flow (via some
    # Assign/Port reference) but NOT in any Write index are reduce axes.
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

    # Reduce body: split at Updates into segments.
    reduce_axes = [a for a in axes if a.name in reduce_axis_names]
    if len(reduce_axes) != 1:
        # Malformed or multi-reduce flat — leave unchanged; caller/validator flags.
        return tuple(body)
    reduce_axis = reduce_axes[0]

    # Split body into segments at Update boundaries. AccumDecls are hoisted
    # OUT of their segment (they must live at the scope enclosing the reduce
    # Loop so Write stmts after the reduce can read the finalized accumulator
    # value). Every other stmt stays in the segment leading up to the Update.
    segments: list[list[Stmt]] = []
    current: list[Stmt] = []
    hoisted_decls: list[Stmt] = []
    for stmt in body:
        if isinstance(stmt, AccumDecl):
            hoisted_decls.append(stmt)
            continue
        current.append(stmt)
        if isinstance(stmt, Update):
            segments.append(current)
            current = []
    tail = current  # post-last-Update stmts

    nested: list[Stmt] = []
    nested.extend(hoisted_decls)
    for seg in segments:
        nested.append(Loop(axis=reduce_axis, body=tuple(seg)))
    nested.extend(tail)

    wrapped = tuple(nested)
    for a in reversed(free_axes):
        wrapped = (Loop(axis=a, body=wrapped),)
    return wrapped
