"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    axes  : tuple[Axis, ...]            — iteration space (free + reduce)
    inputs: tuple[Port, ...]            — per-input access patterns
    locals: tuple[LocalBuffer, ...]     — thread-local state (accumulators,
                                          scratch); v1: scalar + scope=thread
    body  : tuple[Stmt, ...]            — SSA: Assign | Update | Write | Select | Loop

Iteration structure can be expressed two ways during the ongoing refactor:

1. **Flat body** (legacy, in-transition): body is a linear sequence of
   Assign/Update/Write/Select. The positionally-last ``Update`` marks the
   end of the reduce sweep by convention: statements before run per
   reduce-iteration, statements after run once per free-axis point.
2. **Nested body** (target): iteration is explicit via ``Loop(axis, body)``
   statements. Each ``Loop`` is one iteration dimension; ``Loop.body`` runs
   ``axis.extent`` times. Reduce-kind Loops fold ``Update`` statements into
   outer ``LocalBuffer`` accumulators. Free-kind Loops run in parallel with
   no accumulator folding. Reading top-to-bottom matches execution order.

Consumers should call ``_normalize_flat_to_nested`` at entry to upgrade
legacy flat bodies to the nested form transparently during the migration.

Reductions are modeled as explicit accumulator state: each reduction
contributes a ``LocalBuffer`` (with a ``combine`` op and ``init`` value)
plus one or more ``Update`` statements that fold values in.

SSA invariants (unique names, defined-before-use, accumulator pairing)
are enforced at construction time by ``LoopOp.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.tensor import ElementwiseOp

# ---------------------------------------------------------------------------
# Axis — named iteration variable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """One named iteration variable at the loop level.

    Referenced from ``Expr`` subtrees (inside ``Port.index`` etc.) by
    ``Var(name)``. ``kind`` distinguishes parallel free axes (part of the
    output iteration space) from reduce axes (swept sequentially inside
    the loop and collapsed via accumulators).

    ``extent`` is a static integer in v1; future revisions may allow an
    ``Expr`` for dynamic batch/seq dims.
    """

    name: str
    extent: int
    kind: Literal["free", "reduce"]


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
# LocalBuffer — thread-local scratch / accumulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalBuffer:
    """Thread-local state: accumulator (combine set) or scratch (combine=None).

    Accumulators are declared with an ``init`` value and a ``combine``
    ElementwiseOp; ``Update`` statements fold new values into them via the
    combine op. After the reduce sweep, other body statements read the
    accumulator's finalized value by referring to its ``name``.

    v1 invariants (enforced): ``shape == ()`` (scalar) and
    ``scope == "thread"``. The ``shape`` / ``scope`` fields exist for
    forward-compat with Stage-2 smem tiling.
    """

    name: str
    dtype: str = "f32"
    init: Expr | None = None
    combine: ElementwiseOp | None = None
    shape: tuple[int, ...] = ()
    scope: Literal["thread", "warp", "block"] = "thread"


# ---------------------------------------------------------------------------
# Body statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assign:
    """Pure SSA body statement: ``name = op(args)``.

    ``op`` is always an ``ElementwiseOp`` (reductions have moved to
    ``LocalBuffer`` + ``Update``). ``args`` reference ``$N`` ports,
    ``LocalBuffer.name`` (reads current / finalized acc value), or prior
    SSA names.
    """

    name: str
    op: ElementwiseOp
    args: tuple[str, ...]


@dataclass(frozen=True)
class Update:
    """Fold a value into a LocalBuffer accumulator.

    Semantics: ``acc = combine(acc, value)`` where ``combine`` is the
    LocalBuffer's combine op. ``target`` must name a LocalBuffer with
    ``combine is not None``. ``value`` references an SSA name (Assign,
    prior Update target finalized, or ``$N``).

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
    Loops fold any ``Update`` statements in their body into the outer
    ``LocalBuffer`` accumulator (one sweep over the axis per accumulator).
    Free-kind Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``LocalBuffer`` accumulators (written via ``Update``) cross the Loop
    boundary, carrying the finalized reduced value.
    """

    axis: Axis
    body: tuple[Stmt, ...]


Stmt = Assign | Update | Write | Select | Loop


# ---------------------------------------------------------------------------
# LoopOp
# ---------------------------------------------------------------------------


@dataclass
class LoopOp(Op):
    """One kernel's worth of computation as an SSA program over named axes."""

    axes: tuple[Axis, ...] = ()
    inputs: tuple[Port, ...] = ()
    locals: tuple[LocalBuffer, ...] = ()
    body: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
        _validate(self)

    def free_axes(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.kind == "free")

    def reduce_axes(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.kind == "reduce")

    def infer_output_shape(self, input_shapes: list[tuple] | dict[str, tuple] | None = None) -> tuple:
        """Output shape = extents of free axes in declaration order."""
        return tuple(a.extent for a in self.free_axes())

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
        Falls back to ``infer_output_shape`` when no ``Write`` is present.
        """
        import numpy as np

        writes = [s for s in flatten_body(self.body) if isinstance(s, Write)]
        if not writes:
            return self.infer_output_shape()
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


def pretty_print_loop(loop: LoopOp, indent: str = "") -> str:
    """Format a ``LoopOp`` as a compact, human-readable text block.

    Nested ``Loop`` blocks render as indented ``for <axis> in [kind, extent]:``
    sub-blocks; each level of nesting adds two spaces of indent.
    """
    lines: list[str] = []

    axis_parts = [f"{a.kind}({a.name}:{a.extent})" for a in loop.axes]
    lines.append(f"{indent}axes:   {', '.join(axis_parts) if axis_parts else '(none)'}")

    if loop.inputs:
        port_parts = [f"${i}[{', '.join(render_expr(e) for e in p.index)}]" for i, p in enumerate(loop.inputs)]
        lines.append(f"{indent}inputs: {', '.join(port_parts)}")

    locals_by_name: dict[str, LocalBuffer] = {lb.name: lb for lb in loop.locals}
    if loop.locals:
        loc_parts: list[str] = []
        for lb in loop.locals:
            init = render_expr(lb.init) if lb.init is not None else "?"
            if lb.combine is not None:
                loc_parts.append(f"{lb.name}:{lb.dtype}={init} via {lb.combine.fn}")
            else:
                loc_parts.append(f"{lb.name}:{lb.dtype} (scratch)")
        lines.append(f"{indent}locals: {', '.join(loc_parts)}")

    lines.append(f"{indent}body:")
    _render_body(loop.body, indent + "  ", locals_by_name, lines)

    return "\n".join(lines)


def _render_body(
    stmts: tuple[Stmt, ...],
    indent: str,
    locals_by_name: dict[str, LocalBuffer],
    lines: list[str],
) -> None:
    """Render a body tuple (recursive for nested ``Loop``)."""
    for stmt in stmts:
        if isinstance(stmt, Assign):
            args = ", ".join(stmt.args)
            lines.append(f"{indent}{stmt.name} = {stmt.op.fn}({args})")
        elif isinstance(stmt, Update):
            lb = locals_by_name.get(stmt.target)
            fn = lb.combine.fn if lb is not None and lb.combine is not None else "?"
            lines.append(f"{indent}{stmt.target} <- {fn}({stmt.target}, {stmt.value})")
        elif isinstance(stmt, Write):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}out{stmt.output}[{idx}] = {stmt.value}")
        elif isinstance(stmt, Select):
            for bi, br in enumerate(stmt.branches):
                prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                lines.append(f"{indent}{prefix} {br.value} when ({render_expr(br.select)})")
        elif isinstance(stmt, Loop):
            a = stmt.axis
            lines.append(f"{indent}for {a.name} in [{a.kind}, {a.extent}]:")
            _render_body(stmt.body, indent + "  ", locals_by_name, lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness, SSA invariants, accumulator pairing, v1 pins.

    Body validation recurses into ``Loop`` blocks. SSA names defined inside a
    ``Loop.body`` are scoped to that body — invisible outside. Only
    ``LocalBuffer`` accumulators cross Loop boundaries (via ``Update``).
    Axis names (from the stored ``LoopOp.axes`` and from any nested Loops) are
    validated as a flat set for uniqueness across the kernel.
    """
    # LoopOp.axes uniqueness (stored axes).
    all_axis_names: set[str] = set()
    for a in loop.axes:
        if a.name in all_axis_names:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        all_axis_names.add(a.name)

    # LocalBuffer uniqueness + v1 pins.
    local_names: set[str] = set()
    accumulators: dict[str, LocalBuffer] = {}
    for lb in loop.locals:
        if lb.name in local_names:
            raise ValueError(f"LocalBuffer: duplicate name {lb.name!r}")
        local_names.add(lb.name)
        if lb.shape != ():
            raise ValueError(f"LocalBuffer {lb.name!r}: v1 requires shape=() (got {lb.shape!r})")
        if lb.scope != "thread":
            raise ValueError(f"LocalBuffer {lb.name!r}: v1 requires scope='thread' (got {lb.scope!r})")
        if lb.combine is not None:
            accumulators[lb.name] = lb

    # Kernel-scope names (visible everywhere): ports + locals.
    kernel_scope: set[str] = {f"${i}" for i in range(len(loop.inputs))} | local_names

    # Collect body-level axis names (from stored tuple + nested Loop blocks).
    reduce_axes: set[str] = {a.name for a in loop.axes if a.kind == "reduce"}
    update_targets: set[str] = set()
    output_indices: set[int] = set()

    def _walk(stmts: tuple[Stmt, ...], defined: set[str], ancestor_loop_axes: tuple[str, ...]) -> None:
        """Validate a body scope. ``defined`` contains in-scope SSA names.
        ``ancestor_loop_axes`` are the names of Loops enclosing this body —
        a nested Loop axis may not shadow any of them (a Loop inside itself
        is ambiguous). Sibling Loops with the same axis name are fine — that
        is the softmax pattern (two sequential sweeps over the same axis).
        """
        for stmt in stmts:
            if isinstance(stmt, Assign):
                for arg in stmt.args:
                    if arg not in defined:
                        raise ValueError(f"Assign {stmt.name!r}: arg {arg!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Assign {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Update):
                if stmt.target not in accumulators:
                    raise ValueError(f"Update.target {stmt.target!r} does not name a LocalBuffer with combine set")
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
                output_indices.add(stmt.output)
            elif isinstance(stmt, Loop):
                if stmt.axis.name in ancestor_loop_axes:
                    raise ValueError(f"Loop axis {stmt.axis.name!r} shadows enclosing Loop axis")
                if stmt.axis.kind == "reduce":
                    reduce_axes.add(stmt.axis.name)
                # SSA names defined inside Loop.body are scoped to that body.
                _walk(stmt.body, set(defined), ancestor_loop_axes + (stmt.axis.name,))

    _walk(loop.body, set(kernel_scope), ())

    # Accumulator liveness: every acc must be updated at least once (anywhere in the tree).
    for lb in loop.locals:
        if lb.combine is not None and lb.name not in update_targets:
            raise ValueError(f"LocalBuffer {lb.name!r}: combine set but never Updated")

    # Reduce / accumulator pairing — treat stored ``axes`` and nested reduce Loops as one set.
    if reduce_axes and not accumulators:
        raise ValueError("LoopOp has reduce axes but no accumulator LocalBuffer")
    if accumulators and not reduce_axes:
        raise ValueError("LoopOp has accumulator LocalBuffer but no reduce axis")

    # Output indices must form a dense [0, N) range.
    if output_indices:
        expected = set(range(max(output_indices) + 1))
        if output_indices != expected:
            raise ValueError(f"Write.output indices {sorted(output_indices)} do not form a dense [0, N) range")


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


def has_flat_reduce(loop: LoopOp) -> bool:
    """True if ``loop`` carries reduce axes on its ``axes`` tuple but has no
    ``Loop`` block in its body — i.e. the legacy flat representation.
    """
    if any(a.kind == "reduce" for a in loop.axes):
        return not any(isinstance(s, Loop) for s in loop.body)
    # Any Updates at the top level of body (no wrapping Loop) also count as flat.
    if any(isinstance(s, Update) for s in loop.body):
        return not any(isinstance(s, Loop) for s in loop.body)
    return False


# ---------------------------------------------------------------------------
# Flat → nested normalization shim
# ---------------------------------------------------------------------------


def _normalize_flat_to_nested(loop: LoopOp) -> LoopOp:
    """Upgrade a legacy flat-body ``LoopOp`` to the nested ``Loop``-block form.

    Idempotent: if the body already contains any ``Loop``, returns ``loop``
    unchanged.

    The nested form is:

    - Outer free-axis ``Loop``s wrap everything, innermost = last free axis.
    - At the innermost free level, body contains:
      - One ``Loop(reduce_axis, body=segment)`` per segment produced by
        splitting the flat body at ``Update`` boundaries.
      - Post-reduce ``Assign`` / ``Select`` / ``Write`` statements
        (everything after the last ``Update``) sit after the reduce Loops.
    - Non-reduce flat bodies are wrapped only in free Loops; no reduce Loops.

    ``loop.axes`` is preserved as-is for back-compat (all axes still listed
    on the stored field). Phase 4 will drop reduce axes from the stored
    tuple once producers emit nested directly.
    """
    if not any(isinstance(s, (Loop, Update)) for s in loop.body):
        # No Loops and no Updates — purely a flat pointwise body; nothing to
        # segment. Wrap in free Loops for the nested form.
        return _wrap_in_free_loops(loop, body=tuple(loop.body))

    if any(isinstance(s, Loop) for s in loop.body):
        return loop

    # Flat body with at least one Update: split at Updates into segments.
    reduce_axes = [a for a in loop.axes if a.kind == "reduce"]
    if not reduce_axes:
        # No reduce axis but body contains Update — malformed. Leave as-is;
        # the validator will complain.
        return loop
    if len(reduce_axes) != 1:
        # Multiple reduce axes on the flat form — not a softmax-like case the
        # shim supports. Leave unchanged; the validator will flag it.
        return loop
    reduce_axis = reduce_axes[0]

    # Split the flat body at Update boundaries.
    segments: list[list[Stmt]] = []
    current: list[Stmt] = []
    for stmt in loop.body:
        current.append(stmt)
        if isinstance(stmt, Update):
            segments.append(current)
            current = []
    tail = current  # post-last-Update stmts

    # Each Update-terminated segment → one reduce Loop over the same axis.
    nested_body: list[Stmt] = []
    for seg in segments:
        nested_body.append(Loop(axis=reduce_axis, body=tuple(seg)))
    nested_body.extend(tail)

    return _wrap_in_free_loops(loop, body=tuple(nested_body))


def _wrap_in_free_loops(loop: LoopOp, body: tuple[Stmt, ...]) -> LoopOp:
    """Wrap ``body`` in nested ``Loop(free_axis, ...)`` blocks, outermost
    first. Returns a new ``LoopOp`` with the wrapped body.
    """
    free_axes = [a for a in loop.axes if a.kind == "free"]
    wrapped: tuple[Stmt, ...] = body
    for a in reversed(free_axes):
        wrapped = (Loop(axis=a, body=wrapped),)
    return LoopOp(axes=loop.axes, inputs=loop.inputs, locals=loop.locals, body=wrapped)
