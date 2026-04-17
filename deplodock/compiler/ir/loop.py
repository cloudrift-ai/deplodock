"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    axes  : tuple[Axis, ...]            — iteration space (free + reduce)
    inputs: tuple[Port, ...]            — per-input access patterns
    locals: tuple[LocalBuffer, ...]     — thread-local state (accumulators,
                                          scratch); v1: scalar + scope=thread
    body  : tuple[Stmt, ...]            — SSA: Assign | Update | Write | Select

Reductions are modeled as explicit accumulator state: each reduction
contributes a ``LocalBuffer`` (with a ``combine`` op and ``init`` value)
plus one or more ``Update`` statements that fold values in. The former
``LoopOp.outputs`` tuple is gone — writes are inline ``Write`` statements
in the body, carrying their own output index and value.

By convention, the last ``Update`` in ``body`` marks the end of the
reduce sweep: everything before is per-reduce-iteration, everything
after is post-reduce (runs once per free-axis point).

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


Stmt = Assign | Update | Write | Select


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


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def pretty_print_loop(loop: LoopOp, indent: str = "") -> str:
    """Format a ``LoopOp`` as a compact, human-readable text block."""
    lines: list[str] = []

    axis_parts = [f"{a.kind}({a.name}:{a.extent})" for a in loop.axes]
    lines.append(f"{indent}axes:   {', '.join(axis_parts) if axis_parts else '(none)'}")

    if loop.inputs:
        port_parts = [f"${i}[{', '.join(render_expr(e) for e in p.index)}]" for i, p in enumerate(loop.inputs)]
        lines.append(f"{indent}inputs: {', '.join(port_parts)}")

    if loop.locals:
        loc_parts: list[str] = []
        for lb in loop.locals:
            init = render_expr(lb.init) if lb.init is not None else "?"
            if lb.combine is not None:
                loc_parts.append(f"{lb.name}:{lb.dtype}={init} combine={lb.combine.fn}")
            else:
                loc_parts.append(f"{lb.name}:{lb.dtype} (scratch)")
        lines.append(f"{indent}locals: {', '.join(loc_parts)}")

    lines.append(f"{indent}body:")
    for stmt in loop.body:
        if isinstance(stmt, Assign):
            args = ", ".join(stmt.args)
            lines.append(f"{indent}  {stmt.name} = {stmt.op.fn}({args})")
        elif isinstance(stmt, Update):
            lines.append(f"{indent}  {stmt.target} <- combine({stmt.target}, {stmt.value})")
        elif isinstance(stmt, Write):
            idx = ", ".join(render_expr(e) for e in stmt.index)
            lines.append(f"{indent}  out{stmt.output}[{idx}] = {stmt.value}")
        elif isinstance(stmt, Select):
            for bi, br in enumerate(stmt.branches):
                prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
                lines.append(f"{indent}  {prefix} {br.value} when ({render_expr(br.select)})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness, SSA invariants, accumulator pairing, v1 pins."""
    # Axis uniqueness.
    seen_axes: set[str] = set()
    for a in loop.axes:
        if a.name in seen_axes:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        seen_axes.add(a.name)

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

    # SSA scope.
    defined: set[str] = {f"${i}" for i in range(len(loop.inputs))}
    defined |= local_names

    reduce_axes = {a.name for a in loop.axes if a.kind == "reduce"}
    update_targets: set[str] = set()
    output_indices: set[int] = set()

    for stmt in loop.body:
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

    # Accumulator liveness: every acc must be updated at least once.
    for lb in loop.locals:
        if lb.combine is not None and lb.name not in update_targets:
            raise ValueError(f"LocalBuffer {lb.name!r}: combine set but never Updated")

    # Reduce / accumulator pairing.
    if reduce_axes and not accumulators:
        raise ValueError("LoopOp has reduce axes but no accumulator LocalBuffer")
    if accumulators and not reduce_axes:
        raise ValueError("LoopOp has accumulator LocalBuffer but no reduce axis")

    # Output indices must form a dense [0, N) range.
    if output_indices:
        expected = set(range(max(output_indices) + 1))
        if output_indices != expected:
            raise ValueError(f"Write.output indices {sorted(output_indices)} do not form a dense [0, N) range")
