"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    axes  : tuple[Axis, ...]            — iteration space (free + reduce)
    inputs: tuple[Port, ...]            — per-input access patterns
    body  : tuple[Assign, ...]          — SSA: name = op(args)
    outputs: tuple[Port, ...]           — per-output access patterns

"Loop" here refers to the tiled loop-nest that codegen eventually emits —
one LoopOp maps to one ``GpuKernel`` and one CUDA launch.

Each input/output ``Port`` has a ``tuple[Expr, ...]`` index pattern that
describes, per-buffer-dim, how to address the external buffer from the
iteration coords (axis Vars). The former Mux/Combine input-tree variants
are gone; `Combine` semantics are handled by plain ``Assign`` statements
in the body, and `Mux` semantics (coord-predicated dispatch) will be
re-introduced as a body-level ``Select`` statement in a follow-up.

SSA invariants (unique names, defined-before-use, no forward references)
are enforced at construction time by ``LoopOp.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp

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
    """Access pattern for one external buffer (input or output).

    ``index`` is a tuple of ``Expr`` — one per dimension of the external
    buffer. Each Expr computes the offset into its buffer dim from the
    enclosing ``LoopOp.axes`` (via ``Var(axis_name)``), possibly combined
    with literals and arithmetic for transposes, broadcasts, slices.

    Identity access on a buffer whose shape matches the LoopOp's axes is
    ``Port(index=(Var(a.name) for a in axes))``. A broadcast from a
    size-1 dim is ``Literal(0, "int")`` at that position. A transpose
    swaps the axis Vars.

    Input ports are positionally bound: the i-th ``Port`` in
    ``LoopOp.inputs`` is the ``$i`` reference in SSA body args. The
    external buffer name lives at the program level in
    ``LoopLaunch.input_names[i]`` / ``LoopLaunch.output_name``.
    """

    index: tuple[Expr, ...] = ()


# ---------------------------------------------------------------------------
# SSA body
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assign:
    """One named value in the loop's SSA body: ``name = op(args)``.

    SSA invariants (enforced by ``LoopOp.__post_init__``):
      - Each ``name`` is defined exactly once across all Assigns.
      - Every ``arg`` references ``$N`` (input Port N) or a prior
        ``Assign.name``.
      - No forward references.
    """

    name: str
    op: ElementwiseOp | ReduceOp
    args: tuple[str, ...]


# ---------------------------------------------------------------------------
# LoopOp
# ---------------------------------------------------------------------------


@dataclass
class LoopOp(Op):
    """One kernel's worth of computation as an SSA program over named axes.

    Reads external buffers via ``inputs`` Ports, computes through a flat
    sequence of named ``Assign`` statements, and writes the result via
    ``outputs`` Ports.
    """

    axes: tuple[Axis, ...] = ()
    inputs: tuple[Port, ...] = ()
    body: tuple[Assign, ...] = ()
    outputs: tuple[Port, ...] = ()

    def __post_init__(self) -> None:
        _validate(self)

    def free_axes(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.kind == "free")

    def reduce_axes(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.kind == "reduce")

    def infer_output_shape(self, input_shapes: list[tuple] | dict[str, tuple] | None = None) -> tuple:
        """Output shape = extents of free axes in declaration order.

        If there are no free axes (scalar output, unusual), returns ().
        """
        return tuple(a.extent for a in self.free_axes())

    def infer_shapes(self, input_shapes: dict[str, tuple] | None = None) -> dict[str, tuple]:
        """Derive per-SSA-name shape by walking the body.

        Ports read through their ``index`` pattern; identity-like Ports
        yield the iteration shape (free + reduce axes in declaration
        order). Non-identity patterns collapse / transpose accordingly.
        ElementwiseOps broadcast; ReduceOps (keepdim) keep an axis-1 at
        the reduced position.
        """
        iter_shape = tuple(a.extent for a in self.axes)
        axis_name_to_pos = {a.name: i for i, a in enumerate(self.axes)}
        shapes: dict[str, tuple] = {}

        for i, port in enumerate(self.inputs):
            shape = _port_effective_shape(port, iter_shape, axis_name_to_pos)
            shapes[f"${i}"] = shape

        for assign in self.body:
            arg_shapes = [shapes[a] for a in assign.args if a in shapes]
            if not arg_shapes:
                shapes[assign.name] = ()
                continue
            try:
                shapes[assign.name] = assign.op.infer_output_shape(arg_shapes)
            except (ValueError, TypeError):
                shapes[assign.name] = max(arg_shapes, key=len)

        return shapes


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness + SSA invariants at construction time."""
    # Axis uniqueness.
    seen_axes: set[str] = set()
    for a in loop.axes:
        if a.name in seen_axes:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        seen_axes.add(a.name)

    # SSA: unique names, defined-before-use, no forward refs.
    defined: set[str] = {f"${i}" for i in range(len(loop.inputs))}
    for assign in loop.body:
        for arg in assign.args:
            if arg not in defined:
                raise ValueError(f"Assign {assign.name!r}: arg {arg!r} not defined")
        if assign.name in defined:
            raise ValueError(f"Assign {assign.name!r}: name already defined")
        defined.add(assign.name)


def _port_effective_shape(
    port: Port,
    iter_shape: tuple[int, ...],
    axis_name_to_pos: dict[str, int],
) -> tuple[int, ...]:
    """Return the effective shape the body sees through this port.

    Rank equals ``len(port.index)``. For each index Expr: if it's a Var
    naming an axis, that axis's extent; if it's a Literal, 1 (scalar); a
    nontrivial Expr is conservatively reported as 1 (the body sees a
    scalar per iteration point for compound indexings).
    """
    from deplodock.compiler.ir.expr import Literal, Var

    out: list[int] = []
    for e in port.index:
        if isinstance(e, Var) and e.name in axis_name_to_pos:
            out.append(iter_shape[axis_name_to_pos[e.name]])
        elif isinstance(e, Literal):
            out.append(1)
        else:
            out.append(1)
    return tuple(out)
