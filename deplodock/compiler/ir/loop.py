"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion (``rules/fusion/assemble_kernels``), each ``LoopOp`` describes
the compute for one GPU kernel as an SSA program over iteration space:

    inputs (Port | Mux | Combine) →
      body (tuple[Assign, ...])  — SSA: name = op(args) →
      outputs (Port | Mux)

"Loop" here refers to the tiled loop-nest that codegen eventually emits —
one LoopOp maps to one ``GpuKernel`` (``ir/gpu.py``) and one CUDA launch.
Analogies for readers:

- **Dataflow / signal-flow graph** — leaves are buffer-backed sources,
  internal nodes transform values, edges carry per-coord values. This is
  the framing for the ``LoopInput`` tree as a whole.
- **Hardware multiplexer** (FPGA N-to-1 mux / 1-to-N demux) — for
  coord-predicated selection, inputs and outputs both.
- **Operad / expression tree** — N-ary ops composed with operadic
  identity collapse, for ``Combine``.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store;
  MLIR ``linalg`` structured ops) — for ``LoopOp`` as a whole.

Every elementwise chain is a ``tuple[ElementwiseOp, ...]`` (alias
``ElementwiseChain``); every reduction slot is a ``ReduceOp``. SSA
invariants (unique names, defined-before-use, no forward references) are
enforced at construction time by ``LoopOp.__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp

if TYPE_CHECKING:
    from deplodock.compiler.ir.expr import Expr


# ---------------------------------------------------------------------------
# LoopInput tree: Port | Mux | Combine
# ---------------------------------------------------------------------------
#
# One LoopOp = one GPU kernel. Its shape is a tiled dataflow pipeline:
#
#     inputs (LoopInput tree) ──► [contraction] ──► [reduce_stages] ──►
#                                      [epilogue] ──► outputs (Port | Mux)
#
# LoopInput is a recursive tagged union (``Port | Mux | Combine``):
#   - Port    : signal-flow leaf; one external buffer read + optional indexmap.
#   - Mux     : hardware-mux; coord-predicated dispatch among branches.
#   - Combine : operadic composition; elementwise-chain over N sub-inputs.
#
# LoopOutput is a narrower union (``Port | Mux``): outputs don't
# assemble values, they just dispatch writes (Mux on outputs = scatter).


@dataclass
class Port:
    """Signal-flow leaf: one external buffer read/write with optional layout.

    Position in ``LoopOp.inputs`` is the implicit index. The mapping
    from Port index to external buffer name lives in the graph node's
    ``inputs`` list (or equivalently the ``LoopLaunch.input_names`` at
    program level), not inside the kernel.

    ``indexmap`` (when set) describes the per-output coord access pattern
    (transpose, slice, broadcast). ``None`` = identity load/store.
    """

    indexmap: IndexMapOp | None = None


@dataclass
class MuxBranch:
    """One branch of a Mux: an input tree + a coord-predicate selector."""

    input: LoopInput
    select: Expr


@dataclass
class Mux:
    """Hardware multiplexer: coord-predicated dispatch among branches.

    On inputs: at each output coord, exactly one branch's ``select`` is
    True and its ``input`` supplies the value. Branches must be disjoint;
    invariants expect them to be exhaustive or to carry a catch-all in
    the last position (compiler-side convention, not structurally encoded).

    On outputs: same shape, inverted semantics — each branch describes
    where to write when its predicate is True. Unmatched coords produce
    no write (masked scatter).
    """

    branches: tuple[MuxBranch, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Mux.branches must be non-empty")


@dataclass
class Combine:
    """Operadic composition: N sub-inputs combined by an elementwise chain.

    ``sources`` are the operadic inputs (each another ``LoopInput``);
    ``ops`` is an elementwise chain applied to produce one value per
    output coord. Nesting is operadic composition — Combines compose into
    Combines.

    Canonicalization: a no-op wrapper (single source, empty ops) is
    illegal; the tree should already have been collapsed to the source.
    """

    sources: tuple[LoopInput, ...]
    ops: tuple[ElementwiseOp, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("Combine.sources must be non-empty")
        if len(self.sources) == 1 and not self.ops:
            raise ValueError("Combine(sources=(x,), ops=()) is a no-op wrapper; use the inner input directly")
        _assert_elementwise_chain(self.ops, "Combine.ops")


# A loop input slot is a signal-flow tree; the leaves read external
# buffers (``Port``), internal nodes transform values (``Combine``) or
# dispatch between sources (``Mux``).
type LoopInput = Port | Mux | Combine

# A loop output slot is simpler: either a plain write target (``Port``)
# or a scatter/masked writeout (``Mux``). Post-body elementwise work lives
# in ``LoopOp.epilogue``.
type LoopOutput = Port | Mux


# ---------------------------------------------------------------------------
# SSA body: Assign and LoopOp
# ---------------------------------------------------------------------------


@dataclass
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


@dataclass
class LoopOp(Op):
    """One kernel's worth of computation as an SSA program.

    The loop reads external buffers via ``inputs`` (Port | Mux | Combine),
    computes through a flat sequence of named ``Assign`` statements, and
    writes the result via ``outputs`` (Port | Mux).

    Every loop reads as a program::

        mul = mul(a, b)
        dot = reduce_sum(mul)
        out = add(dot, bias)

    The codegen walks the body sequentially, maintaining a ``values`` dict
    mapping Assign names to C expressions. Contraction (matmul K-loop) is
    detected by pattern-matching the SSA graph, not by a separate field.
    """

    inputs: tuple[LoopInput, ...]
    body: tuple[Assign, ...] = ()
    outputs: tuple[LoopOutput, ...] = ()

    def __post_init__(self) -> None:
        _validate_ssa(self)

    def infer_shapes(self, input_shapes: dict[str, tuple] | None = None) -> dict[str, tuple]:
        """Derive the shape of every named value (inputs + Assigns).

        ``input_shapes`` maps ``$N`` (or external buffer names) → shape.
        When a Port carries an ``indexmap``, its effective shape is
        ``indexmap.out_shape`` regardless of the provided shape.
        """
        ext = input_shapes or {}
        shapes: dict[str, tuple] = {}
        port_idx = 0
        for inp in self.inputs:
            if isinstance(inp, Port):
                key = f"${port_idx}"
                if inp.indexmap is not None:
                    shapes[key] = tuple(inp.indexmap.out_shape)
                else:
                    shapes[key] = tuple(ext.get(key, ()))
                port_idx += 1
            elif isinstance(inp, Combine):
                for src in inp.sources:
                    if isinstance(src, Port):
                        key = f"${port_idx}"
                        if src.indexmap is not None:
                            shapes[key] = tuple(src.indexmap.out_shape)
                        else:
                            shapes[key] = tuple(ext.get(key, ()))
                        port_idx += 1
        for assign in self.body:
            if assign.name in ext:
                shapes[assign.name] = tuple(ext[assign.name])
                continue
            arg_shapes = [shapes[a] for a in assign.args if a in shapes]
            if arg_shapes:
                try:
                    shapes[assign.name] = assign.op.infer_output_shape(arg_shapes)
                except (ValueError, TypeError):
                    shapes[assign.name] = max(arg_shapes, key=len)
        return shapes

    def infer_output_shape(self, input_shapes: dict[str, tuple] | list[tuple] | None = None) -> tuple:
        """Derive the loop's output shape from the SSA body."""
        ext = input_shapes if isinstance(input_shapes, dict) else None
        shapes = self.infer_shapes(ext)
        if self.body:
            return shapes.get(self.body[-1].name, ())
        if shapes:
            return next(iter(shapes.values()))
        return ()


type ElementwiseChain = tuple[ElementwiseOp, ...]


# ---------------------------------------------------------------------------
# Runtime invariant helpers
# ---------------------------------------------------------------------------


def _assert_elementwise_chain(chain: ElementwiseChain, where: str) -> None:
    for i, op in enumerate(chain):
        if not isinstance(op, ElementwiseOp):
            raise TypeError(f"{where}[{i}] is {type(op).__name__}, expected ElementwiseOp")


def _validate_ssa(loop: LoopOp) -> None:
    """Enforce SSA invariants: unique names, defined-before-use."""
    defined: set[str] = set()
    port_idx = 0
    for inp in loop.inputs:
        if isinstance(inp, Port):
            defined.add(f"${port_idx}")
            port_idx += 1
        elif isinstance(inp, Combine):
            for src in inp.sources:
                if isinstance(src, Port):
                    defined.add(f"${port_idx}")
                    port_idx += 1
    for assign in loop.body:
        for arg in assign.args:
            if arg not in defined:
                raise ValueError(f"Assign {assign.name!r}: arg {arg!r} not defined")
        if assign.name in defined:
            raise ValueError(f"Assign {assign.name!r}: name already defined")
        defined.add(assign.name)
