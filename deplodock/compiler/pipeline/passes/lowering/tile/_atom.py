"""Per-kernel atom *eligibility* — the planner-side gate for each matmul atom.

The :class:`~deplodock.compiler.ir.tile.ir.Atom` spec (cell shape, per-operand
dtypes, group size) + the ``ATOM_REGISTRY`` + the ``atom_spec`` / ``atom_shape``
/ ``atom_group_size`` lookups live in ``ir/tile/ir.py`` (next to the other
tile-IR types, and carried directly on the ``Mma`` op). They are re-exported
here so existing ``from ...tile._atom import atom_spec`` call sites keep working.

This module owns only the part that *can't* live in the IR layer: the per-kind
**eligibility** predicate (does a given ``LoopOp`` admit this atom on this
device?). It depends on the loop body / graph / context and on
``is_matmul_reduce`` — pipeline-layer concerns — so it stays in the planner.
:func:`is_atom_eligible` dispatches via the ``_ELIGIBILITY`` map; adding a kind
(future: wgmma, NVFP4) registers an ``Atom`` in ``ir/tile/ir.py`` and a
predicate here.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from deplodock.compiler.ir.tile.ir import (
    ATOM_KINDS,
    ATOM_REGISTRY,
    Atom,
    atom_group_size,
    atom_shape,
    atom_spec,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.loop import LoopOp

# Back-compat re-exports (canonical definitions live in ``ir/tile/ir.py``).
__all__ = [
    "ATOM_KINDS",
    "ATOM_REGISTRY",
    "Atom",
    "_ATOM_KINDS_V1",
    "atom_group_size",
    "atom_shape",
    "atom_spec",
    "is_atom_eligible",
]

# Legacy alias — some call sites / tests import the pre-move name.
_ATOM_KINDS_V1 = ATOM_KINDS


def _mma_eligible_factory(atom: Atom, *, min_cc: tuple[int, int] = (8, 0)) -> Callable[[LoopOp, Context, Graph], bool]:
    """Build an mma.sync eligibility predicate for ``atom`` — its cell shape +
    A-operand dtype come straight off the spec; ``min_cc`` is the device gate.

    The predicate checks:

    - At least one matmul-reduce in the body (``is_matmul_reduce``).
    - Every K-indexed Load resolves to the atom's operand dtype via
      ``graph.nodes[buf].output.dtype``.
    - ``ctx.compute_capability >= min_cc``.
    - K extent divisible by the cell's K dim.
    - Each output extent divisible by the corresponding cell dim (M / N).
    """
    cell_m, cell_n, cell_k = atom.shape
    operand_dtype = atom.operand_dtype("a")

    def predicate(loop_op: LoopOp, ctx: Context, graph: Graph) -> bool:
        from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, StridedLoop  # noqa: PLC0415

        if ctx.compute_capability < min_cc:
            return False
        matmul_reduces = [lp for lp in loop_op.body.iter_of_type(Loop, StridedLoop) if lp.is_reduce and is_matmul_reduce(lp)]
        if not matmul_reduces:
            return False
        for k_loop in matmul_reduces:
            K_name = k_loop.axis.name
            if k_loop.axis.extent.is_static and k_loop.axis.extent.as_static() % cell_k != 0:
                return False
            for load in k_loop.body.iter_of_type(Load):
                if K_name not in {v for e in load.index for v in e.free_vars()}:
                    continue
                node = graph.nodes.get(load.input)
                if node is None or node.output.dtype != operand_dtype:
                    return False
            # Body purity: ``kernel/005_lower_atom_tile`` only handles the
            # canonical ``[Load a, Load b, Assign(multiply, a, b), Accum]``
            # shape — extra Assigns (e.g. constant-folded pre-scaling like
            # ``a * 0.1`` from torch's per-input scale) would be silently
            # dropped by the mma.sync emit. Until the MMA emit handles
            # in-cell scaling, gate those kernels off so the scalar path
            # picks them up correctly.
            top_level = list(k_loop.body)
            n_loads = sum(1 for s in top_level if isinstance(s, Load))
            n_assigns = sum(1 for s in top_level if isinstance(s, Assign))
            n_accums = sum(1 for s in top_level if isinstance(s, Accum))
            # Pure matmul cell: 2 Loads, 1 Assign (the multiply), 1 Accum.
            if not (n_loads == 2 and n_assigns == 1 and n_accums == 1):
                return False
            # Verify the Assign feeds the Accum and reads only the two Loads.
            loads = [s for s in top_level if isinstance(s, Load)]
            assigns = [s for s in top_level if isinstance(s, Assign)]
            accums = [s for s in top_level if isinstance(s, Accum)]
            multiply = assigns[0]
            accum = accums[0]
            load_names = {ld.names[0] for ld in loads if ld.names}
            if set(multiply.args) != load_names:
                return False
            if accum.value != multiply.name:
                return False
        # Each output free-axis extent must divide cleanly. The body's outer
        # free Loops contribute the M (outer) / N (inner) extents the planner
        # will partition into output cells; gate the outermost-to-inner
        # static free axes against (cell_m, cell_n) in their order of
        # appearance.
        free_extents = [
            lp.axis.extent.as_static()
            for lp in loop_op.body.iter_of_type(Loop, StridedLoop)
            if not lp.is_reduce and lp.axis.extent.is_static
        ]
        cell_dims = (cell_m, cell_n)
        for ext, cell in zip(reversed(free_extents), reversed(cell_dims), strict=False):
            if ext > 1 and ext % cell != 0:
                return False
        return True

    return predicate


# Per-atom eligibility predicates, keyed by the ``Atom`` itself (cell shape +
# operand dtype come from the spec). Kept parallel to the registry rather than
# on ``Atom`` so the spec stays a pure data record in ``ir/tile/ir.py`` with no
# loop/graph/context dependency.
_ELIGIBILITY: dict[Atom, Callable[[LoopOp, Context, Graph], bool]] = {atom: _mma_eligible_factory(atom) for atom in ATOM_REGISTRY.values()}


def is_atom_eligible(atom: Atom, loop_op: LoopOp, ctx: Context, *, graph: Graph) -> bool:
    """Dispatch the per-atom eligibility predicate. ``graph`` is the
    :class:`Graph` the ``loop_op`` lives in — the predicate uses it to resolve
    Load source-buffer dtypes (Loop-IR Loads don't carry ``.dtype`` until the
    Kernel-IR ``030_stamp_types`` pass)."""
    return _ELIGIBILITY[atom](loop_op, ctx, graph)
