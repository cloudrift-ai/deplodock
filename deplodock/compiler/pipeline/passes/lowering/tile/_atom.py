"""Per-kernel atom *eligibility* — the planner-side gate for each matmul atom.

The atom **specs** (shape, per-operand dtypes, group size) + the
``ATOM_REGISTRY`` + the ``atom_spec`` / ``atom_shape`` / ``atom_group_size``
lookups live in :mod:`deplodock.compiler.dtype` — the type module is the single
source of truth for "what does kind X mean". They are re-exported here so
existing ``from ...tile._atom import atom_spec`` call sites keep working.

This module owns only the part that *can't* live in ``dtype.py``: the per-kind
**eligibility** predicate (does a given ``LoopOp`` admit this atom on this
device?). It depends on the loop body / graph / context and on
``is_matmul_reduce`` — pipeline-layer concerns — so it stays in the planner.
:func:`is_atom_eligible` dispatches via the ``_ELIGIBILITY`` map; adding a kind
(future: wgmma, NVFP4) registers a spec in ``dtype.py`` and a predicate here.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from deplodock.compiler.dtype import (
    ATOM_KINDS,
    ATOM_REGISTRY,
    BF16,
    F16,
    AtomSpec,
    DataType,
    atom_group_size,
    atom_shape,
    atom_spec,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.loop import LoopOp

# Back-compat re-exports (canonical definitions live in ``dtype``).
__all__ = [
    "ATOM_KINDS",
    "ATOM_REGISTRY",
    "AtomSpec",
    "_ATOM_KINDS_V1",
    "atom_group_size",
    "atom_shape",
    "atom_spec",
    "is_atom_eligible",
]

# Legacy alias — some call sites / tests import the pre-move name.
_ATOM_KINDS_V1 = ATOM_KINDS


def _mma_eligible_factory(
    *,
    cell_shape: tuple[int, int, int],
    operand_dtype: DataType,
    min_cc: tuple[int, int],
) -> Callable[[LoopOp, Context, Graph], bool]:
    """Build an mma.sync eligibility predicate for a given cell shape + operand
    dtype + min cc gate. Parametrising the predicate lets the f16 and bf16
    kinds reuse the same shape gate without copying the body.

    The predicate checks:

    - At least one matmul-reduce in the body (``is_matmul_reduce``).
    - Every K-indexed Load resolves to ``operand_dtype`` via
      ``graph.nodes[buf].output.dtype``.
    - ``ctx.compute_capability >= min_cc``.
    - K extent divisible by ``cell_shape[2]``.
    - Each output extent divisible by the corresponding cell dim
      (M by ``cell_shape[0]``, N by ``cell_shape[1]``).
    """
    cell_m, cell_n, cell_k = cell_shape

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


# Per-kind eligibility predicates, keyed like ``ATOM_REGISTRY`` (cell shape +
# operand dtype + min cc come from each kind's spec). Kept parallel to the
# registry rather than on ``AtomSpec`` so the spec stays a pure data record in
# ``dtype.py`` with no loop/graph/context dependency.
_ELIGIBILITY: dict[str, Callable[[LoopOp, Context, Graph], bool]] = {
    "mma_m16n8k16_f16": _mma_eligible_factory(cell_shape=(16, 8, 16), operand_dtype=F16, min_cc=(8, 0)),
    "mma_m16n8k16_bf16": _mma_eligible_factory(cell_shape=(16, 8, 16), operand_dtype=BF16, min_cc=(8, 0)),
}


def is_atom_eligible(kind: str, loop_op: LoopOp, ctx: Context, *, graph: Graph) -> bool:
    """Dispatch the per-kind eligibility predicate. Raises ``KeyError`` for an
    unregistered kind. ``graph`` is the :class:`Graph` the ``loop_op`` lives in
    — the predicate uses it to resolve Load source-buffer dtypes (Loop-IR Loads
    don't carry ``.dtype`` until the Kernel-IR ``030_stamp_types`` pass)."""
    return _ELIGIBILITY[kind](loop_op, ctx, graph)
