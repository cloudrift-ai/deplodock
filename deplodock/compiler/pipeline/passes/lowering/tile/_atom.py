"""Per-kernel atom *eligibility* — the planner-side gate for each matmul atom.

The :class:`~deplodock.compiler.ir.tile.ir.Atom` spec (cell shape, per-operand
dtypes, group size) + the ``ATOM_REGISTRY`` live in ``ir/tile/ir.py`` (next to
the other tile-IR types, and carried directly on the ``Mma`` op). They are
re-exported here so existing ``from ...tile._atom import ATOM_REGISTRY`` call
sites keep working.

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

from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.loop import LoopOp

# Back-compat re-exports (canonical definitions live in ``ir/tile/ir.py``).
__all__ = [
    "ATOM_REGISTRY",
    "Atom",
    "is_atom_eligible",
]


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
        from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, StridedLoop, Write  # noqa: PLC0415

        if ctx.compute_capability < min_cc:
            return False
        matmul_reduces = [lp for lp in loop_op.body.iter_of_type(Loop, StridedLoop) if lp.is_reduce and is_matmul_reduce(lp)]
        if not matmul_reduces:
            return False
        # Buffers produced *inside* this fused kernel (e.g. an attention output
        # feeding the o_proj matmul). ``ldmatrix`` is smem→register only and
        # ``020_stage_inputs`` stages external gmem inputs, not register-resident
        # intermediates — so a matmul whose operand is produced here can't have
        # that operand staged, and the mma path would crash in ``kernel/005``.
        # Gate it off so the scalar register-tile path picks it up.
        produced = {w.output for w in loop_op.body.iter_of_type(Write)}
        accum_names: set[str] = set()
        for k_loop in matmul_reduces:
            K_name = k_loop.axis.name
            if k_loop.axis.extent.is_static and k_loop.axis.extent.as_static() % cell_k != 0:
                return False
            for load in k_loop.body.iter_of_type(Load):
                k_dims = [d for d, e in enumerate(load.index) if K_name in e.free_vars()]
                if not k_dims:
                    continue
                # The mma path needs both operands staged for ``ldmatrix``.
                # ``020_stage_inputs._classify`` only stages a load whose cache
                # var (the K axis here) lands in a *single* index dim; a
                # collapsed-reshape operand (e.g. an attention output reaching
                # the o_proj via ``[(a/128)%16, …, a%128]`` — K split across two
                # dims) is rejected there, leaving it gmem-direct. Mirror that
                # rejection so the atom isn't offered for an operand the stager
                # can't serve; same for a fused intermediate produced in-kernel.
                if len(k_dims) > 1 or load.input in produced:
                    return False
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
            accum_names.add(accum.name)
        # Post-reduce epilogue: ``kernel/005_lower_atom_tile`` stores the mma
        # accumulator *fragment* straight to the output, so scalar epilogue
        # Assigns have no accumulator SSA name to read (each lane holds 4
        # elements of the C tile, not one ``acc0``). ONE shape is foldable
        # into the fragment store — the ``matmul_add`` residual
        # ``v = add(acc, r)`` with ``r`` loaded at exactly the Write's index:
        # every fragment element owns a known (row, col) of the output, so
        # ``RegStore`` loads the residual at the same coordinates and adds it
        # per element before the downconvert (the CUTLASS-epilogue pattern;
        # the Qwen3 down_proj+residual fusion that previously locked the
        # whole op out of the tensor-core tier). Anything else — scaling,
        # activations, multi-step chains — still gates to the scalar
        # register-tile path, which threads the scalar accumulator through
        # the epilogue correctly.
        epilogue_assigns = [s for s in loop_op.body.iter_of_type(Assign) if accum_names & set(s.args)]
        if epilogue_assigns and not _is_foldable_residual_epilogue(loop_op, accum_names, epilogue_assigns):
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


def _is_foldable_residual_epilogue(loop_op: LoopOp, accum_names: set[str], epilogue_assigns: list) -> bool:
    """True iff the post-reduce epilogue is exactly the ``matmul_add`` shape
    the mma fragment store can fold: a single ``v = add(acc, r)`` where ``r``
    is a Load at exactly the (single, scalar) Write's index, and the Write
    stores ``v``. Mirrors ``_splitk_residual.is_linear_in_accum``'s linearity
    but is deliberately narrower — one add, one residual operand, identical
    index space — because ``RegStore`` reads the residual at the fragment
    element's own output coordinates (a differently-indexed operand, e.g. a
    bias broadcast by column, would need per-operand index substitution the
    render doesn't do yet)."""
    from deplodock.compiler.ir.stmt import Load, Write  # noqa: PLC0415

    if len(epilogue_assigns) != 1 or len(accum_names) != 1:
        return False
    assign = epilogue_assigns[0]
    if assign.op.name != "add" or len(assign.args) != 2:
        return False
    acc = next(iter(accum_names))
    others = [arg for arg in assign.args if arg != acc]
    if len(others) != 1:
        return False
    writes = list(loop_op.body.iter_of_type(Write))
    if len(writes) != 1 or writes[0].value != assign.name or writes[0].is_vector:
        return False
    residual_loads = [ld for ld in loop_op.body.iter_of_type(Load) if ld.names and ld.names[0] == others[0]]
    if len(residual_loads) != 1:
        return False
    return tuple(residual_loads[0].index) == tuple(writes[0].index)


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
