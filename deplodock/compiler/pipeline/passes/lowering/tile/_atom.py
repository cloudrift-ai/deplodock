"""Atom-kind registry — hardware-instruction specs per matmul atom.

An *atom* is the hardware-atomic shape of one matmul-reduce cell. Scalar matmul
isn't represented here (it's the absence of an atom, modelled by
:class:`ScalarTileParams` in ``_enumeration``); only MMA / tensor-core families
register here. Each spec carries the cell shape ``(M, N, K)``, the per-operand
dtype map (``"a"`` / ``"b"`` / ``"c"``; future scaled kinds add ``"a_scale"`` /
``"b_scale"``), the hardware instruction family name (``"wmma"`` / future
``"wgmma"`` / ``"mma_scaled"``), the threads-per-cell group size (32 for WMMA,
128 for wgmma), and the per-kind eligibility callable used by
:func:`is_atom_eligible` to gate the kind on a specific kernel.

The MMA fragment-factorization plan (``plans/mma-fragment-factorization.md``)
threads ``ATOM_KIND`` through the planner; this module is the single source of
truth for "what does kind X mean structurally". The eligibility predicate
:func:`is_atom_eligible` dispatches via :class:`AtomSpec.eligibility` so adding
a kind (M9: bf16, skewed shapes; future: wgmma, NVFP4) only touches this file.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it: this is
a sibling helper, not a pass.

At M2 :data:`ATOM_REGISTRY` is seeded with ``wmma_m16n16k16_f16`` (square WMMA,
f16 operands + f32 accumulator, sm_70+). M9 adds bf16 + skewed shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.dtype import BF16, F16, F32, DataType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.loop import LoopOp


@dataclass(frozen=True)
class AtomSpec:
    """Hardware-instruction spec for one matmul atom kind.

    - ``shape`` is the cell shape ``(M, N, K)`` one instruction realises.
    - ``operand_dtypes`` maps each operand role (``"a"`` / ``"b"`` / ``"c"`` for
      WMMA; scaled kinds extend with ``"a_scale"`` / ``"b_scale"``) to its
      element dtype. The materializer reads this to declare each fragment.
    - ``instruction`` names the hardware instruction family (``"wmma"`` for
      sm_70+ ``wmma::mma_sync``; future ``"wgmma"`` for sm_90+, ``"mma_scaled"``
      for sm_100+ NVFP4/MXFP4). The materializer's per-cell emit branches on
      this — synchronous WMMA vs async wgmma issue/wait, etc.
    - ``group_size`` is the threads-per-cell count (32 for WMMA — one warp;
      128 for wgmma — one warp-group). Used by the warp-tier launch-geometry
      math when computing per-CTA thread count.
    - ``eligibility`` is the per-kind predicate :func:`is_atom_eligible`
      dispatches to. Takes the candidate ``LoopOp`` + ``Context`` + the graph
      it lives in (for Load-dtype lookup via
      ``graph.nodes[buf].output.dtype`` — Loop-IR Loads don't carry ``.dtype``
      until the Kernel-IR ``030_stamp_types`` pass).
    """

    shape: tuple[int, int, int]
    operand_dtypes: Mapping[str, DataType]
    instruction: str
    group_size: int
    eligibility: Callable[[LoopOp, Context, Graph], bool]


def _wmma_eligible_factory(
    *,
    cell_shape: tuple[int, int, int],
    operand_dtype: DataType,
    min_cc: tuple[int, int],
) -> Callable[[LoopOp, Context, Graph], bool]:
    """Build a WMMA eligibility predicate for a given cell shape + operand
    dtype + min cc gate. Parametrising the predicate lets the M9 entries
    (bf16, skewed F16) reuse the same shape gate without copying the body.

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
        from deplodock.compiler.ir.stmt import Load, Loop, StridedLoop  # noqa: PLC0415

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


def _wmma_m16n16k16_f16_eligible(loop_op: LoopOp, ctx: Context, graph: Graph) -> bool:
    """Eligibility predicate for the ``wmma_m16n16k16_f16`` atom kind.

    Per Design decision 4 of ``plans/mma-fragment-factorization.md``:

    - At least one reduce in the body matches the matmul signature
      (``is_matmul_reduce``).
    - Every K-indexed Load resolves to a ``F16`` source buffer via
      ``graph.nodes[buf].output.dtype`` (Loop-IR Loads don't carry
      ``.dtype`` yet — pre-``030_stamp_types``).
    - ``ctx.compute_capability >= (7, 0)`` (WMMA F16 first shipped on Volta).
    - Each output M, N extent divisible by 16; K extent divisible by 16.
      The ``% (16 · BR)`` divisibility check is deferred until BR is picked
      in the enumerator (today's plan forces BR=1 for MMA, so this is just
      ``% 16``).

    The Accum target dtype isn't checked here — at planner time
    ``Accum.dtype`` is ``None`` (frozen by the Kernel-IR ``020_place_inits``
    pass). WMMA's C fragment is F32 by convention for the ``_f16`` operand
    kind; the existing F16-matmul path also accumulates in F32 (see
    ``kernel/020_place_inits``), so the typical kernel is already in the
    right shape. A future kind requiring F16 accumulation would need a
    distinct entry.
    """
    from deplodock.compiler.ir.stmt import Load, Loop, StridedLoop  # noqa: PLC0415

    if ctx.compute_capability < (7, 0):
        return False

    # Find at least one matmul-reduce Loop in the LoopOp body.
    matmul_reduces = [lp for lp in loop_op.body.iter_of_type(Loop, StridedLoop) if lp.is_reduce and is_matmul_reduce(lp)]
    if not matmul_reduces:
        return False

    # Every K-indexed Load in the matmul reduce body must be F16. Walk
    # ``iter_of_type`` so nested Loads (e.g. inside a prologue subtree) are
    # not missed.
    for k_loop in matmul_reduces:
        K_name = k_loop.axis.name
        if k_loop.axis.extent.is_static and k_loop.axis.extent.as_static() % 16 != 0:
            return False
        for load in k_loop.body.iter_of_type(Load):
            if K_name not in {v for e in load.index for v in e.free_vars()}:
                continue
            node = graph.nodes.get(load.input)
            if node is None or node.output.dtype != F16:
                return False

    # Each output M / N axis extent (the outer free-Loop chain leading to
    # the matmul reduce) must be divisible by 16.
    for lp in loop_op.body.iter_of_type(Loop, StridedLoop):
        if lp.is_reduce or not lp.axis.extent.is_static:
            continue
        # Only gate on axes that actually flow into matmul-reduce K-indexed
        # Loads — the outer free chain is what the planner partitions into
        # output M/N. A non-static or 1-extent axis can't be MMA-tiled.
        ext = lp.axis.extent.as_static()
        if ext > 1 and ext % 16 != 0:
            return False

    return True


# Seeded at M2 with the WMMA square F16 cell (sm_70+ Volta+). M9 extends
# with bf16 (Ampere+) + skewed shapes (m8n32k16, m32n8k16) for skinny
# attention projections.
ATOM_REGISTRY: dict[str, AtomSpec] = {
    "wmma_m16n16k16_f16": AtomSpec(
        shape=(16, 16, 16),
        operand_dtypes={"a": F16, "b": F16, "c": F32},
        instruction="wmma",
        group_size=32,
        eligibility=_wmma_m16n16k16_f16_eligible,
    ),
    "wmma_m16n16k16_bf16": AtomSpec(
        shape=(16, 16, 16),
        operand_dtypes={"a": BF16, "b": BF16, "c": F32},
        instruction="wmma",
        group_size=32,
        eligibility=_wmma_eligible_factory(cell_shape=(16, 16, 16), operand_dtype=BF16, min_cc=(8, 0)),
    ),
    "wmma_m8n32k16_f16": AtomSpec(
        shape=(8, 32, 16),
        operand_dtypes={"a": F16, "b": F16, "c": F32},
        instruction="wmma",
        group_size=32,
        eligibility=_wmma_eligible_factory(cell_shape=(8, 32, 16), operand_dtype=F16, min_cc=(7, 0)),
    ),
    "wmma_m32n8k16_f16": AtomSpec(
        shape=(32, 8, 16),
        operand_dtypes={"a": F16, "b": F16, "c": F32},
        instruction="wmma",
        group_size=32,
        eligibility=_wmma_eligible_factory(cell_shape=(32, 8, 16), operand_dtype=F16, min_cc=(7, 0)),
    ),
}


# Module-level priority-ordered tuple of MMA kinds the planner enumerates.
# Square F16 first (broadest arch coverage); then bf16 square (Ampere+);
# then skewed F16 shapes (m8n32k16 for tall-N attention projections,
# m32n8k16 for tall-M skinny outputs). Scalar is not in this list —
# scalar is the absence of an atom (modelled by :class:`ScalarTileParams`).
_ATOM_KINDS_V1: tuple[str, ...] = (
    "wmma_m16n16k16_f16",
    "wmma_m16n16k16_bf16",
    "wmma_m8n32k16_f16",
    "wmma_m32n8k16_f16",
)


def atom_spec(kind: str) -> AtomSpec:
    """Resolve ``kind`` to its :class:`AtomSpec`. Raises ``KeyError`` for an
    unregistered kind — there's no "scalar" entry (scalar is the absence of an
    atom, see module docstring)."""
    return ATOM_REGISTRY[kind]


def atom_shape(kind: str) -> tuple[int, int, int]:
    """Cell shape ``(M, N, K)`` of ``kind``."""
    return ATOM_REGISTRY[kind].shape


def atom_group_size(kind: str) -> int:
    """Threads-per-cell of ``kind`` (32 for WMMA, 128 for wgmma)."""
    return ATOM_REGISTRY[kind].group_size


def is_atom_eligible(kind: str, loop_op: LoopOp, ctx: Context, *, graph: Graph) -> bool:
    """Dispatch the per-kind eligibility predicate. Raises ``KeyError`` for
    an unregistered kind. ``graph`` is the :class:`Graph` the ``loop_op``
    lives in — the predicate uses it to resolve Load source-buffer dtypes
    (Loop-IR Loads don't carry ``.dtype`` until the Kernel-IR
    ``030_stamp_types`` pass)."""
    spec = atom_spec(kind)
    return spec.eligibility(loop_op, ctx, graph)
