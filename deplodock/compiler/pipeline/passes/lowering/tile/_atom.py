"""Atom-kind registry — hardware-instruction specs per matmul atom.

An *atom* is the hardware-atomic shape of one matmul-reduce cell. Scalar matmul
isn't represented here (it's the absence of an atom, modelled by
:class:`ScalarTileParams` in ``_enumeration``); only MMA / tensor-core families
register here. Each spec carries the cell shape ``(M, N, K)``, the per-operand
dtype map (``"a"`` / ``"b"`` / ``"c"``; future scaled kinds add ``"a_scale"`` /
``"b_scale"``), the hardware instruction family name (``"mma_sync"`` today;
future ``"wgmma"`` / ``"mma_scaled"``), the threads-per-cell group size (32 for
the warp-level mma.sync atom, 128 for a future wgmma warp-group), and the
per-kind eligibility callable used by :func:`is_atom_eligible` to gate the kind
on a specific kernel.

The MMA fragment-factorization plan (``plans/mma-fragment-factorization.md``)
threads ``ATOM_KIND`` through the planner; this module is the single source of
truth for "what does kind X mean structurally". The eligibility predicate
:func:`is_atom_eligible` dispatches via :class:`AtomSpec.eligibility` so adding
a kind (future: wgmma, NVFP4) only touches this file.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it: this is
a sibling helper, not a pass.

:data:`ATOM_REGISTRY` holds the s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix``
kinds — f16 and bf16 operands, f32 accumulate, Ampere+ (sm_80) — the sole
tensor-core family. The legacy ``nvcuda::wmma`` kinds were removed once the
swizzled mma.sync slab beat them (see plans/mma-sync-smem-swizzle.md).
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
    - ``operand_dtypes`` maps each operand role (``"a"`` / ``"b"`` / ``"c"``;
      scaled kinds extend with ``"a_scale"`` / ``"b_scale"``) to its element
      dtype. The materializer reads this to declare each register array.
    - ``instruction`` names the hardware instruction family (``"mma_sync"`` for
      the sm_80+ s16816 ``mma.sync.aligned`` + ``ldmatrix`` path; future
      ``"wgmma"`` for sm_90+, ``"mma_scaled"`` for sm_100+ NVFP4/MXFP4). The
      per-cell emit (``kernel/005_lower_atom_tile``) branches on this.
    - ``group_size`` is the threads-per-cell count (32 for the warp-level
      mma.sync atom; 128 for a future wgmma warp-group). Used by the warp-tier
      launch-geometry math when computing per-CTA thread count.
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


ATOM_REGISTRY: dict[str, AtomSpec] = {
    # Modern warp-level MMA: ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` (the
    # ``s16816`` cell cuBLAS/CUTLASS use) — the sole tensor-core family. f16 /
    # bf16 operands, f32 accumulate, sm_80+ (the m16n8k16 op is Ampere+).
    # ``instruction="mma_sync"``: ``kernel/005_lower_atom_tile`` emits the
    # RegFragment/LdmatrixLoad/MmaSyncPtx/RegStore chain. The path has **no
    # gmem-direct load** (ldmatrix is smem→register only) — the lowering
    # requires a staged smem source and RuleSkips an unstaged leaf, dropping
    # the warp-tier variant so the scalar tier covers that shape.
    "mma_m16n8k16_f16": AtomSpec(
        shape=(16, 8, 16),
        operand_dtypes={"a": F16, "b": F16, "c": F32},
        instruction="mma_sync",
        group_size=32,
        eligibility=_mma_eligible_factory(cell_shape=(16, 8, 16), operand_dtype=F16, min_cc=(8, 0)),
    ),
    # bf16 sibling: same s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix``
    # path (bf16 and f16 share the 16-bit fragment layout / ldmatrix.b16, so
    # only the PTX dtype field differs — ``MmaSyncPtx.ab_dtype`` selects the
    # ``dpl_mma_…_bf16`` wrapper). bf16 mma.sync is Ampere+ (sm_80). This is
    # the bf16 tensor-core path now that WMMA is gone.
    "mma_m16n8k16_bf16": AtomSpec(
        shape=(16, 8, 16),
        operand_dtypes={"a": BF16, "b": BF16, "c": F32},
        instruction="mma_sync",
        group_size=32,
        eligibility=_mma_eligible_factory(cell_shape=(16, 8, 16), operand_dtype=BF16, min_cc=(8, 0)),
    ),
}


# Module-level priority-ordered tuple of MMA atom kinds the planner enumerates.
# The s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` path is the sole
# tensor-core family (the legacy ``nvcuda::wmma`` kinds were removed — the
# swizzled mma.sync slab beats WMMA, which can't swizzle). f16 first, then
# bf16 (both Ampere+). The m16n8k16 atom tiles any divisible shape, so it
# covers what the old skewed WMMA kinds (m8n32 / m32n8) did via tiling.
# Scalar is not in this list — scalar is the absence of an atom (modelled by
# :class:`ScalarTileParams`), the fallback when no mma.sync kind is eligible
# (single-warp tiles, non-divisible extents, sm < 8.0, or no TMA staging).
_ATOM_KINDS_V1: tuple[str, ...] = (
    "mma_m16n8k16_f16",
    "mma_m16n8k16_bf16",
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
