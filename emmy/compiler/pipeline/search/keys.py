"""Op-key derivation + source-chain walking for the search package.

Shared by :mod:`.db`, :mod:`.policy`, and the bench-terminal helper
in :mod:`emmy.compiler.pipeline.pipeline`.

``op_cache_key`` keys any kernel-bearing op:

- ``CudaOp`` — digest of rendered kernel source + launch params (the
  bits that determine runtime behavior).
- ``LoopOp`` / ``TileOp`` / ``KernelOp`` — digest of the dialect tag
  plus :meth:`Body.structural_key` (canonicalizes SSA, axis,
  commutative-arg, and external-buffer names). KernelOp works because
  ``kernel/ir.py`` registers ``rewrite`` handlers for every Kernel-IR
  stmt (Smem, Sync, CpAsync*, Tma*, Mbarrier*, TreeHalve, WarpShuffle),
  letting ``normalize_body`` walk the body without bailing.

Same kernel reached via different rewrite paths produces the same key
— ``Op.source`` is not part of the digest.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from emmy.compiler.structural import digest

Dialect = Literal["loop", "tile", "kernel", "cuda"]

# The native ``PLACE@cone`` placement records the demoted-cone keep-vs-cut **structural**
# (kernel-set-changing) decision — ``inline`` keep / ``cut`` materialize-to-gmem
# (``enumeration/_families``). Its canonical ``cone`` element is distinct from any
# operand-staging ``PLACE@<buffer>`` or score ``PLACE@<edge>``, so a hop that introduces it
# is unambiguously the structural fork (the `cut` value also self-describes the
# materialization vs an operand `gmem`-direct read). A source-chain hop that introduces it
# is a decomposition hop whose cost is a Σ owned by the two-level tuner, NOT a ``lowering``
# row. The literal is kept here (search/ shouldn't import a lowering pass) — it mirrors
# ``_families.cone_key()``.
_CONE_PLACE = "PLACE@cone"


def structural_decision_delta(knobs: dict) -> dict:
    """The structural-decision knobs ``knobs`` carries — today the single ``PLACE@cone``
    placement (keep ``inline`` / cut ``cut``); ``{}`` when the op carries no cone decision.
    Read by the candidate replay (``search/candidate``) and the decomposition-row featurizer
    (``two_level._decomposition_rows``) to attribute each side's Σ cost."""
    return {_CONE_PLACE: knobs[_CONE_PLACE]} if _CONE_PLACE in knobs else {}


def introduces_structural_decision(parent_op: object, child_op: object) -> bool:
    """True when ``child_op`` carries the ``PLACE@cone`` structural decision the
    ``parent_op`` lacks — the keep-vs-cut fork hop. Covers both the cut (loop→loop fragment
    splice) and the keep (loop→tile ``seed_fused`` jump), so the recorder skips the decision
    hop regardless of dialect crossing."""
    p = getattr(parent_op, "knobs", None) or {}
    c = getattr(child_op, "knobs", None) or {}
    return _CONE_PLACE in c and _CONE_PLACE not in p


def op_cache_key(op: object) -> str | None:
    """Cache key for any kernel-bearing op, or ``None`` if not cacheable."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileGraphOp, TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        # Name-invariant: the kernel function name is rendered into the source
        # (``void <name>(...)``) but doesn't change runtime behavior. Normalize
        # it out so renaming a kernel (e.g. via op provenance) neither busts the
        # perf cache nor blocks an isolated-kernel tune from transferring to a
        # whole-model compile.
        src = op.kernel_source.replace(op.kernel_name, "_K_") if op.kernel_name else op.kernel_source
        return digest("CudaOp", src, op.arg_order, op.grid, op.block, op.smem_bytes)
    if isinstance(op, (LoopOp, TileOp, KernelOp)):
        # Knobs are part of the key: same-body / different-knobs variants
        # (e.g. ``020_stage_inputs`` emits a no-op TileOp with a
        # ``STAGE="0..0"`` knob to mark "considered, declined" decisions)
        # must not collide with their parent in the search tree, or
        # ``SearchTree.expand`` self-parents the node and
        # ``_propagate_expected`` walks the parent chain forever.
        knob_key = tuple(sorted(op.knobs.items())) if op.knobs else ()
        return digest(type(op).__name__, op.body.structural_key(), knob_key)
    if isinstance(op, TileGraphOp):
        # The enumeration-pass output (a chosen Schedule's TileGraph, pre-assembly):
        # key on the TileGraph's canonical algorithm + Schedule identity + knobs.
        knob_key = tuple(sorted(op.knobs.items())) if op.knobs else ()
        return digest("TileGraphOp", op.structural_key(), knob_key)
    return None


def dialect_of(op: object) -> Dialect | None:
    """Return the dialect tag for any kernel-bearing op, or ``None``."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        return "cuda"
    if isinstance(op, KernelOp):
        return "kernel"
    if isinstance(op, TileOp):
        return "tile"
    if isinstance(op, LoopOp):
        return "loop"
    from emmy.compiler.ir.tile.ir import TileGraphOp  # noqa: PLC0415

    if isinstance(op, TileGraphOp):
        return "tile"  # the enumeration output, still the tile dialect (pre-assembly)
    return None


def _is_kernel_bearing(op: object) -> bool:
    """True for any op that represents one kernel of work in the pipeline
    (lowering states from ``LoopOp`` through ``CudaOp``)."""
    return dialect_of(op) is not None


def source_chain(op: object) -> Iterator[object]:
    """Yield ``op`` and every predecessor along ``Op.source``."""
    cur = op
    while cur is not None:
        yield cur
        cur = getattr(cur, "source", None)
