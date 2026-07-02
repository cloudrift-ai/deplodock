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

# The ``PLACE@<element>`` placement family records a **structural** (kernel-set-changing)
# decision — ``fuse`` (registers) / ``cut`` (materialize the edge to gmem) per edge class
# (``cone`` producer-cone inlining, ``fold`` flash vs multi-kernel attention; ``tuple`` is
# dominance policy and never stamped). A source-chain hop that introduces a ``PLACE@`` key is
# a decomposition hop whose cost is a Σ owned by the two-level tuner, NOT a ``lowering`` row.
# Matched by family prefix (search/ shouldn't import the lowering pass that stamps them).
_PLACE_PREFIX = "PLACE@"


def structural_decision_delta(knobs: dict) -> dict:
    """The structural-decision knobs ``knobs`` carries — the ``PLACE@<element>`` placements
    (resolved ``fuse`` / ``cut``); ``{}`` when the op carries none. Read by the candidate
    replay (``search/candidate``) and the decomposition-row featurizer
    (``two_level._decomposition_rows``) to attribute each side's Σ cost."""
    return {k: v for k, v in knobs.items() if k.startswith(_PLACE_PREFIX)}


def introduces_structural_decision(parent_op: object, child_op: object) -> bool:
    """True when ``child_op`` carries a ``PLACE@<element>`` structural decision the
    ``parent_op`` lacks — the fuse-vs-cut fork hop. Covers both the cut (fragment splice)
    and the fuse (the fused-kernel jump), so the recorder skips the decision hop regardless
    of dialect crossing."""
    p = getattr(parent_op, "knobs", None) or {}
    c = getattr(child_op, "knobs", None) or {}
    return any(k.startswith(_PLACE_PREFIX) and k not in p for k in c)


def op_cache_key(op: object) -> str | None:
    """Cache key for any kernel-bearing op, or ``None`` if not cacheable."""
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from emmy.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        # Name-invariant: the kernel function name is rendered into the source
        # (``void <name>(...)``) but doesn't change runtime behavior. Normalize
        # it out so renaming a kernel (e.g. via op provenance) neither busts the
        # perf cache nor blocks an isolated-kernel tune from transferring to a
        # whole-model compile.
        src = op.kernel_source.replace(op.kernel_name, "_K_") if op.kernel_name else op.kernel_source
        return digest("CudaOp", src, op.arg_order, op.grid, op.block, op.smem_bytes)
    if isinstance(op, (LoopOp, TileOp, KernelOp)):
        # Knobs are part of the key: same-body / different-knobs variants must
        # not collide with their parent in the search tree, or
        # ``SearchTree.expand`` self-parents the node and
        # ``_propagate_expected`` walks the parent chain forever.
        knob_key = tuple(sorted(op.knobs.items())) if op.knobs else ()
        # ``TileOp`` has no stored body — its compute is the ``op`` tree; lower it on
        # demand for the structural key (the body proper is generated at materialize).
        if isinstance(op, TileOp):
            from emmy.compiler.ir.stmt.body import Body  # noqa: PLC0415
            from emmy.compiler.ir.tile.ops import lower  # noqa: PLC0415

            body = Body(lower(op.op)) if op.op is not None else Body(())
        else:
            body = op.body
        return digest(type(op).__name__, body.structural_key(), knob_key)
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
