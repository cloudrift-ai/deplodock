"""Op-key derivation + source-chain walking for the search package.

Shared by :mod:`.db`, :mod:`.policy`, and the bench-terminal helper
in :mod:`deplodock.compiler.pipeline.pipeline`.

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

from deplodock.compiler.structural import digest

Dialect = Literal["loop", "tile", "kernel", "cuda"]


def op_cache_key(op: object) -> str | None:
    """Cache key for any kernel-bearing op, or ``None`` if not cacheable."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        return digest("CudaOp", op.kernel_source, op.arg_order, op.grid, op.block, op.smem_bytes)
    if isinstance(op, (LoopOp, TileOp, KernelOp)):
        # Knobs are part of the key: same-body / different-knobs variants
        # (e.g. ``020_stage_inputs`` emits a no-op TileOp with a
        # ``STAGE="0..0"`` knob to mark "considered, declined" decisions)
        # must not collide with their parent in the search tree, or
        # ``SearchTree.expand`` self-parents the node and
        # ``_propagate_expected`` walks the parent chain forever.
        knob_key = tuple(sorted(op.knobs.items())) if op.knobs else ()
        return digest(type(op).__name__, op.body.structural_key(), knob_key)
    return None


def dialect_of(op: object) -> Dialect | None:
    """Return the dialect tag for any kernel-bearing op, or ``None``."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

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
