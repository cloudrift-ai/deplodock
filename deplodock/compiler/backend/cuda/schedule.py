"""Kernel schedule: true scheduling knobs (tile dims, K-split, load strategy,
occupancy hints).  Derived facts (dim_params, tma descriptors, 1D bound
variable, grid flavor) are computed by consumers from the KernelOp via the
``_*`` helpers in ``loop_lower.py``.

``build_schedule()`` synthesizes a Schedule from a KernelOp + strategy
and tuning hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ops import KernelOp


@dataclass
class TMALoadConfig:
    """TMA descriptor references for async bulk copy."""

    a_tma_ref: str  # "&A_tma" or "&A_tma[batch]"
    b_tma_ref: str  # "&B_tma" or "&B_tma[batch]"


@dataclass
class GridSpec:
    """How thread blocks map to output elements."""

    block_size: tuple[int, int, int]


@dataclass
class Schedule:
    """Complete kernel structure specification."""

    grid: GridSpec

    # Contraction tile dims (populated when the region is a contraction)
    tile_m: int | None = None
    tile_n: int | None = None
    thread_m: int | None = None
    thread_n: int | None = None

    # K-splitting
    k_splits: int = 1

    # Load strategy for contraction K-loop
    load_strategy: str = "global"  # "global" | "smem" | "tma"
    block_k: int = 16

    # Write base expressions (smem uses row_base/col_base instead of bm+tr/bn+tc)
    row_base_var: str | None = None  # "row_base"
    col_base_var: str | None = None  # "col_base"

    # Occupancy / smem hints (consumed by KernelDef, not by LoopIR)
    extra_smem_bytes: int = 0
    min_blocks_per_sm: int = 0


def build_schedule(
    region: KernelOp,
    shapes: dict[str, tuple],
    strategy: str = "naive",
    hints: dict | None = None,
) -> Schedule:
    """Synthesize a Schedule from region + shapes + strategy + tuning hints.

    All structural facts (pattern, reduce count, contraction operands,
    batch size) come from KernelOp accessors.
    """
    hints = hints or {}
    out_shape = shapes.get(region.outputs[0].buffer_id, (1,))
    pattern = region.tile_pattern(shapes, out_shape)
    reduce_fns = region.reduce_fn_names()

    if pattern == "pointwise":
        return Schedule(grid=GridSpec((256, 1, 1)))

    if pattern == "contraction":
        load_strat = {"smem": "smem", "tma": "tma", "tma_db": "tma"}.get(strategy, "global")
        k_splits = int(hints.get("k_splits", 1))

        if load_strat == "smem":
            tx, ty = 32, 4
            thread_m = int(hints.get("thread_m", 4))
            block_k = int(hints.get("block_k", 32))
        else:
            tx, ty = 32, 8
            thread_m = int(hints.get("thread_m", 8))
            block_k = int(hints.get("block_k", 16 if load_strat == "global" else 32))

        thread_n = 4
        tile_m = ty * thread_m
        tile_n = tx * thread_n  # 128

        # Contraction + multi-reduce (e.g. softmax): tiles over N (see _grid_type).
        if len(reduce_fns) > 1:
            k_splits = 1  # incompatible with online reduction

        return Schedule(
            grid=GridSpec((tx, ty, 1)),
            tile_m=tile_m,
            tile_n=tile_n,
            thread_m=thread_m,
            thread_n=thread_n,
            k_splits=k_splits,
            load_strategy=load_strat,
            block_k=block_k,
            row_base_var="row_base" if load_strat == "smem" else None,
            col_base_var="col_base" if load_strat == "smem" else None,
        )

    # row_reduce or reduce_broadcast
    return Schedule(grid=GridSpec((256, 1, 1)))
