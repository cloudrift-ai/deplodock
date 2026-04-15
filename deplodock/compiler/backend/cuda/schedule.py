"""Kernel schedule: all parameters that determine kernel structure.

A Schedule fully specifies how a fused region maps to GPU threads,
what accumulator shape to use, how to loop over reduced dimensions,
and how to write outputs.  The lowering function ``lower_generic()``
reads the Schedule and emits LoopIR mechanically — no pattern matching.

``build_schedule()`` synthesizes a Schedule from a ``TileAnalysis``
plus strategy and tuning hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis


@dataclass
class TMALoadConfig:
    """TMA descriptor references for async bulk copy."""

    a_tma_ref: str  # "&A_tma" or "&A_tma[batch]"
    b_tma_ref: str  # "&B_tma" or "&B_tma[batch]"


def _safe(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


@dataclass
class GridSpec:
    """How thread blocks map to output elements."""

    type: str  # "1d" | "2d_swizzle" | "2d_standard"
    block_size: tuple[int, int, int]
    # 1d grids use a single bound variable.
    bound: str = "n"  # "n" for pointwise, "rows" for reduce


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

    # Batching
    is_batched: bool = False

    # Epilogue
    epilogue_per_element: bool = False

    # TMA-specific (set by build_schedule when strategy is tma)
    tma_params: list[str] | None = None
    tma_config: object | None = None
    includes: list[str] | None = None

    # Write base expressions (smem uses row_base/col_base instead of bm+tr/bn+tc)
    row_base_var: str | None = None  # "row_base"
    col_base_var: str | None = None  # "col_base"

    # Scalar params to append to kernel signature (dim args)
    dim_params: list[tuple[str, str]] = field(default_factory=list)

    # Occupancy / smem hints (consumed by KernelDef, not by LoopIR)
    extra_smem_bytes: int = 0
    min_blocks_per_sm: int = 0


def build_schedule(
    analysis: TileAnalysis,
    strategy: str = "naive",
    hints: dict | None = None,
) -> Schedule:
    """Synthesize a Schedule from analysis + strategy + tuning hints.

    This replaces the implicit schedule decisions previously scattered
    across six lowering functions.
    """
    hints = hints or {}
    pattern = analysis.pattern
    is_batched = analysis.batch_size > 1

    if pattern == "pointwise":
        return Schedule(
            grid=GridSpec("1d", (256, 1, 1), bound="n"),
            dim_params=[("int", "n")],
        )

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

        dim_params: list[tuple[str, str]] = [("int", "M"), ("int", "N"), ("int", "K")]
        if is_batched:
            dim_params.append(("int", "batch_count"))
        elif k_splits > 1:
            dim_params.append(("int", "k_splits"))

        # Contraction + multi-reduce (e.g. softmax): use 1D grid with N-tiling.
        has_multi_reduce = len(analysis.reduce_fns) > 1
        if has_multi_reduce:
            grid_type = "1d_contraction"
            k_splits = 1  # incompatible with online reduction
        elif load_strat == "smem":
            grid_type = "2d_standard"
        else:
            grid_type = "2d_swizzle"

        # TMA-specific fields
        tma_params = None
        tma_config = None
        includes = None
        if load_strat == "tma" and analysis.contraction_a:
            a_name = _safe(analysis.contraction_a)
            b_name = _safe(analysis.contraction_b)
            tma_a_ref = f"&{a_name}_tma[batch]" if is_batched else f"&{a_name}_tma"
            tma_b_ref = f"&{b_name}_tma[batch]" if is_batched else f"&{b_name}_tma"
            tma_params = [f"{a_name}_tma", f"{b_name}_tma"]
            tma_config = TMALoadConfig(a_tma_ref=tma_a_ref, b_tma_ref=tma_b_ref)
            includes = ["cuda.h"]

        return Schedule(
            grid=GridSpec(grid_type, (tx, ty, 1)),
            tile_m=tile_m,
            tile_n=tile_n,
            thread_m=thread_m,
            thread_n=thread_n,
            k_splits=k_splits,
            load_strategy=load_strat,
            block_k=block_k,
            is_batched=is_batched,
            dim_params=dim_params,
            tma_params=tma_params,
            tma_config=tma_config,
            includes=includes,
            row_base_var="row_base" if load_strat == "smem" else None,
            col_base_var="col_base" if load_strat == "smem" else None,
        )

    # row_reduce or reduce_broadcast
    return Schedule(
        grid=GridSpec("1d", (256, 1, 1), bound="rows"),
        epilogue_per_element=analysis.epilogue_needs_per_element,
        dim_params=[("int", "rows"), ("int", "cols")],
    )
