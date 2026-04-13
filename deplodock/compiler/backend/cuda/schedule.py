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
class GridSpec:
    """How thread blocks map to output elements."""

    type: str  # "1d" | "2d_swizzle" | "2d_standard"
    block_size: tuple[int, int, int]
    # 1d grids use a single bound variable.
    bound: str = "n"  # "n" for pointwise, "rows" for reduce


@dataclass
class AccumulatorSpec:
    """Shape and type of per-thread accumulators.

    shape=None  → pointwise (no accumulators)
    shape=()    → scalar accumulator per reduce op
    shape=(M,N) → 2D register tile (contraction)
    """

    shape: tuple[int, ...] | None
    dtype: str = "float"


@dataclass
class ReductionSpec:
    """One reduction loop in the kernel."""

    dim: str  # semantic name: "cols" | "K"
    loop_var: str  # C variable: "j" | "k"
    start: str  # "threadIdx.x" | "0"
    end: str  # "cols" | "K"
    step: str | None  # "blockDim.x" | None (increment by 1)
    warp_reduce_after: bool  # emit WarpReduce after this loop


@dataclass
class Schedule:
    """Complete kernel structure specification."""

    grid: GridSpec
    accum: AccumulatorSpec
    reductions: list[ReductionSpec]

    # Contraction tile dims (only when accum.shape is 2D)
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

    # Scalar params to append to kernel signature (dim args)
    dim_params: list[tuple[str, str]] = field(default_factory=list)


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
            accum=AccumulatorSpec(None),
            reductions=[],
            dim_params=[("int", "n")],
        )

    if pattern == "contraction":
        tx, ty = 32, 8
        thread_m = int(hints.get("thread_m", 8))
        thread_n = 4
        tile_m = ty * thread_m
        tile_n = tx * thread_n  # 128
        k_splits = int(hints.get("k_splits", 1))
        block_k = int(hints.get("block_k", 16))

        dim_params: list[tuple[str, str]] = [("int", "M"), ("int", "N"), ("int", "K")]
        if is_batched:
            dim_params.append(("int", "batch_count"))

        return Schedule(
            grid=GridSpec("2d_swizzle", (tx, ty, 1)),
            accum=AccumulatorSpec((thread_m, thread_n)),
            reductions=[
                ReductionSpec(
                    dim="K",
                    loop_var="k",
                    start="0",
                    end="K",
                    step=None,
                    warp_reduce_after=False,
                ),
            ],
            tile_m=tile_m,
            tile_n=tile_n,
            thread_m=thread_m,
            thread_n=thread_n,
            k_splits=k_splits,
            load_strategy=strategy if strategy in ("smem", "tma") else "global",
            block_k=block_k,
            is_batched=is_batched,
            dim_params=dim_params,
        )

    # row_reduce or reduce_broadcast
    n_reduces = len(analysis.op_phases.reduces)
    reduce_specs = []
    for _i in range(n_reduces):
        reduce_specs.append(
            ReductionSpec(
                dim="cols",
                loop_var="j",
                start="threadIdx.x",
                end="cols",
                step="blockDim.x",
                warp_reduce_after=True,
            )
        )

    return Schedule(
        grid=GridSpec("1d", (256, 1, 1), bound="rows"),
        accum=AccumulatorSpec(()),
        reductions=reduce_specs,
        epilogue_per_element=analysis.epilogue_needs_per_element,
        dim_params=[("int", "rows"), ("int", "cols")],
    )
