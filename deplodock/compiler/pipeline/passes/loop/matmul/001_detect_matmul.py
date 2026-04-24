"""Annotate ``LoopOp`` nodes whose body is a contraction as TMA-matmul.

Uses the shared ``analyze_matmul`` oracle from the matmul emitter: if it
returns a ``MatmulInfo`` the body is emittable by the tiled-SGEMM
template (bare matmul, matmul+epilogue, or matmul+prologue+epilogue),
so the pass stamps ``cuda.matmul.*`` hints. Otherwise the graph is left
unchanged and the scalar emitter takes over.

The pass only *annotates* — returning ``None`` from ``rewrite`` keeps
the graph unchanged; downstream ``lowering/kernel`` reads the hint and
dispatches to the matmul template.

Tile config is picked by M (the output's leading free-axis extent).
TinyLlama hits M∈{32, 512} at seq∈{32, 512}; other LLMs hit similar
powers of two. The two regimes pick very different optima — small-M
wants a modest BM and few threads to avoid over-launching, large-M
wants BM=128 with TM=8 to amortize smem stage over more compute.
Picked by ``scripts/sweep_matmul_tiles.py``:

  M ≥ 128 (and M % 128 == 0):  BM=128, BN=32, BK=64, TM=8, TN=4, threads=128
  M  = 32..64                  BM=32,  BN=16, BK=64, TM=4, TN=2, threads=64

Each value is overridable via ``DEPLODOCK_MATMUL_<KEY>`` so sweep scripts
can probe alternatives without patching the source. When env-overriden,
the value replaces the auto-picked tile entry.
"""

from __future__ import annotations

import os

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._emit_matmul import analyze_matmul

PATTERN = [Pattern("root", LoopOp)]

_TILE_LARGE_M = {
    "tile_m": 128,
    "tile_n": 32,
    "block_k": 64,
    "thread_m": 8,
    "thread_n": 4,
    "threads": 128,
}

_TILE_SMALL_M = {
    "tile_m": 32,
    "tile_n": 16,
    "block_k": 64,
    "thread_m": 4,
    "thread_n": 2,
    "threads": 64,
}


def _pick_tile_for_m(m: int) -> dict[str, int] | None:
    """Pick a shape-appropriate tile config for the matmul's M extent.

    Returns None when M doesn't divide any supported BM — caller falls
    back to the scalar emitter.
    """
    if m % _TILE_LARGE_M["tile_m"] == 0:
        return dict(_TILE_LARGE_M)
    if m % _TILE_SMALL_M["tile_m"] == 0:
        return dict(_TILE_SMALL_M)
    return None


def _apply_env_overrides(tile: dict[str, int]) -> dict[str, int]:
    for key in tile:
        env = os.environ.get(f"DEPLODOCK_MATMUL_{key.upper()}")
        if env is not None:
            tile[key] = int(env)
    return tile


def _hints_for(info) -> dict | None:
    m = int(info.m_axis.extent)
    n = int(info.n_axis.extent)
    k = int(info.k_axis.extent)

    # Env override path: if *any* DEPLODOCK_MATMUL_* env var is set, the
    # sweep script (or a manual override) wants a fixed tile. Start from
    # the large-M defaults as a base, then let env fully override.
    env_any = any(os.environ.get(f"DEPLODOCK_MATMUL_{k_.upper()}") is not None for k_ in _TILE_LARGE_M)
    if env_any:
        tile = _apply_env_overrides(dict(_TILE_LARGE_M))
    else:
        tile = _pick_tile_for_m(m)
        if tile is None:
            return None

    # M / N / K must divide their tile / block sizes. The template uses a 2D
    # grid (gridDim.y = M / BM, gridDim.x = N / BN), so M >= BM with exact
    # divisibility is enough. The thread-tile arithmetic (BM/TM * BN/TN ==
    # threads) is re-checked in the emitter.
    if m % tile["tile_m"] or n % tile["tile_n"] or k % tile["block_k"]:
        return None
    if (tile["tile_m"] // tile["thread_m"]) * (tile["tile_n"] // tile["thread_n"]) != tile["threads"]:
        return None
    return {
        "cuda.matmul.strategy": "tma_matmul",
        "cuda.matmul.m": m,
        "cuda.matmul.n": n,
        "cuda.matmul.k": k,
        **{f"cuda.matmul.{k_}": v for k_, v in tile.items()},
    }


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.nodes["root"]]
    if not isinstance(node.op, LoopOp):
        return None
    if node.hints.has("cuda.matmul.strategy"):
        return None
    info = analyze_matmul(node.op)
    if info is None:
        return None
    hints = _hints_for(info)
    if hints is None:
        return None
    for key, val in hints.items():
        node.hints.set(key, val)
    return None
