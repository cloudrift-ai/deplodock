"""Annotate ``LoopOp`` nodes whose body is a contraction as TMA-matmul.

Uses the shared ``analyze_matmul`` oracle from the matmul emitter: if it
returns a ``MatmulInfo`` the body is emittable by the tiled-SGEMM
template (bare matmul, matmul+epilogue, or matmul+prologue+epilogue),
so the pass stamps ``cuda.matmul.*`` hints. Otherwise the graph is left
unchanged and the scalar emitter takes over.

The pass only *annotates* — returning ``None`` from ``rewrite`` keeps
the graph unchanged; downstream ``lowering/kernel`` reads the hint and
dispatches to the matmul template.
"""

from __future__ import annotations

import os

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._emit_matmul import analyze_matmul

PATTERN = [Pattern("root", LoopOp)]

# Default tile config — fits every TinyLlama matmul (M=32, K∈{2048,5632},
# N∈{256,2048,5632}) at seq_len=32: BM=32, BN=32, BK=64, TM=2, TN=4,
# threads=128. Picked by ``scripts/sweep_matmul_tiles.py`` — top or
# near-top across all four TinyLlama shapes (~2× over the previous
# BN=64/BK=32/TN=8 config). Each key is overridable via
# ``DEPLODOCK_MATMUL_<KEY>`` so sweep scripts can probe alternatives
# without patching the source.
_TILE_DEFAULTS = {
    "tile_m": 32,
    "tile_n": 32,
    "block_k": 64,
    "thread_m": 2,
    "thread_n": 4,
    "threads": 128,
}


def _current_tile() -> dict[str, int]:
    tile = dict(_TILE_DEFAULTS)
    for key in tile:
        env = os.environ.get(f"DEPLODOCK_MATMUL_{key.upper()}")
        if env is not None:
            tile[key] = int(env)
    return tile


def _hints_for(info) -> dict | None:
    tile = _current_tile()
    m = int(info.m_axis.extent)
    n = int(info.n_axis.extent)
    k = int(info.k_axis.extent)
    # Template assumes M == BM (one block processes the full M dimension).
    # N and K must divide their tile / block sizes. The thread-tile
    # arithmetic (BM/TM * BN/TN == threads) is re-checked in the emitter.
    if m != tile["tile_m"] or n % tile["tile_n"] or k % tile["block_k"]:
        return None
    if (tile["tile_m"] // tile["thread_m"]) * (tile["tile_n"] // tile["thread_n"]) != tile["threads"]:
        return None
    return {
        "cuda.matmul.strategy": "tma_matmul",
        "cuda.matmul.m": m,
        "cuda.matmul.n": n,
        "cuda.matmul.k": k,
        "cuda.matmul.a_source": info.a_source,
        "cuda.matmul.b_source": info.b_source,
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
