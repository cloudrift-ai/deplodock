"""Compile-time tuning knobs — heuristic defaults + env-var overrides.

Single matmul tile config (formerly the "asymmetric" / cuBLAS-style tile;
the symmetric ``PAT × PAT`` tier was retired). Per-CTA ``(BN=128, BM=64)``,
per-thread ``(F_M=8, F_N=4)`` → 32 outputs / thread, post-split
``(BN/F_N, BM/F_M) = (32, 8) = 256`` threads / CTA. Block dim ``(32, 8)``:
each warp shares one M row → A-load broadcasts; threads stride consecutive
N cols → B-load LDS.128. Validated at 105% of cuBLAS on 2048² fp32 RTX 5090.

Env vars:

- ``DEPLODOCK_BN``, ``DEPLODOCK_BM`` — per-CTA tile (innermost N, outer M).
- ``DEPLODOCK_FN``, ``DEPLODOCK_FM`` — per-thread output cells.
- ``DEPLODOCK_BK`` — K-split size for ``002_split_matmul_k`` (subject
  to ``K % BK == 0 and K > BK``). Default is M-adaptive.
- ``DEPLODOCK_TB`` — total thread budget per CTA for non-matmul kernels.
- ``DEPLODOCK_COOP_BLOCK`` — cooperative-reduce thread count.
- ``DEPLODOCK_TMA`` — emit ``cp.async.bulk.tensor`` (TMA) loads + runtime
  weight transpose (``004a_fold_constant_transpose``). Off by default
  — TMA's descriptor encoder rejects source ranks > 5, which the SDPA
  Q/K/V path produces; enabling globally regresses attention kernels.
  Set ``DEPLODOCK_TMA=1`` for ``nn.Linear``-only models with
  ``≥1024`` per-dim shapes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Tile


# Per-CTA matmul tile (innermost N, outer M).
_TILE_SHAPE = (128, 64)
# Per-thread output cells (F_M, F_N). Block dim = (BN/F_N, BM/F_M) = (32, 8) = 256 threads.
_F_PER_AXIS = (8, 4)

# Per-stage K-tile, M-adaptive. Sweep on TinyLlama Q/Gate/Down at seq ∈
# {32, 128, 512} on RTX 5090 fp32: M ≤ 256 → BK=64 wins (small grid, K
# loop must amortize CTA setup); M > 256 → BK=16 (cp.async) or BK=32
# (TMA) — larger BK at M=512 catastrophically slow (smem overflow).
_M_THRESHOLD = 256
_BK_SMALL_M = 64
_BK_LARGE_M_DEFAULT = 16  # cp.async path
_BK_LARGE_M_TMA = 32      # TMA path


# --- Helpers ------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _has_matmul_reduce(stmts) -> bool:
    """Body contains a reduce ``Loop`` whose immediate Loads touch ≥2
    distinct buffers — the structural signature of a matmul. Recurses
    through wrapper loops / conds so chunked or output-loop-wrapped
    matmuls (SDPA V-projection) are still detected."""
    from deplodock.compiler.ir.stmt import Load, Loop
    from deplodock.compiler.ir.stmt.body import Body

    return any(
        isinstance(s, Loop) and s.is_reduce and len({ld.input for ld in s.body.of_type(Load)}) >= 2 for s in Body.coerce(stmts).iter()
    )


def _logical_output_extents(tile: Tile) -> list[int]:
    """Recover the pre-blockify output extents from a (possibly
    blockified) ``Tile``. Walks ``tile.axes`` folding adjacent
    BLOCK-then-THREAD pairs into a single extent. Returns sorted
    descending."""
    from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD

    extents: list[int] = []
    i = 0
    while i < len(tile.axes):
        ba = tile.axes[i]
        ext = int(ba.axis.extent)
        if ba.bind == BIND_BLOCK and i + 1 < len(tile.axes) and tile.axes[i + 1].bind == BIND_THREAD:
            extents.append(ext * int(tile.axes[i + 1].axis.extent))
            i += 2
            continue
        extents.append(ext)
        i += 1
    return sorted(extents, reverse=True)


def _matmul_M(tile: Tile) -> int:
    """The matmul's M extent — the smaller of the two largest output
    axes. Convention: output[M, N] has M=batch×seq, N=hidden."""
    extents = _logical_output_extents(tile)
    return extents[1] if len(extents) >= 2 else 0


# --- Public API ---------------------------------------------------------


def _tma_enabled() -> bool:
    return os.environ.get("DEPLODOCK_TMA") == "1"


def thread_tile_shape(tile: Tile | None = None) -> tuple[int, ...]:
    """Per-axis THREAD-tile widths ``005_blockify_launch`` should emit,
    innermost-first. ``(BN, BM)`` for matmul, ``(thread_budget,)`` for
    non-matmul kernels."""
    if tile is not None and _has_matmul_reduce(tile.body):
        bn = _int_env("DEPLODOCK_BN", _TILE_SHAPE[0])
        bm = _int_env("DEPLODOCK_BM", _TILE_SHAPE[1])
        return (bn, bm)
    return (thread_budget(),)


def register_tile_shape(tile: Tile | None = None) -> tuple[int, int]:
    """Per-thread output tile ``(F_M, F_N)``. ``(1, 1)`` to skip
    register tiling on non-matmul bodies and on matmuls whose post-
    blockify THREAD axes are too small to host the F-tile (tiny matmul
    fallback — blockify kept the original output extents whole because
    they were already < tile width)."""
    if tile is None or not _has_matmul_reduce(tile.body):
        return (1, 1)
    f_m = _int_env("DEPLODOCK_FM", _F_PER_AXIS[0])
    f_n = _int_env("DEPLODOCK_FN", _F_PER_AXIS[1])
    bn = _int_env("DEPLODOCK_BN", _TILE_SHAPE[0])
    bm = _int_env("DEPLODOCK_BM", _TILE_SHAPE[1])
    # Post-blockify: each output's THREAD axis should equal the per-axis
    # tile width (BN, BM). If neither matches, blockify kept the axes
    # whole — register-tiling would just rename them, no parallelism.
    from deplodock.compiler.ir.axis import BIND_THREAD

    if any(ba.bind == BIND_THREAD for ba in tile.axes):
        thread_extents = {int(ba.axis.extent) for ba in tile.axes if ba.bind == BIND_THREAD}
        if not (thread_extents & {bn, bm}):
            return (1, 1)
    return (f_m, f_n)


def thread_budget() -> int:
    return _int_env("DEPLODOCK_TB", 256)


def forced_bk(tile: Tile | None = None) -> int | None:
    """Force BK via env, or pick the M-adaptive default. Backs off to
    a smaller BK when the M-adaptive choice would overflow the 48 KB
    static-smem cap at the active tile shape — happens for small
    matmuls where the THREAD axes stay at the full output extent and
    BK_SMALL_M=64 produces a stage too large for two buffers."""
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if tile is None or not _has_matmul_reduce(tile.body):
        return None
    bk = _BK_SMALL_M if _matmul_M(tile) <= _M_THRESHOLD else (_BK_LARGE_M_TMA if _tma_enabled() else _BK_LARGE_M_DEFAULT)
    return _bk_fits_smem(tile, bk)


# sm_120 hard cap is 48 KB static smem; 014c_pad_smem_banks adds +1
# row of padding per stage to break 32-way bank conflicts, so reserve
# ~4 KB of headroom for that swelling.
_SMEM_BUDGET_BYTES = 44 * 1024
_DTYPE_BYTES = 4


def _bk_fits_smem(tile: Tile, bk: int) -> int:
    """Halve BK until the (BN+BM)·BK·4·2 stage fits in 48 KB static
    smem. Returns the largest BK ≥ 1 that fits."""
    extents = _logical_output_extents(tile)
    if len(extents) < 2:
        return bk
    bn_actual = min(extents[0], _int_env("DEPLODOCK_BN", _TILE_SHAPE[0]))
    bm_actual = min(extents[1], _int_env("DEPLODOCK_BM", _TILE_SHAPE[1]))
    while bk > 1 and (bn_actual + bm_actual) * bk * _DTYPE_BYTES * 2 > _SMEM_BUDGET_BYTES:
        bk //= 2
    return bk


def cooperative_block_size() -> int:
    return _int_env("DEPLODOCK_COOP_BLOCK", 256)
