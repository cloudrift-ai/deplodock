"""Compile-time tuning knobs — heuristic defaults + env-var overrides.

Each tunable is a hardcoded compiler constant (tile factor, thread tile
size, K-chunk size, …) that scheduling decisions make heuristically.
The defaults pick a config based on the kernel's shape: the **default
small-matmul / pointwise / reduce config** is ``PAT=16, F=2, TB=256,
BK=64`` — solid across most kernels in our benchmark set. When the
kernel is a **big matmul** (parallel output ≥ 4096 elements *and* the
body has a reduce loop with ≥2 distinct buffer Loads), the heuristic
switches to ``PAT=32, F=4, TB=256, BK=32`` — empirically ~1.6× faster
than the small config on Linear(3584, 3584) at seq=512.

Env vars override the heuristic for sweeps:

- ``DEPLODOCK_F`` — register-tile factor (per-thread tile is F × F
  output cells; F=1 disables register tiling).
- ``DEPLODOCK_PAT`` — innermost M / N tile width carved by
  ``005_blockify_launch``.
- ``DEPLODOCK_TB`` — total thread budget per CTA.
- ``DEPLODOCK_BK`` — K-chunk size for ``004_chunk_reduce`` (subject to
  the ``K % BK == 0 and K > BK`` divisibility check).
- ``DEPLODOCK_COOP_BLOCK`` — cooperative-reduce thread count.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Tile


# Threshold: every output axis must be ≥ this for a matmul to be "big".
# Both M and N need at least two PAT=32 tiles to make the larger CTA
# pay off — otherwise the grid loses dimensions and we under-saturate.
# Empirically: Linear(3584,3584) at seq=512 (M=512,N=3584) → big config
# 1.66× the default; same kernel at seq=32 (M=32) → big config 11×
# slower than default. 64 is the boundary.
_BIG_MATMUL_AXIS_MIN = 64

# Default knobs for non-matmul / small kernels.
_PAT_DEFAULT = 16
_F_DEFAULT = 2
_BK_DEFAULT: int | None = None  # use the built-in candidate list (picks 64 for K≥64)

# Knobs for big matmul kernels.
_PAT_BIG = 32
_F_BIG = 4
_BK_BIG = 32


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
    from deplodock.compiler.ir.stmt import Cond, Loop, StridedLoop

    for s in stmts:
        if isinstance(s, Loop):
            if s.is_reduce and len({ld.input for ld in s.loads}) >= 2:
                return True
            if _has_matmul_reduce(s.body):
                return True
        elif isinstance(s, StridedLoop):
            if _has_matmul_reduce(s.body):
                return True
        elif isinstance(s, Cond):
            if _has_matmul_reduce(s.body) or _has_matmul_reduce(s.else_body):
                return True
    return False


def _is_big_matmul(tile: Tile) -> bool:
    """Tile is a matmul kernel large enough to benefit from PAT=32 / F=4 tiles.

    Every output axis must be ≥ ``_BIG_MATMUL_AXIS_MIN`` so both M and N
    yield ≥ 2 PAT=32 blocks each — otherwise the grid loses an axis and
    SM utilization drops below the small-tile config. Block axes are
    excluded since post-blockify they encode the M_o / N_o split, not
    the original axis extent.
    """
    if not _has_matmul_reduce(tile.body):
        return False
    for ba in tile.axes:
        ext = int(ba.axis.extent)
        if ext < _BIG_MATMUL_AXIS_MIN:
            return False
    return True


def per_axis_threads(tile: Tile | None = None) -> int:
    raw = os.environ.get("DEPLODOCK_PAT")
    if raw:
        return _int_env("DEPLODOCK_PAT", _PAT_DEFAULT)
    if tile is not None and _is_big_matmul(tile):
        return _PAT_BIG
    return _PAT_DEFAULT


def register_tile_factor(tile: Tile | None = None) -> int:
    raw = os.environ.get("DEPLODOCK_F")
    if raw:
        return _int_env("DEPLODOCK_F", _F_DEFAULT)
    if tile is not None and _is_big_matmul(tile):
        return _F_BIG
    return _F_DEFAULT


def thread_budget() -> int:
    return _int_env("DEPLODOCK_TB", 256)


def forced_bk(tile: Tile | None = None) -> int | None:
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if tile is not None and _is_big_matmul(tile):
        return _BK_BIG
    return _BK_DEFAULT


def cooperative_block_size() -> int:
    return _int_env("DEPLODOCK_COOP_BLOCK", 256)
