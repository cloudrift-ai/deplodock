"""Compile-time tuning knobs — heuristic defaults + env-var overrides.

Two matmul-shape configs:

- **Asymmetric "cuBLAS" tile** (TMA-only, big matmul): per-CTA
  ``(BN=128, BM=64)``, per-thread ``(F_M=8, F_N=4)`` → 32 outputs /
  thread, post-split ``(32, 8) = 256`` threads / CTA. Block dim ``(32,
  8)``: each warp shares one M row → A-load broadcasts; threads stride
  consecutive N cols → B-load LDS.128. No bank conflicts. Validated at
  105% of cuBLAS on 2048² fp32 RTX 5090.

- **Symmetric tile** (cp.async fallback / small matmul): per-CTA
  ``(PAT × PAT)``, per-thread ``(F × F)``. PAT scales with shape (32 /
  64 / 128 tiers). Used when TMA is off OR when the matmul is too
  small for the asymmetric tile (small ``M`` or ``N`` doesn't yield
  enough CTA parallelism, and TMA's 16 B alignment requirement breaks
  on tiny operands).

Env vars:

- ``DEPLODOCK_TB`` — total thread budget per CTA (default 256).
- ``DEPLODOCK_BK`` — K-split size for ``002_split_matmul_k`` (subject
  to the ``K % BK == 0 and K > BK`` divisibility check). Default 32
  for the asymmetric path; for symmetric, picked from the candidate
  list (or the big / huge tier's BK heuristic).
- ``DEPLODOCK_COOP_BLOCK`` — cooperative-reduce thread count.
- ``DEPLODOCK_TMA`` — emit ``cp.async.bulk.tensor`` (TMA) loads + the
  asymmetric tile + runtime weight transpose
  (``004a_fold_constant_transpose``). Off by default — small matmuls
  hit TMA's alignment requirement; SDPA / RoPE interact with the
  weight pre-transpose in ways the current passes don't fully handle.
  Set ``DEPLODOCK_TMA=1`` for ``nn.Linear``-only models with
  ``≥1024`` per-dim shapes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Tile


# --- Asymmetric (TMA / big matmul) config -------------------------------

# Per-CTA tile: BM=64 rows, BN=128 cols. Innermost-first.
_ASYM_TILE_SHAPE = (128, 64)
# Per-thread output cells (F_M, F_N). Post-008 split, THREAD axes are
# (BN/F_N, BM/F_M) = (32, 8) = 256 threads/CTA.
_ASYM_F_PER_AXIS = (8, 4)
# Per-stage K-tile, M-adaptive. Sweep on TinyLlama Q/Gate/Down
# projections at seq ∈ {32, 128, 512} on RTX 5090 fp32:
#
#   M ≤ 128: BK=64 wins everywhere (default 67-178us, TMA 55-135us)
#   M = 512:
#     - default: BK=16 wins universally (207us Q, 415us Gate, 551us Down)
#       BK=64 catastrophically slow (866-2242us) — likely smem overflow
#       at the BIG-tier PAT=64 shape.
#     - TMA: BK=32 wins for Q (104us) and Down (303us);
#       Gate seq=512 wins at BK=16 (246us) but BK=32 close (253us).
#
# Threshold at M ≤ 256 (cuts cleanly between Q-128 / Q-512). Q-32
# TMA is the one suboptimal corner (heuristic picks BK=64=111us;
# best is BK=16=56us) — small K + small M + small N means TMA
# overhead doesn't amortize.
_ASYM_M_THRESHOLD = 256
_ASYM_BK_SMALL_M = 64
_ASYM_BK_LARGE_M_DEFAULT = 16  # cp.async path, M > 256
_ASYM_BK_LARGE_M_TMA = 32  # TMA path, M > 256


# --- Symmetric (cp.async fallback) tier thresholds ----------------------

# Output extent threshold: both M and N must be ≥ this for "big matmul"
# to engage the BIG tier (PAT=64). Below it, the K-loop's smem-load
# round-trip overhead per output element wipes out the per-thread
# arithmetic gain.
_BIG_MATMUL_AXIS_MIN = 256

# HUGE tier (PAT=128): only kicks in when the parallel grid is large
# enough to amortize the larger tile's setup cost.
_HUGE_MATMUL_M_MIN = 128
_HUGE_MATMUL_N_MIN = 2048
_HUGE_MATMUL_MN_MIN = 819_200

# Symmetric per-axis thread tile widths and paired F factors. Each tier
# pairs PAT/F = 16 so post-register-tile lands at 256 threads/CTA.
_PAT_DEFAULT = 32
_PAT_BIG = 64
_PAT_HUGE = 128
_PAT_TO_FACTOR = {32: 2, 64: 4, 128: 8}

# HUGE-tier BK (PAT=128 path — keeps stage smem under the 48 KB cap
# at the larger tile shape).
_BK_HUGE = 16


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
    blockified) ``Tile``. ``005_blockify_launch`` splits each output
    axis into ``axis_o BLOCK + axis_i THREAD`` adjacent in
    ``tile.axes`` (BLOCK outer, THREAD inner). The IR's normalize step
    renames axes to ``a{n}`` so the ``_o``/``_i`` suffix is gone by
    the time 008 sees the tile — but the *positional pairing* of
    BLOCK-then-THREAD (or just THREAD when the source extent matched
    the target whole) is preserved. Walk the axes, fold each
    BLOCK-THREAD pair into a single extent (their product), and
    standalone axes (BLOCK or THREAD alone) pass through.

    Returns the list of recovered extents sorted descending so callers
    can apply per-axis thresholds pre- and post-blockify uniformly."""
    from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD

    extents: list[int] = []
    i = 0
    while i < len(tile.axes):
        ba = tile.axes[i]
        ext = int(ba.axis.extent)
        # BLOCK axis followed by THREAD axis = a split pair.
        if ba.bind == BIND_BLOCK and i + 1 < len(tile.axes) and tile.axes[i + 1].bind == BIND_THREAD:
            extents.append(ext * int(tile.axes[i + 1].axis.extent))
            i += 2
            continue
        extents.append(ext)
        i += 1
    return sorted(extents, reverse=True)


def _is_big_matmul(tile: Tile) -> bool:
    """Tile's two largest output axes both ≥ ``_BIG_MATMUL_AXIS_MIN``."""
    if not _has_matmul_reduce(tile.body):
        return False
    extents = _logical_output_extents(tile)
    if len(extents) < 2:
        return False
    return extents[0] >= _BIG_MATMUL_AXIS_MIN and extents[1] >= _BIG_MATMUL_AXIS_MIN


def _is_huge_matmul(tile: Tile) -> bool:
    """Matmul with M ≥ 128, N ≥ 2048, M·N ≥ 819 200."""
    if not _has_matmul_reduce(tile.body):
        return False
    extents = _logical_output_extents(tile)
    if len(extents) < 2:
        return False
    n, m = extents[0], extents[1]
    return m >= _HUGE_MATMUL_M_MIN and n >= _HUGE_MATMUL_N_MIN and m * n >= _HUGE_MATMUL_MN_MIN


def _matmul_M(tile: Tile) -> int:
    """The matmul's M extent — the smaller of the two largest output
    axes. By convention output[M, N] has M=batch×seq (small at
    inference, large at training) and N=hidden (large)."""
    extents = _logical_output_extents(tile)
    return extents[1] if len(extents) >= 2 else 0


def _pick_big_bk(tile: Tile) -> int:
    """BK for the symmetric (cp.async) big-matmul path."""
    if _matmul_M(tile) <= _ASYM_M_THRESHOLD:
        return _ASYM_BK_SMALL_M  # 64 — LLM inference shape
    return _ASYM_BK_LARGE_M_DEFAULT  # 16 — square training shape


def _pick_asym_bk(tile: Tile) -> int:
    """BK for the asymmetric (TMA) path."""
    if _matmul_M(tile) <= _ASYM_M_THRESHOLD:
        return _ASYM_BK_SMALL_M  # 64 — LLM inference shape
    return _ASYM_BK_LARGE_M_TMA  # 32 — square / training shape


def _is_thread_axis(ba) -> bool:
    from deplodock.compiler.ir.axis import BIND_THREAD

    return ba.bind == BIND_THREAD


def _has_blockified_thread_axes(tile: Tile) -> bool:
    return any(_is_thread_axis(ba) for ba in tile.axes)


def _detect_pat(tile: Tile) -> int | None:
    """Return PAT if ≥2 THREAD axes share a known PAT candidate's
    extent; ``None`` pre-blockify or no match."""
    for cand in sorted(_PAT_TO_FACTOR, reverse=True):
        if sum(1 for ba in tile.axes if _is_thread_axis(ba) and int(ba.axis.extent) == cand) >= 2:
            return cand
    return None


def _predicted_pat(tile: Tile) -> int:
    if _is_huge_matmul(tile):
        return _PAT_HUGE
    if _is_big_matmul(tile):
        return _PAT_BIG
    return _PAT_DEFAULT


# --- Public API ---------------------------------------------------------


def _tma_enabled() -> bool:
    """TMA is the cuBLAS-beating SGEMM path on sm_90+. Off by default —
    small matmuls hit TMA's 16 B global-address alignment requirement;
    SDPA / RoPE interact with the weight pre-transpose in ways the
    current passes don't fully handle. Set ``DEPLODOCK_TMA=1`` for
    ``nn.Linear``-only models with ``≥1024`` per-dim shapes."""
    return os.environ.get("DEPLODOCK_TMA") == "1"


def _use_asymmetric(tile: Tile | None) -> bool:
    """Asymmetric tile fires when TMA is on AND the matmul is big
    enough to amortize TMA setup. Otherwise we fall back to the
    symmetric tile (small matmul / non-TMA)."""
    if not _tma_enabled() or tile is None:
        return False
    if not _has_matmul_reduce(tile.body):
        return False
    return _is_big_matmul(tile)


def thread_tile_shape(tile: Tile | None = None) -> tuple[int, ...]:
    """Per-axis THREAD-tile widths ``005_blockify_launch`` should emit,
    innermost-first. ``(BN=128, BM=64)`` for the asymmetric TMA path,
    ``(PAT, PAT)`` for symmetric matmul, ``(thread_budget,)`` for
    non-matmul kernels.

    Env overrides for sweeps: ``DEPLODOCK_BN`` and ``DEPLODOCK_BM``
    replace the asymmetric tile dims (innermost N then outer M)."""
    if _use_asymmetric(tile):
        return _asym_tile_shape_env_override()
    if tile is not None and _has_matmul_reduce(tile.body):
        return (_predicted_pat(tile), _predicted_pat(tile))
    return (thread_budget(),)


def _asym_tile_shape_env_override() -> tuple[int, int]:
    bn = _int_env("DEPLODOCK_BN", _ASYM_TILE_SHAPE[0])
    bm = _int_env("DEPLODOCK_BM", _ASYM_TILE_SHAPE[1])
    return (bn, bm)


def _asym_f_per_axis_env_override() -> tuple[int, int]:
    f_m = _int_env("DEPLODOCK_FM", _ASYM_F_PER_AXIS[0])
    f_n = _int_env("DEPLODOCK_FN", _ASYM_F_PER_AXIS[1])
    return (f_m, f_n)


def register_tile_shape(tile: Tile | None = None) -> tuple[int, int]:
    """Per-thread output tile ``(F_M, F_N)``. Asymmetric ``(8, 4)`` for
    the TMA-eligible big-matmul path, symmetric ``(F, F)`` for
    fallback. Returns ``(1, 1)`` to skip register tiling on small
    matmuls and non-matmul bodies.

    Env overrides for sweeps: ``DEPLODOCK_FM`` / ``DEPLODOCK_FN``
    replace the asymmetric per-thread cell dims."""
    if _use_asymmetric(tile):
        return _asym_f_per_axis_env_override()
    if tile is None:
        return (1, 1)
    if not _has_matmul_reduce(tile.body):
        return (1, 1)
    if not _has_blockified_thread_axes(tile):
        # Pre-blockify: predict PAT and pair it with F.
        pat = _predicted_pat(tile)
        f = _PAT_TO_FACTOR[pat]
        return (f, f)
    pat = _detect_pat(tile)
    if pat is None:
        return (1, 1)
    f = _PAT_TO_FACTOR[pat]
    return (f, f)


def thread_budget() -> int:
    return _int_env("DEPLODOCK_TB", 256)


def forced_bk(tile: Tile | None = None) -> int | None:
    """Force BK via env, or pick a default per the tile's matmul tier."""
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if tile is None or not _has_matmul_reduce(tile.body):
        return None
    if _use_asymmetric(tile):
        return _pick_asym_bk(tile)
    if _is_huge_matmul(tile):
        return _BK_HUGE
    if _is_big_matmul(tile):
        return _pick_big_bk(tile)
    return None


def cooperative_block_size() -> int:
    return _int_env("DEPLODOCK_COOP_BLOCK", 256)
