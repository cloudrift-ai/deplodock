"""Compile-time tuning knobs — heuristic defaults + env-var overrides.

Each tunable is a hardcoded compiler constant (tile factor, thread tile
size, K-chunk size, …) that scheduling decisions make heuristically.
The defaults pick a config based on the kernel's shape: the **default
small-matmul / pointwise / reduce config** is ``PAT=32, F=2, TB=256,
BK=64`` — solid across most kernels in our benchmark set. When the
kernel is a **big matmul** (parallel output ≥ 4096 elements *and* the
body has a reduce loop with ≥2 distinct buffer Loads), the heuristic
switches to ``PAT=64, F=4, TB=256, BK=32`` — empirically ~1.6× faster
than the small config on Linear(3584, 3584) at seq=512.

PAT and F are paired so ``PAT/F = 16`` in every tier. ``005_blockify_
launch`` emits PAT-extent THREAD axes per output dim;
``008_register_tile`` splits each by F and F²-replicates the body. The
result: ``(PAT/F)² = 256`` threads/CTA exactly, no rebalance pass
needed.

Env vars override the heuristic for sweeps:

- ``DEPLODOCK_F`` — register-tile factor (per-thread tile is F × F
  output cells; F=1 disables register tiling). Ignored under TMA mode
  which uses the asymmetric ``(F_M=8, F_N=4)`` shape.
- ``DEPLODOCK_PAT`` — innermost M / N tile width carved by
  ``005_blockify_launch``. Ignored under TMA mode (uses ``(BN=128,
  BM=64)``).
- ``DEPLODOCK_TB`` — total thread budget per CTA.
- ``DEPLODOCK_BK`` — K-split size for ``002_split_matmul_k`` (subject to
  the ``K % BK == 0 and K > BK`` divisibility check).
- ``DEPLODOCK_COOP_BLOCK`` — cooperative-reduce thread count.
- ``DEPLODOCK_TMA`` — emit ``cp.async.bulk.tensor`` (TMA) loads for
  staged matmul operands. Implies the cuBLAS-style asymmetric tile
  ``(BN=128, BM=64)`` + per-thread ``(F_M=8, F_N=4)`` and runtime
  weight transpose (via ``004a_fold_constant_transpose``). Beats cp.async
  baseline on sm_120 fp32 SIMT matmul (105% of cuBLAS at 4096²).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Tile


# Threshold: every output axis must be ≥ this for a matmul to be "big".
# Empirically the BIG tier (PAT=32 F=4) only wins when M is "real" —
# 256 is the boundary. Below that, the K-loop's smem-load round-trip
# overhead per output element wipes out the per-thread arithmetic gain.
# Llama/Qwen-shape Linear at seq=128 (M=128,N=3584) lost 30% on BIG;
# seq=512 (M=512,N=3584) won 30% on BIG.
_BIG_MATMUL_AXIS_MIN = 256

# Thresholds for the PAT=64 "huge" tier. M ≥ 128 (any smaller and
# M_grid=1 ⇒ no row parallelism), N ≥ 2048 (need enough 64-wide N
# tiles), M·N ≥ 819 200 (= 200 × 64² ⇒ ≥ 200 CTAs at PAT=64, enough
# to saturate ~70 SMs over multiple waves and amortize the larger
# tile's setup cost). At PAT=64 with BK=16 smem fits regardless of K,
# but the K loop's per-iter overhead means HUGE only wins when the
# parallel grid is large enough — Qwen MLP gate/up at seq=128
# (M=128, N=18944, K=3584) hits the rule with grid=592, and goes
# 2.6× faster than DEFAULT. Down_proj at the same M (M=128, N=3584,
# K=18944, M·N=459K) misses on M·N — DEFAULT wins. Matches sweep.
_HUGE_MATMUL_M_MIN = 128
_HUGE_MATMUL_N_MIN = 2048
_HUGE_MATMUL_MN_MIN = 819_200

# Default knobs for non-matmul / small kernels. PAT is the per-axis
# THREAD-tile width that ``005_blockify_launch`` emits per output dim;
# ``008_register_tile`` then splits it by F so the final per-axis thread
# count is ``PAT/F``. We pick ``PAT/F = 16`` across every tier so that a
# matmul kernel lands at exactly ``thread_budget = 256`` threads/CTA
# right out of register_tile, with no post-hoc rebalance pass needed.
_PAT_DEFAULT = 32
_BK_DEFAULT: int | None = None  # use the built-in candidate list (picks 64 for K≥64)

# Knobs for big matmul kernels.
_PAT_BIG = 64

# Knobs for huge matmul kernels (M·N·K all large but K not too large).
# At PAT=128 the smem stage budget forces BK=16 (BK=32 overflows the
# 48 KB static-smem limit at typical fp32 stages).
_PAT_HUGE = 128
_BK_HUGE = 16
# BK for big matmul: shape-dependent (see ``_pick_big_bk``). Sweep on
# Llama / Qwen-shape Linears showed BK=32 wins for very large K (≥ 8192)
# and for kernels with few CTAs (M·N small), but BK=16 wins for the
# common K=3584 case at large M·N where higher occupancy hides latency.
_BIG_K_LARGE = 8192
_BIG_MN_OVERSATURATED = 1_000_000

# Per-thread register tile factor as a function of the per-axis thread
# tile width chosen by ``005_blockify_launch``. Each thread owns
# ``F × F`` output cells; the table fixes ``PAT/F = 16`` across every
# tier so a matmul kernel lands at ``(PAT/F)² = 256`` threads/CTA after
# ``008_register_tile`` splits — exactly the thread budget, no post-
# hoc rebalance needed. Per-thread arithmetic scales with ``F²``
# (4, 16, 64 outputs/thread for the three tiers).
_PAT_TO_FACTOR = {32: 2, 64: 4, 128: 8}


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


def _is_big_matmul(tile: Tile) -> bool:
    """Tile is a matmul kernel large enough to benefit from PAT=32 / F=4 tiles.

    The two **largest** output axes — the matmul's M and N — must both
    be ≥ ``_BIG_MATMUL_AXIS_MIN``. Any further axes (multi-head /
    batch / per-token batches) get distributed across BLOCKs and don't
    constrain the per-CTA tile shape. Empirically: SDPA's masked
    Q·Kᵀ kernel has axes (heads=32, M=512, N=512); the two largest
    (512, 512) both fit BIG, so we want PAT=32 even though heads=32
    is small. Linear at seq=128 has axes (M=128, N=3584); top two are
    (128, 3584), and M=128 < 256 keeps it on the default path.
    """
    if not _has_matmul_reduce(tile.body):
        return False
    extents = sorted((int(ba.axis.extent) for ba in tile.axes), reverse=True)
    if len(extents) < 2:
        return False
    if extents[0] < _BIG_MATMUL_AXIS_MIN or extents[1] < _BIG_MATMUL_AXIS_MIN:
        return False
    return True


def _is_huge_matmul(tile: Tile) -> bool:
    """Tile is a matmul large enough to want PAT=64 / F=8 tiles.

    Conditions: matmul-shaped, ``M ≥ _HUGE_MATMUL_M_MIN``,
    ``N ≥ _HUGE_MATMUL_N_MIN``, ``K ≤ _HUGE_MATMUL_K_MAX``. M and N are
    derived from the smallest and largest output axis extent (a
    proxy that works pre- and post-blockify since axis splits multiply
    back to the original); K comes from the matmul reduce loop.
    """
    if not _has_matmul_reduce(tile.body):
        return False
    extents = sorted((int(ba.axis.extent) for ba in tile.axes), reverse=True)
    if len(extents) < 2:
        return False
    # The matmul's M and N are the two largest output axes; further
    # axes are batch / head dims that get block-distributed.
    N = extents[0]
    M = extents[1]
    if M < _HUGE_MATMUL_M_MIN or N < _HUGE_MATMUL_N_MIN:
        return False
    if M * N < _HUGE_MATMUL_MN_MIN:
        return False
    return True


def _matmul_K(stmts) -> int | None:
    """Total K extent of the first matmul-shaped reduce loop in ``stmts``.

    Pre-split: the reduce loop holds the full K. Post-``002_split_matmul_k``:
    the structure is ``Loop(K_o, free, body=(Loop(K_i, reduce, ...),))``
    and the total K is ``K_o.extent × K_i.extent``. We recognize that
    chunked shape (free Loop with a single reduce child whose body has
    ≥2 buffer Loads) and return the product so heuristics keyed on
    "original K" don't get fooled by chunked tiles.
    """
    from deplodock.compiler.ir.stmt import Load, Loop
    from deplodock.compiler.ir.stmt.body import Body

    for s in Body.coerce(stmts).iter():
        if not isinstance(s, Loop):
            continue
        if s.is_reduce and len({ld.input for ld in s.body.of_type(Load)}) >= 2:
            return int(s.axis.extent)
        if (
            not s.is_reduce
            and len(s.body) == 1
            and isinstance(s.body[0], Loop)
            and s.body[0].is_reduce
            and len({ld.input for ld in s.body[0].body.of_type(Load)}) >= 2
        ):
            return int(s.axis.extent) * int(s.body[0].axis.extent)
    return None


def _parallel_output(tile: Tile) -> int:
    """Total output volume (product of all axis extents). Pre-blockify this
    is M·N for a matmul; post-blockify the splits multiply back to the same
    product. Excludes reduce axes (they're inside the body, not on tile.axes)."""
    n = 1
    for ba in tile.axes:
        n *= int(ba.axis.extent)
    return n


def _pick_big_bk(tile: Tile) -> int:
    """Shape-aware BK choice for big-matmul tiles.

    - K ≥ 8192 → BK=32: large K dominates runtime, fewer chunk iters wins.
    - M·N ≥ 1M → BK=16: many CTAs, occupancy=50% beats occupancy=33%.
    - else → BK=32: few CTAs, fewer K-iters per CTA wins regardless of occ.
    """
    K = _matmul_K(tile.body)
    if K is not None and K >= _BIG_K_LARGE:
        return 32
    if _parallel_output(tile) >= _BIG_MN_OVERSATURATED:
        return 16
    return 32


def per_axis_threads(tile: Tile | None = None) -> int:
    raw = os.environ.get("DEPLODOCK_PAT")
    if raw:
        return _int_env("DEPLODOCK_PAT", _PAT_DEFAULT)
    if tile is not None:
        if _is_huge_matmul(tile):
            return _PAT_HUGE
        if _is_big_matmul(tile):
            return _PAT_BIG
    return _PAT_DEFAULT


def thread_tile_shape(tile: Tile | None = None) -> tuple[int, ...]:
    """Per-axis THREAD-tile widths ``005_blockify_launch`` should emit, in
    innermost-first order. The product equals the *pre*-register-tile
    thread-budget (``thread_budget × F²`` for matmul; ``thread_budget``
    for non-matmul where F=1).

    Matmul kernels: ``(PAT, PAT)`` — two PAT-extent thread axes for M and
    N. After ``008_register_tile`` splits each by F, the post-register-
    tile layout is ``(PAT/F, PAT/F) = (16, 16) = thread_budget``.

    With ``DEPLODOCK_TMA=1``: matmul returns the asymmetric
    ``(BN, BM)`` shape that matches the cuBLAS-beating layout from
    ``feature/mini-gpu-compiler/tma_db`` — innermost N=128 (32 threads ×
    tn=4), outer M=64 (8 threads × tm=8). Block dim becomes (32, 8),
    each warp shares one M row → A-load broadcasts (no conflict);
    threads stride consecutive N cols → B-load vectorizes to LDS.128
    (no conflict). No swizzle needed. Pairs with ``004a`` const-fold
    transpose so PyTorch ``[N, K]`` weights end up physically ``[K, N]``
    at bind time (the storage order LDS.128 vectorization requires).

    Non-matmul kernels: ``(thread_budget,)`` — a single fat thread axis
    for the innermost parallel dim (RMSNorm-row, pointwise-flat, …).
    """
    if tile is not None and _has_matmul_reduce(tile.body):
        if _tma_cublas_layout():
            return _CUBLAS_TILE_SHAPE  # (BN, BM) innermost-first
        pat = per_axis_threads(tile)
        return (pat, pat)
    return (thread_budget(),)


# cuBLAS-style tile shape: per-CTA BM=64, BN=128. After 008 register-tile
# split by F_M=8 / F_N=4: post-split THREAD axes (32, 8) = 256 threads,
# per-thread 8×4 = 32 output cells. Validated against feature/mini-gpu-
# compiler tma_db at 105% of cuBLAS on 2048² fp32 RTX 5090.
_CUBLAS_TILE_SHAPE = (128, 64)
_CUBLAS_F_PER_AXIS = (8, 4)  # (F_M, F_N) — M-axis F, N-axis F


def _tma_enabled() -> bool:
    """TMA is the cuBLAS-beating SGEMM path on sm_90+ (Hopper /
    Blackwell). Off by default — small matmuls (Linear(8, 4) test
    fixtures) hit TMA's 16-byte global-address alignment requirement
    and crash, and complex models (SDPA, RoPE) interact with the
    weight pre-transpose in ways the current passes don't fully
    handle. Set ``DEPLODOCK_TMA=1`` for nn.Linear-only models with
    ``≥1024`` per-dim shapes."""
    return os.environ.get("DEPLODOCK_TMA") == "1"


def _tma_cublas_layout() -> bool:
    return _tma_enabled()


def register_tile_shape(tile: Tile | None = None) -> tuple[int, int]:
    """Per-thread output tile as ``(F_M, F_N)``. Asymmetric in the
    cuBLAS-style mode; symmetric ``(F, F)`` otherwise."""
    if _tma_cublas_layout() and tile is not None and _has_matmul_reduce(tile.body):
        return _CUBLAS_F_PER_AXIS
    f = register_tile_factor(tile)
    return (f, f)


def detect_pat(tile: Tile) -> int | None:
    """Return the per-axis thread tile width ``005_blockify_launch``
    chose for this tile, by counting THREAD axes whose extent matches a
    known PAT candidate. Returns ``None`` pre-blockify (axes are still
    pristine output dims) or when no candidate matches."""
    from deplodock.compiler.ir.axis import BIND_THREAD

    for cand in sorted(_PAT_TO_FACTOR, reverse=True):
        if sum(1 for ba in tile.axes if ba.bind == BIND_THREAD and int(ba.axis.extent) == cand) >= 2:
            return cand
    return None


def register_tile_factor(tile: Tile | None = None) -> int:
    """Per-thread output-cell factor (per-thread tile = F × F).

    Env override (``DEPLODOCK_F``) wins. Otherwise the factor is paired
    with the PAT that ``005_blockify_launch`` produced — detected from
    the tile's THREAD axes via ``_PAT_TO_FACTOR``. Pre-blockify (no
    matching axis), the factor falls back to the big-matmul vs default
    pairing for whatever PAT ``per_axis_threads`` would pick.
    """
    raw = os.environ.get("DEPLODOCK_F")
    if raw:
        return _int_env("DEPLODOCK_F", _PAT_TO_FACTOR[_PAT_DEFAULT])
    if tile is None:
        return _PAT_TO_FACTOR[_PAT_DEFAULT]
    pat = detect_pat(tile)
    if pat is None:
        # Pre-blockify path: pair F with the PAT we'll choose at 005.
        if _is_huge_matmul(tile):
            pat = _PAT_HUGE
        elif _is_big_matmul(tile):
            pat = _PAT_BIG
        else:
            pat = _PAT_DEFAULT
    return _PAT_TO_FACTOR.get(pat, _PAT_TO_FACTOR[_PAT_DEFAULT])


def thread_budget() -> int:
    return _int_env("DEPLODOCK_TB", 256)


def forced_bk(tile: Tile | None = None) -> int | None:
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    # cuBLAS-style TMA path uses ``BM=64, BN=128`` cache extents — BK=32
    # is the only choice that keeps both per-stage smem slabs (weight
    # 128×32 = 16 KB, input 64×32 = 8 KB) under the per-stage budget
    # *and* matches what the empirical sweep showed wins at every shape
    # we tested (1024², 2048²: BK=32 < BK=16 < BK=64).
    if tile is not None and _tma_cublas_layout() and _has_matmul_reduce(tile.body):
        return 32
    if tile is not None:
        if _is_huge_matmul(tile):
            return _BK_HUGE
        if _is_big_matmul(tile):
            return _pick_big_bk(tile)
    return _BK_DEFAULT


def cooperative_block_size() -> int:
    return _int_env("DEPLODOCK_COOP_BLOCK", 256)
