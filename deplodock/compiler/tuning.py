"""Compile-time tuning knobs — heuristic defaults + env-var overrides.

Three-class matmul tile, picked from the logical output extents:

- **huge** (``M ≥ 256 AND N ≥ 8192``) — Qwen gate/up_proj.s512-class
  GEMMs. ``(BN=128, BM=128, FM=16, FN=4)`` → 32 outputs / thread,
  256 threads / CTA. Splitk waves target = 8.
- **compact** (``N ≤ 1024``) — kv_proj-class and SDPA-shaped matmuls
  (small head_dim K, M=seq). ``(BN=64, BM=64, FM=8, FN=4)`` → 16
  outputs / thread, 128 threads / CTA. Splitk waves target = 2.
- **default** (everything else) — Q/O/down/MLP-up proj-class. Original
  asymmetric cuBLAS-style ``(BN=128, BM=64, FM=8, FN=4)`` validated
  at 105% of cuBLAS on 2048² fp32 RTX 5090. Splitk waves target = 8.

Orthogonal splitk-waves overrides (tile shape unchanged):

- **low-grid** (``M ≤ 128 AND N ≤ 2048``): waves=2. The M·N grid is too
  small (~32 CTAs) for the 8-wave target — splitk inflates and atomic
  contention dominates.
- **wide-grid** (default class, ``M ≥ 256 AND N ≥ 4096``): waves=16.
  Big enough to absorb extra split-K parallelism without contention.

Sweep on RTX 5090 fp32: this 3-class split cuts kv_proj.s512 ~48µs →
~20µs, kv_proj.s128 ~30µs → ~13µs, sdpa.tinyllama.s512 ~456µs → ~178µs,
gate_proj.s512 (qwen) ~1430µs → ~1320µs. Env vars override per-axis.

Env vars:

- ``DEPLODOCK_BN``, ``DEPLODOCK_BM`` — per-CTA tile (innermost N, outer M).
- ``DEPLODOCK_FN``, ``DEPLODOCK_FM`` — per-thread output cells.
- ``DEPLODOCK_BK`` — K-split size for ``002_chunk_matmul_k`` (subject
  to ``K % BK == 0 and K > BK``). Default is M-adaptive.
- ``DEPLODOCK_SPLITK`` — force a cross-CTA split-K factor (>0 wins).
- ``DEPLODOCK_COOP_BLOCK`` — cooperative-reduce thread count.
- ``DEPLODOCK_TMA`` — emit ``cp.async.bulk.tensor`` (TMA) loads + runtime
  weight transpose (``004a_fold_into_constant``). Default-on for sm_90+
  (Hopper / Blackwell), default-off below. ``=1`` forces on, ``=0``
  forces off. ``011_tma_copy`` gates eligibility on rank ≤ 5,
  ConstantOp source with a recorded transpose load_op chain, and
  source-inner-extent alignment ≥ 16 B with ≥ 2× headroom past the
  box inner extent.
- ``DEPLODOCK_TMA_SWIZZLE`` — opt in to TMA hardware-swizzle modes
  (``SWIZZLE_{128,64,32}B``). ``=1`` enables; default off. Stages whose
  inner box-dim byte size matches a swizzle width pick the matching
  mode in ``011_tma_copy``; the materializer pairs each swizzled stage
  with body-Load XOR decoding and a 1024-byte (B128) / 512-byte (B64)
  / 256-byte (B32) smem alignment so the swizzle pattern lines up with
  the buffer base.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Tile


# Per-CTA matmul tile defaults. Three classes, picked from logical output
# extents (sweep on RTX 5090 fp32):
#
# - "huge" (``M ≥ _HUGE_M_MIN AND N ≥ _HUGE_N_MIN``): big GEMMs like
#   ``M=512, N=18944`` (Qwen gate/up_proj.s512). ``(BN=128, BM=128,
#   FM=16, FN=4) → 32 outputs/thread, 256 threads/CTA`` beats the
#   default tile by ~8%.
#
# - "compact" (``N ≤ _COMPACT_N_MAX``): kv_proj-class (narrow N) and
#   SDPA-shaped matmuls (small head_dim K, M=seq). ``(BN=64, BM=64,
#   FM=8, FN=4)`` + ``waves=2`` raises grid count and tames atomic-add
#   contention. Cuts kv_proj.s512 ~48µs → ~20µs and sdpa.tinyllama.s512
#   ~456µs → ~178µs.
#
# - "default" (everything else): proj-class GEMMs (Q/O/down/MLP up). The
#   original cuBLAS-style asymmetric ``(BN=128, BM=64, FM=8, FN=4)``
#   tile, validated at ~105% of cuBLAS on 2048² fp32.
_TILE_SHAPE_DEFAULT = (128, 64)  # (BN, BM)
_F_PER_AXIS_DEFAULT = (8, 4)  # (FM, FN)
_TILE_SHAPE_HUGE = (128, 128)
_F_PER_AXIS_HUGE = (16, 4)
_TILE_SHAPE_COMPACT = (64, 64)
_F_PER_AXIS_COMPACT = (8, 4)
# "Fused-prologue" class: the matmul-reduce body has a third buffer
# being loaded inside the K-loop (e.g. ``silu_mul_matmul`` reads g, u,
# *and* w each K-iter to compute the fused ``silu(g)*u`` operand). The
# extra Load chain plus the per-K-iter prologue values eat registers
# the matmul's F=8x4 accumulator tile would normally use, dropping
# occupancy from the register-file headroom. Sweep on RTX 5090 fp32 +
# silu_mul_matmul.qwen.{s128,s512} cuts the case from 0.17× → 0.37×
# (s128, 2039µs → 933µs) and 0.12× → 0.28× (s512, 9270µs → 3920µs).
_TILE_SHAPE_FUSED = (64, 64)
_F_PER_AXIS_FUSED = (4, 4)
# "Default-large" class: default-class shapes with enough M to fill
# the grid benefit from doubling the per-thread N register tile from
# FN=4 → FN=8. The chunk pass (``009``) keeps LDS.128 vectorized
# and prevents the FN=8 bank-conflict cliff. Sweep on RTX 5090 fp32:
# qwen.q_proj.s512 0.87× → 0.92× (286 → 266µs), qwen.down_proj.s512
# 0.79× → 0.86× (1434 → 1331µs), tl.gate_proj.s512 0.95× → 0.99×.
# Small-M cases (s32) regress because the grid is too small to
# amortize the extra register pressure (e.g. tl.q_proj.s32 1.47×
# → 1.32×); the threshold ``_DEFAULT_LARGE_M_MIN = 128`` gates the
# upgrade to cases where it actually wins.
_TILE_SHAPE_DEFAULT_LARGE = (128, 64)
_F_PER_AXIS_DEFAULT_LARGE = (8, 8)
_DEFAULT_LARGE_M_MIN = 128
_HUGE_M_MIN = 256
_HUGE_N_MIN = 8192
_COMPACT_N_MAX = 1024
# "Low-grid" cap: when both ``M ≤ _LOW_GRID_M_MAX`` and ``N ≤
# _LOW_GRID_N_MAX``, the M·N grid stays small (~32 CTAs at default tile)
# — auto-splitk's 8-wave target inflates splitk to ~32 and atomic-add
# contention dominates. Drop waves to 2. Sweep on RTX 5090 fp32 (cuts in
# µs vs default): tinyllama.o_proj.s32 33→14, q_proj.s32 33→14,
# o_proj.s128 53→31, q_proj.s128 53→31, qwen.{o,q}_proj.s32 42→37.
_LOW_GRID_M_MAX = 128
_LOW_GRID_N_MAX = 2048
# "Wide-grid" floor: default-class GEMMs with ``M ≥ _WIDE_GRID_M_MIN``
# and ``N ≥ _WIDE_GRID_N_MIN`` (but below the huge threshold, ``N ≥
# _HUGE_N_MIN``) have enough M·N CTAs that more split-K parallelism is a
# net win — bumping waves to 16 cuts tinyllama.{gate,up}_proj.s512
# ~245µs → ~225µs. Picked symmetric to ``_HUGE_N_MIN/2`` to avoid
# straddling the huge-class boundary.
_WIDE_GRID_M_MIN = 256
_WIDE_GRID_N_MIN = 4096

# Per-stage K-tile, M-adaptive. Sweep on TinyLlama Q/Gate/Down at seq ∈
# {32, 128, 512} on RTX 5090 fp32: M ≤ 256 → BK=64 wins (small grid, K
# loop must amortize CTA setup); M > 256 → BK=16 (cp.async) or BK=32
# (TMA) — larger BK at M=512 catastrophically slow (smem overflow).
_M_THRESHOLD = 256
_BK_SMALL_M = 64
_BK_LARGE_M_DEFAULT = 16  # cp.async path
_BK_LARGE_M_TMA = 32  # TMA path


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


def _has_fused_prologue(stmts) -> bool:
    """True iff some matmul-reduce loop's body has ``≥3`` distinct
    buffer Loads — i.e. a fused prologue feeds extra operand values
    into the reduction (``silu_mul_matmul``: ``g_smem, u_smem,
    w_smem``). Pure ``matmul`` and ``matmul_add`` (epilogue-fused)
    have exactly 2 buffer Loads inside the K-reduce; the residual /
    bias of ``matmul_add`` is loaded *outside* the reduce loop and
    doesn't push register pressure inside it.

    The extra buffer Load is a structural proxy for "operand values
    held across K iters that compete with the F-tile accumulators for
    register space" — exactly the case where the canonical F=8×4
    tile starves occupancy. ``_default_tile`` switches to F=4×4 BN=64
    for these.
    """
    from deplodock.compiler.ir.stmt import Load, Loop
    from deplodock.compiler.ir.stmt.body import Body

    for s in Body.coerce(stmts).iter():
        if isinstance(s, Loop) and s.is_reduce:
            bufs = {ld.input for ld in s.body.of_type(Load)}
            if len(bufs) >= 3:
                return True
    return False


def _external_input_count(stmts) -> int:
    """Distinct external buffers a Tile body references — sum of
    ``Stage.buf`` (staged inputs) plus any direct ``Load.input`` that
    doesn't name a Stage.

    Pure ``matmul`` = 2 (a, b). ``matmul_add`` = 3 (a, b, residual).
    ``silu_mul_matmul`` = 3 (g, u, w). The default-large class is
    gated on ``count == 2`` to keep extra epilogue Loads (which steal
    register slots per-output-cell) from forcing F=8×8 — sweep showed
    ``matmul_add.tinyllama.o_proj.s512`` regressing 0.80× → 0.72×
    when the residual Load piles on the F=8×8 accumulator tile
    (regs 118 → 209, occ 33% → 17%).
    """
    from deplodock.compiler.ir.stmt import Load
    from deplodock.compiler.ir.stmt.body import Body
    from deplodock.compiler.ir.tile.ir import Stage

    body = Body.coerce(stmts)
    stage_names = {s.name for s in body.iter() if isinstance(s, Stage)}
    bufs: set[str] = set()
    for s in body.iter():
        if isinstance(s, Stage):
            # Fused stages load from multiple gmem buffers (silu-mul → gate, up);
            # count each unique source so the heuristic sees the right input count.
            for src_load in s.source_loads:
                bufs.add(src_load.input)
        elif isinstance(s, Load) and s.input not in stage_names:
            bufs.add(s.input)
    return len(bufs)


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


def _tile_class(tile: Tile | None) -> str:
    """``"fused" | "huge" | "compact" | "default-large" | "default"``
    — picks the matmul tile class from the logical output extents and
    body structure. Non-matmul tiles fall to ``"default"`` (the env
    vars wouldn't apply anyway).

    ``"fused"`` is checked first because a fused-prologue matmul
    (``silu_mul_matmul``) can have output extents that would otherwise
    classify it as ``huge`` / ``default``, but its register pressure
    profile is fundamentally different — the standard tiles starve
    occupancy.

    ``"default-large"`` is the default-class subclass for M ≥ 128 +
    N > _COMPACT_N_MAX + N < _HUGE_N_MIN. FN=8 (vs FN=4) doubles
    arithmetic intensity per thread; the chunk pass (``009``) keeps
    LDS.128 vectorized.
    """
    if tile is None or not _has_matmul_reduce(tile.body):
        return "default"
    if _has_fused_prologue(tile.body):
        return "fused"
    extents = _logical_output_extents(tile)
    if len(extents) < 2:
        return "default"
    n, m = extents[0], extents[1]
    if m >= _HUGE_M_MIN and n >= _HUGE_N_MIN:
        return "huge"
    if n <= _COMPACT_N_MAX:
        return "compact"
    if m >= _DEFAULT_LARGE_M_MIN and _external_input_count(tile.body) == 2:
        return "default-large"
    return "default"


def _default_tile(tile: Tile | None) -> tuple[tuple[int, int], tuple[int, int]]:
    """``((BN, BM), (FM, FN))`` defaults for the matmul tile, picked
    by ``_tile_class``."""
    cls = _tile_class(tile)
    if cls == "fused":
        return _TILE_SHAPE_FUSED, _F_PER_AXIS_FUSED
    if cls == "huge":
        return _TILE_SHAPE_HUGE, _F_PER_AXIS_HUGE
    if cls == "compact":
        return _TILE_SHAPE_COMPACT, _F_PER_AXIS_COMPACT
    if cls == "default-large":
        return _TILE_SHAPE_DEFAULT_LARGE, _F_PER_AXIS_DEFAULT_LARGE
    return _TILE_SHAPE_DEFAULT, _F_PER_AXIS_DEFAULT


# --- Public API ---------------------------------------------------------


def _tma_enabled() -> bool:
    """TMA staging gate. Default-on for sm_90+ (Hopper / Blackwell) which
    have ``cp.async.bulk.tensor``; default-off below sm_90. ``DEPLODOCK_TMA``
    overrides either way: ``=1`` forces on, ``=0`` forces off."""
    raw = os.environ.get("DEPLODOCK_TMA")
    if raw == "1":
        return True
    if raw == "0":
        return False
    from deplodock.compiler.pipeline.passes.lowering.tile._helpers import compute_capability  # noqa: PLC0415

    return compute_capability() >= (9, 0)


def _tma_swizzle_enabled() -> bool:
    """TMA hardware-swizzle gate. Off by default — only fires when the
    inner box-dim byte size matches a swizzle width AND ``DEPLODOCK_TMA_SWIZZLE``
    opts in. The materializer pairs swizzle with body-Load XOR decoding +
    1024-byte smem alignment for ``B128`` (proportionally smaller for
    ``B64`` / ``B32``); a current default-on would regress kernels whose
    slab geometry happens to fit a swizzle width but whose body Loads
    weren't validated against the decoder."""
    return os.environ.get("DEPLODOCK_TMA_SWIZZLE", "0") in ("1", "true", "True")


_NON_MATMUL_THREAD_BUDGET = 256


def thread_tile_shape(tile: Tile | None = None) -> tuple[int, ...]:
    """Per-axis THREAD-tile widths ``005_blockify_launch`` should emit,
    innermost-first. ``(BN, BM)`` for matmul, ``(256,)`` for non-matmul
    kernels."""
    if tile is not None and _has_matmul_reduce(tile.body):
        (def_bn, def_bm), _ = _default_tile(tile)
        bn = _int_env("DEPLODOCK_BN", def_bn)
        bm = _int_env("DEPLODOCK_BM", def_bm)
        return (bn, bm)
    return (_NON_MATMUL_THREAD_BUDGET,)


def register_tile_shape(tile: Tile | None = None) -> tuple[int, int]:
    """Per-thread output tile ``(FM, FN)``. ``(1, 1)`` to skip
    register tiling on non-matmul bodies and on tiny matmuls whose
    post-blockify THREAD product is already at-or-below one warp.

    For tiles whose THREAD extents match the class's default (BN, BM),
    return the class-tuned ``(FM, FN)``. For *off-default* (BN, BM)
    — the autotune sweep over ``005_blockify_launch``'s ``_TUNE_AXIS_CHOICES``
    — derive a heuristic F from the actual thread extents that targets
    ~256 post-split threads, so 008 still forks (and the autotuner can
    explore its F neighborhood) instead of bailing to ``(1, 1)`` and
    leaving small-tile variants register-tile-less.
    """
    if tile is None or not _has_matmul_reduce(tile.body):
        return (1, 1)
    (def_bn, def_bm), (def_fm, def_fn) = _default_tile(tile)
    f_m = _int_env("DEPLODOCK_FM", def_fm)
    f_n = _int_env("DEPLODOCK_FN", def_fn)
    bn = _int_env("DEPLODOCK_BN", def_bn)
    bm = _int_env("DEPLODOCK_BM", def_bm)
    from deplodock.compiler.ir.axis import BIND_THREAD

    thread_axes = [ba for ba in tile.axes if ba.bind == BIND_THREAD]
    if not thread_axes:
        return (f_m, f_n)
    thread_extents = {int(ba.axis.extent) for ba in thread_axes}
    if thread_extents & {bn, bm}:
        return (f_m, f_n)

    # Off-default (BN, BM) — pick the largest symmetric F (power of 2)
    # that divides both extents and leaves at least one warp's worth of
    # threads post-split. The autotune fork over ``_TUNE_F_CHOICES``
    # then explores neighbours; this just has to be non-(1, 1) so 008
    # fires and emits the fork.
    sorted_ba = sorted(thread_axes, key=lambda ba: int(ba.axis.extent))
    m_ext = int(sorted_ba[0].axis.extent)
    n_ext = int(sorted_ba[-1].axis.extent)
    cur_threads = m_ext * n_ext
    if cur_threads <= 32:
        return (1, 1)
    for f in (8, 4, 2):
        if m_ext % f == 0 and n_ext % f == 0 and (m_ext // f) * (n_ext // f) >= 32:
            return (f, f)
    return (1, 1)


def forced_bk(tile: Tile | None = None, static_smem_cap: int | None = None) -> int | None:
    """Force BK via env, or pick the M-adaptive default. Backs off to
    a smaller BK when the M-adaptive choice would overflow the static-
    smem cap at the active tile shape — happens for small matmuls where
    the THREAD axes stay at the full output extent and BK_SMALL_M=64
    produces a stage too large for two buffers.

    ``static_smem_cap`` defaults to ``Context.static_smem_cap`` (48 KB);
    callers that have a ``Context`` should pass ``ctx.static_smem_cap``
    so the budget tracks the target arch."""
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if tile is None or not _has_matmul_reduce(tile.body):
        return None
    bk = _BK_SMALL_M if _matmul_M(tile) <= _M_THRESHOLD else (_BK_LARGE_M_TMA if _tma_enabled() else _BK_LARGE_M_DEFAULT)
    if static_smem_cap is None:
        from deplodock.compiler.context import STATIC_SMEM_CAP  # noqa: PLC0415

        static_smem_cap = STATIC_SMEM_CAP
    return _bk_fits_smem(tile, bk, static_smem_cap)


# Reserve ~4 KB of headroom under the static-smem cap so 014_pad_smem's
# ``+1`` per-stage padding (to break 32-way bank conflicts) doesn't
# push the kernel over.
_PAD_HEADROOM_BYTES = 4 * 1024
_DTYPE_BYTES = 4


def _bk_fits_smem(tile: Tile, bk: int, static_smem_cap: int) -> int:
    """Halve BK until the per-stage smem fits in ``static_smem_cap``
    minus pad headroom. Returns the largest BK ≥ 1 that fits.

    Stage footprint = ``n_inputs · max(BN, BM) · BK · 4 · 2``: each
    external input gets staged at roughly ``tile × BK`` floats, with
    two buffers for double-buffering. ``n_inputs`` is read from the
    tile body — a plain ``matmul`` has 2 inputs (A, B); fused-prologue
    matmuls (``silu_mul_matmul``, naive attention's ``softmax·mask @ V``)
    have ≥3 and need a tighter BK to leave room for the extra stage.
    The previous 2-input fixed assumption let naive_attn at head_dim=128
    pick a BK that overflowed the dynamic-smem cap at launch.
    """
    budget = static_smem_cap - _PAD_HEADROOM_BYTES
    extents = _logical_output_extents(tile)
    if len(extents) < 2:
        return bk
    (def_bn, def_bm), _ = _default_tile(tile)
    bn_actual = min(extents[0], _int_env("DEPLODOCK_BN", def_bn))
    bm_actual = min(extents[1], _int_env("DEPLODOCK_BM", def_bm))
    n_inputs = max(_external_input_count(tile.body), 2)
    stage_footprint = n_inputs * max(bn_actual, bm_actual) * _DTYPE_BYTES * 2
    while bk > 1 and stage_footprint * bk > budget:
        bk //= 2
    return bk


def cooperative_block_size() -> int:
    return _int_env("DEPLODOCK_COOP_BLOCK", 256)


# Cross-CTA split-K target. Class-adaptive: small-tile matmuls (narrow N
# kv_proj-class, small-M s32, SDPA-shaped) win with a low waves target —
# their grids already cover most of the SM array, so chasing 8 waves
# inflates split-K and atomic-add contention dominates. Sweep on RTX 5090
# fp32: ``waves=2`` cuts kv_proj.s512 48µs → 20µs and sdpa.qwen.s128 85µs
# → 46µs, but regresses gate_proj.s512 — large-large GEMMs need ``waves=8``
# (the prior comment's ``{4, 8, 16}`` empirical elbow on TinyLlama MLP).
_SPLITK_TARGET_WAVES_HIGH = 16
_SPLITK_TARGET_WAVES_LARGE = 8
_SPLITK_TARGET_WAVES_SMALL = 2
_SPLITK_NUM_SMS = 170  # RTX 5090; conservative upper bound for sm_120


def _splitk_target_waves(tile: Tile | None) -> int:
    cls = _tile_class(tile)
    if tile is not None and _has_matmul_reduce(tile.body):
        extents = _logical_output_extents(tile)
        n = extents[0] if extents else 0
        m = extents[1] if len(extents) >= 2 else 0
    else:
        n = m = 0
    if 0 < m <= _LOW_GRID_M_MAX and 0 < n <= _LOW_GRID_N_MAX:
        return _SPLITK_TARGET_WAVES_SMALL
    if cls == "compact":
        return _SPLITK_TARGET_WAVES_SMALL
    if cls == "default" and m >= _WIDE_GRID_M_MIN and n >= _WIDE_GRID_N_MIN:
        return _SPLITK_TARGET_WAVES_HIGH
    return _SPLITK_TARGET_WAVES_LARGE


def auto_splitk(tile: Tile, k_o_extent: int) -> int:
    """Auto-pick a cross-CTA split-K factor for the given matmul Tile.

    ``DEPLODOCK_SPLITK`` env wins when set. Otherwise: target
    ``waves_target * num_sms`` total CTAs, divide by the current
    M-N grid count, clamp to the largest divisor of ``k_o_extent``
    that is ≤ the target. Returns 1 when no useful split exists."""
    forced = _int_env("DEPLODOCK_SPLITK", 0)
    if forced > 0:
        return forced
    if not _has_matmul_reduce(tile.body):
        return 1
    (def_bn, def_bm), _ = _default_tile(tile)
    bn = _int_env("DEPLODOCK_BN", def_bn)
    bm = _int_env("DEPLODOCK_BM", def_bm)
    targets = (bn, bm)
    grid = 1
    extents = sorted([int(ba.axis.extent) for ba in tile.axes], reverse=True)
    # Innermost-2 axes become the matmul tile dims; map them to (BN, BM).
    # Outer axes go BLOCK whole (multiplying grid by their extents).
    for i, ext in enumerate(extents):
        if i < len(targets):
            tgt = targets[i]
            grid *= max(1, ext // tgt) if ext >= tgt else 1
        else:
            grid *= ext
    target_total = _splitk_target_waves(tile) * _SPLITK_NUM_SMS
    if grid >= target_total:
        return 1
    desired = max(1, target_total // grid)
    # Largest divisor of k_o_extent that is ≤ desired.
    splitk = max((d for d in range(1, min(desired, k_o_extent) + 1) if k_o_extent % d == 0), default=1)
    return splitk
