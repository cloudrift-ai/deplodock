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
- ``DEPLODOCK_BK`` — K-split size (subject to ``K % BK == 0 and K > BK``).
  Default is M-adaptive.
- ``DEPLODOCK_SPLITK`` — force a cross-CTA split-K factor (>0 wins).
- ``DEPLODOCK_TMA_SWIZZLE`` — opt in to TMA hardware-swizzle modes
  (``SWIZZLE_{128,64,32}B``). ``=1`` enables; default off. Stages whose
  inner box-dim byte size matches a swizzle width pick the matching
  mode in ``050_use_tma``; the materializer pairs each swizzled stage
  with body-Load XOR decoding and a 1024-byte (B128) / 512-byte (B64)
  / 256-byte (B32) smem alignment so the swizzle pattern lines up with
  the buffer base.

## API

Heuristics are pure numeric functions: they take ``output_extents``
(logical, sorted descending) and a :class:`BodyInfo` summary computed
once per LoopOp at the start of the planner. Tile-taking callers
compute these inputs themselves and call the heuristics directly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.body import Body


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

# Reserve ~4 KB of headroom under the static-smem cap so 070_pad_smem's
# ``+1`` per-stage padding (to break 32-way bank conflicts) doesn't
# push the kernel over.
_PAD_HEADROOM_BYTES = 4 * 1024
_DTYPE_BYTES = 4

# CTA inner-axis width default for paths that produce a single THREAD
# axis (non-matmul kernels). Mutually exclusive with the matmul path,
# so it shares the ``DEPLODOCK_BN`` env namespace.
_NON_MATMUL_BN_DEFAULT = 256

# Cross-CTA split-K target.
_SPLITK_TARGET_WAVES_HIGH = 16
_SPLITK_TARGET_WAVES_LARGE = 8
_SPLITK_TARGET_WAVES_SMALL = 2
_SPLITK_NUM_SMS = 170  # RTX 5090; conservative upper bound for sm_120


# --- Helpers ------------------------------------------------------------


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- BodyInfo -----------------------------------------------------------


@dataclass(frozen=True)
class BodyInfo:
    """Structural summary of a LoopOp / Tile body — what the heuristics
    need to know that doesn't depend on axis extents.

    Computed once per LoopOp at the start of the planner's ``rewrite``
    and reused across every heuristic call as the body undergoes
    σ-substitution. None of these flags change under axis splits or
    σ-rewriting: matmul shape, fused-prologue load count, and external
    input count are all about the body's def-use shape, not its axis
    iteration extents.
    """

    has_matmul: bool
    has_fused_prologue: bool
    external_input_count: int

    @classmethod
    def of(cls, body) -> BodyInfo:
        return cls(
            has_matmul=_has_matmul_reduce(body),
            has_fused_prologue=_has_fused_prologue(body),
            external_input_count=_external_input_count(body),
        )


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
    doesn't push register pressure inside it."""
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
    gated on ``count == 2``."""
    from deplodock.compiler.ir.stmt import Load
    from deplodock.compiler.ir.stmt.body import Body
    from deplodock.compiler.ir.tile.ir import Stage, StageBundle

    body = Body.coerce(stmts)
    # Collect smem decl names so we skip Loads that read from staged smem
    # (not external gmem). Both Stage members (yielded inside a bundle's
    # synthetic body) and the StageBundle itself expose ``local_decls()``.
    stage_smem_names: set[str] = set()
    for s in body.iter():
        if isinstance(s, (Stage, StageBundle)):
            stage_smem_names.update(s.local_decls())
    bufs: set[str] = set()
    for s in body.iter():
        if isinstance(s, (Stage, StageBundle)):
            # ``external_reads()`` returns gmem buffer names; Stage members
            # with ``compute != None`` (hoisted-invariant cooperative
            # compute) override to ``()`` because they read sibling smem.
            for buf_name in s.external_reads():
                bufs.add(buf_name)
        elif isinstance(s, Load) and s.input not in stage_smem_names:
            bufs.add(s.input)
    return len(bufs)


# --- Extent recovery ----------------------------------------------------


_SUB_AXIS_SUFFIX = re.compile(r"_(o|i|reg|b|t|r|s)$")


def recover_logical_extents(body: Body) -> tuple[int, ...]:
    """Recover the pre-split logical output extents from a LoopOp /
    Tile body. Walks the outer free-Loop chain and folds adjacent
    σ-split sub-axes (``_o`` / ``_i`` / ``_reg`` / ``_b`` / ``_t`` /
    ``_r`` / ``_s``) sharing the same parent name back into a single
    extent. Returns extents sorted descending so callers can read
    ``extents[0]`` as N (largest) and ``extents[1]`` as M.

    Used after the planner has σ-split output axes so heuristics still
    see logical extents."""
    from deplodock.compiler.ir.stmt import Loop

    chain_extents: list[tuple[str, int]] = []
    cur = tuple(body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        chain_extents.append((cur[0].axis.name, cur[0].axis.extent.as_static()))
        cur = tuple(cur[0].body)

    folded: dict[str, int] = {}
    order: list[str] = []
    for name, ext in chain_extents:
        parent = _SUB_AXIS_SUFFIX.sub("", name)
        if parent in folded:
            folded[parent] *= ext
        else:
            folded[parent] = ext
            order.append(parent)
    return tuple(sorted((folded[p] for p in order), reverse=True))


# --- Class & default-tile pickers --------------------------------------


def _tile_class(output_extents: tuple[int, ...], body_info: BodyInfo) -> str:
    """``"fused" | "huge" | "compact" | "default-large" | "default"``
    — picks the matmul tile class from the logical output extents and
    body structure. Non-matmul bodies fall to ``"default"`` (the env
    vars wouldn't apply anyway).

    ``"fused"`` is checked first because a fused-prologue matmul
    (``silu_mul_matmul``) can have output extents that would otherwise
    classify it as ``huge`` / ``default``, but its register pressure
    profile is fundamentally different.

    ``"default-large"`` is the default-class subclass for M ≥ 128 +
    N > _COMPACT_N_MAX + N < _HUGE_N_MIN. FN=8 (vs FN=4) doubles
    arithmetic intensity per thread; the chunk pass (``009``) keeps
    LDS.128 vectorized."""
    if not body_info.has_matmul:
        return "default"
    if body_info.has_fused_prologue:
        return "fused"
    if len(output_extents) < 2:
        return "default"
    n, m = output_extents[0], output_extents[1]
    if m >= _HUGE_M_MIN and n >= _HUGE_N_MIN:
        return "huge"
    if n <= _COMPACT_N_MAX:
        return "compact"
    if m >= _DEFAULT_LARGE_M_MIN and body_info.external_input_count == 2:
        return "default-large"
    return "default"


def _default_tile(output_extents: tuple[int, ...], body_info: BodyInfo) -> tuple[tuple[int, int], tuple[int, int]]:
    """``((BN, BM), (FM, FN))`` defaults for the matmul tile, picked
    by :func:`_tile_class`."""
    cls = _tile_class(output_extents, body_info)
    if cls == "fused":
        return _TILE_SHAPE_FUSED, _F_PER_AXIS_FUSED
    if cls == "huge":
        return _TILE_SHAPE_HUGE, _F_PER_AXIS_HUGE
    if cls == "compact":
        return _TILE_SHAPE_COMPACT, _F_PER_AXIS_COMPACT
    if cls == "default-large":
        return _TILE_SHAPE_DEFAULT_LARGE, _F_PER_AXIS_DEFAULT_LARGE
    return _TILE_SHAPE_DEFAULT, _F_PER_AXIS_DEFAULT


# --- TMA gates ---------------------------------------------------------


def _tma_swizzle_enabled() -> bool:
    """TMA hardware-swizzle gate. Off by default — only fires when the
    inner box-dim byte size matches a swizzle width AND ``DEPLODOCK_TMA_SWIZZLE``
    opts in."""
    return os.environ.get("DEPLODOCK_TMA_SWIZZLE", "0") in ("1", "true", "True")


# --- Heuristics (pure numeric) -----------------------------------------


def thread_tile_shape(output_extents: tuple[int, ...], body_info: BodyInfo) -> tuple[int, ...]:
    """Per-axis THREAD-tile widths the launch-geometry step should emit,
    innermost-first. ``(BN, BM)`` for matmul, ``(BN,)`` for non-matmul
    kernels (single THREAD axis; same env namespace as the matmul case)."""
    if body_info.has_matmul:
        (def_bn, def_bm), _ = _default_tile(output_extents, body_info)
        bn = _int_env("DEPLODOCK_BN", def_bn)
        bm = _int_env("DEPLODOCK_BM", def_bm)
        return (bn, bm)
    return (_int_env("DEPLODOCK_BN", _NON_MATMUL_BN_DEFAULT),)


def register_tile_shape(
    output_extents: tuple[int, ...],
    thread_extents: tuple[int, ...],
    body_info: BodyInfo,
) -> tuple[int, int]:
    """Per-thread output tile ``(FM, FN)``. ``(1, 1)`` to skip register
    tiling on non-matmul bodies and on tiny matmuls whose THREAD product
    is already at-or-below one warp.

    For tiles whose THREAD extents match the class's default (BN, BM),
    return the class-tuned ``(FM, FN)``. For off-default extents
    derive a heuristic F from the actual thread extents that targets
    ~256 post-split threads."""
    if not body_info.has_matmul:
        return (1, 1)
    (def_bn, def_bm), (def_fm, def_fn) = _default_tile(output_extents, body_info)
    f_m = _int_env("DEPLODOCK_FM", def_fm)
    f_n = _int_env("DEPLODOCK_FN", def_fn)
    bn = _int_env("DEPLODOCK_BN", def_bn)
    bm = _int_env("DEPLODOCK_BM", def_bm)
    if not thread_extents:
        return (f_m, f_n)
    te_set = {int(e) for e in thread_extents}
    if te_set & {bn, bm}:
        return (f_m, f_n)
    sorted_te = sorted(int(e) for e in thread_extents)
    m_ext = sorted_te[0]
    n_ext = sorted_te[-1]
    cur_threads = m_ext * n_ext
    if cur_threads <= 32:
        return (1, 1)
    for f in (8, 4, 2):
        if m_ext % f == 0 and n_ext % f == 0 and (m_ext // f) * (n_ext // f) >= 32:
            return (f, f)
    return (1, 1)


def forced_bk(
    output_extents: tuple[int, ...],
    body_info: BodyInfo,
    static_smem_cap: int | None = None,
) -> int | None:
    """Force BK via env, or pick the M-adaptive default. Backs off to a
    smaller BK when the M-adaptive choice would overflow the static-
    smem cap at the active tile shape.

    ``static_smem_cap`` defaults to ``Context.static_smem_cap`` (48 KB)."""
    raw = os.environ.get("DEPLODOCK_BK")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if not body_info.has_matmul:
        return None
    m = output_extents[1] if len(output_extents) >= 2 else 0
    from deplodock.compiler.target import compute_capability  # noqa: PLC0415

    tma_path = compute_capability() >= (9, 0)
    bk = _BK_SMALL_M if m <= _M_THRESHOLD else (_BK_LARGE_M_TMA if tma_path else _BK_LARGE_M_DEFAULT)
    if static_smem_cap is None:
        from deplodock.compiler.context import STATIC_SMEM_CAP  # noqa: PLC0415

        static_smem_cap = STATIC_SMEM_CAP
    return _bk_fits_smem(output_extents, body_info, bk, static_smem_cap)


def _bk_fits_smem(output_extents: tuple[int, ...], body_info: BodyInfo, bk: int, static_smem_cap: int) -> int:
    """Halve BK until the per-stage smem fits in ``static_smem_cap``
    minus pad headroom. Returns the largest BK ≥ 1 that fits.

    Stage footprint = ``n_inputs · max(BN, BM) · BK · 4 · 2``."""
    budget = static_smem_cap - _PAD_HEADROOM_BYTES
    if len(output_extents) < 2:
        return bk
    (def_bn, def_bm), _ = _default_tile(output_extents, body_info)
    bn_actual = min(output_extents[0], _int_env("DEPLODOCK_BN", def_bn))
    bm_actual = min(output_extents[1], _int_env("DEPLODOCK_BM", def_bm))
    n_inputs = max(body_info.external_input_count, 2)
    stage_footprint = n_inputs * max(bn_actual, bm_actual) * _DTYPE_BYTES * 2
    while bk > 1 and stage_footprint * bk > budget:
        bk //= 2
    return bk


def auto_splitk(
    output_extents: tuple[int, ...],
    body_info: BodyInfo,
    k_o_extent: int,
    thread_extents: tuple[int, ...],
) -> int:
    """Auto-pick a cross-CTA split-K factor.

    ``DEPLODOCK_SPLITK`` env wins when set. Otherwise: target
    ``waves_target * num_sms`` total CTAs, divide by the current
    M-N grid count, clamp to the largest divisor of ``k_o_extent``
    that is ≤ the target. Returns 1 when no useful split exists."""
    forced = _int_env("DEPLODOCK_SPLITK", 0)
    if forced > 0:
        return forced
    if not body_info.has_matmul:
        return 1
    (def_bn, def_bm), _ = _default_tile(output_extents, body_info)
    bn = _int_env("DEPLODOCK_BN", def_bn)
    bm = _int_env("DEPLODOCK_BM", def_bm)
    targets = (bn, bm)
    # Pre-blockify thread extents — same shape as ``tile.axes`` extents
    # in the legacy synthetic-Tile call. Sort descending; innermost-2
    # axes become the matmul tile dims and map to (BN, BM).
    extents = sorted((int(e) for e in thread_extents), reverse=True)
    grid = 1
    for i, ext in enumerate(extents):
        if i < len(targets):
            tgt = targets[i]
            grid *= max(1, ext // tgt) if ext >= tgt else 1
        else:
            grid *= ext
    target_total = _splitk_target_waves(output_extents, body_info) * _SPLITK_NUM_SMS
    if grid >= target_total:
        return 1
    desired = max(1, target_total // grid)
    splitk = max((d for d in range(1, min(desired, k_o_extent) + 1) if k_o_extent % d == 0), default=1)
    return splitk


def _splitk_target_waves(output_extents: tuple[int, ...], body_info: BodyInfo) -> int:
    cls = _tile_class(output_extents, body_info)
    n = output_extents[0] if output_extents else 0
    m = output_extents[1] if len(output_extents) >= 2 else 0
    if 0 < m <= _LOW_GRID_M_MAX and 0 < n <= _LOW_GRID_N_MAX:
        return _SPLITK_TARGET_WAVES_SMALL
    if cls == "compact":
        return _SPLITK_TARGET_WAVES_SMALL
    if cls == "default" and m >= _WIDE_GRID_M_MIN and n >= _WIDE_GRID_N_MIN:
        return _SPLITK_TARGET_WAVES_HIGH
    return _SPLITK_TARGET_WAVES_LARGE


def cooperative_block_size() -> int:
    """Threads/CTA for synthetic-thread axes in non-matmul paths.
    Shares the matmul ``DEPLODOCK_BN`` env namespace."""
    return _int_env("DEPLODOCK_BN", _NON_MATMUL_BN_DEFAULT)
