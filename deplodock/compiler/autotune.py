"""Analytical tile-shape autotuner for SIMT FP32 matmul kernels.

The bank-conflict count for a given ``(BM, BN, F_M, F_N, BK)`` tile is
deterministic from the access pattern: for each smem load, enumerate
the 32 warp lanes, decode each lane's ``(m_thread, n_thread)`` per the
post-008 block-dim convention ``(PAT_N=BN/F_N, PAT_M=BM/F_M)`` (lane_x
= innermost), compute the smem byte offset, take ``addr // 4 % 32``,
count distinct lanes per bank.

This module is a closed-form version of the same analysis
``013_pad_smem.py`` already performs via
``_helpers.py:max_bank_conflict`` — but parameterized by hypothetical
tile params so we can search the candidate space analytically before
generating any code.

Two analyses:

- :func:`predict_b_bank_conflicts` — N-direction loads of weight smem
  (the consumer Loads the cuBLAS-style asymmetric tile vectorizes).
- :func:`predict_a_bank_conflicts` — M-direction loads of input smem.
  Always 1 (broadcast) under the cuBLAS layout, but useful for sanity.

Then :func:`find_best_tile` enumerates ``(BM, BN, F_M, F_N)`` configs,
predicts conflicts, and ranks by ``(conflicts, -reuse, smem_bytes)``.
"""

from __future__ import annotations

from dataclasses import dataclass

WARP_SIZE = 32
BANKS = 32
BYTES_PER_ELEM = 4


@dataclass(frozen=True)
class TileCandidate:
    bm: int
    bn: int
    f_m: int
    f_n: int

    @property
    def threads(self) -> int:
        """Total threads/CTA after 008 splits the THREAD axes."""
        return (self.bm // self.f_m) * (self.bn // self.f_n)

    @property
    def per_thread_outputs(self) -> int:
        return self.f_m * self.f_n

    @property
    def reuse(self) -> float:
        """FMAs per smem load. Higher = less LSU pressure per FMA."""
        # Per K_inner iter: F_M + F_N smem loads, F_M * F_N FMAs.
        return self.per_thread_outputs / (self.f_m + self.f_n)

    def smem_bytes(self, bk: int, n_buffers: int = 2) -> int:
        """Per-stage smem footprint at this BK (weight + input,
        ``n_buffers`` slabs each for double-buffering)."""
        weight = self.bn * bk
        input_ = self.bm * bk
        return (weight + input_) * BYTES_PER_ELEM * n_buffers


def _max_distinct_addrs_per_bank(addrs_by_lane: dict[int, int]) -> int:
    """Worst-way conflict = max DISTINCT addresses per bank across the
    warp's 32 lanes. Same-address-multiple-lanes hits broadcast and
    counts as 1, not N. Mirrors ``_helpers.py:max_bank_conflict``'s
    accounting."""
    bank_to_addrs: dict[int, set[int]] = {}
    for _lane, addr in addrs_by_lane.items():
        bank_to_addrs.setdefault(addr % BANKS, set()).add(addr)
    return max((len(s) for s in bank_to_addrs.values()), default=1)


def predict_b_bank_conflicts(cand: TileCandidate, bk: int, *, weight_layout: str = "KN", pad: int = 0) -> int:
    """Worst-way bank conflict for weight (B-operand) loads.

    Block dim is ``(PAT_N=BN/F_N, PAT_M=BM/F_M, 1)`` so lane_x is the
    fastest-varying within a warp. Per-warp, lane_y may vary too if
    PAT_N < 32 (warp spans multiple lane_y values, each with its own
    m_thread). For B reads ``weight_smem[k_inner, n_thread + j_n]``
    with ``KN`` layout (N inner), the bank only depends on
    ``n_thread + j_n`` per LDS (k is broadcast)."""
    if weight_layout != "KN":
        raise NotImplementedError(f"weight_layout={weight_layout!r} not analyzed yet")
    pat_n = cand.bn // cand.f_n
    max_way = 1
    for j_n in range(cand.f_n):
        addrs_by_lane: dict[int, int] = {}
        for lane in range(WARP_SIZE):
            lane_x = lane % pat_n
            n_index = lane_x * cand.f_n + j_n
            addrs_by_lane[lane] = n_index  # in fp32 elements; bank = addr % 32
        max_way = max(max_way, _max_distinct_addrs_per_bank(addrs_by_lane))
    _ = pad  # future: across-K shift analysis
    return max_way


def predict_a_bank_conflicts(cand: TileCandidate, bk: int, *, pad: int = 0) -> int:
    """Worst-way bank conflict for input (A-operand) loads at smem
    layout ``[BM, BK+pad]`` (M outer, K inner). Per LDS reads
    ``input_smem[m_thread + j_m, k_inner]``. With block dim
    ``(PAT_N, PAT_M)`` lane_x = lane % PAT_N, lane_y = lane / PAT_N,
    m_thread = lane_y * F_M. PAT_N ≥ 32 → all warp lanes share the
    same lane_y → 1 distinct address (broadcast). PAT_N < 32 → warp
    spans multiple lane_y values; each address may hit a unique bank
    or alias depending on ``(BK + pad) mod 32``."""
    pat_n = cand.bn // cand.f_n
    stride = bk + pad
    max_way = 1
    for j_m in range(cand.f_m):
        addrs_by_lane: dict[int, int] = {}
        for lane in range(WARP_SIZE):
            lane_y = lane // pat_n
            m_thread = lane_y * cand.f_m
            m_index = m_thread + j_m
            addrs_by_lane[lane] = m_index * stride
        max_way = max(max_way, _max_distinct_addrs_per_bank(addrs_by_lane))
    return max_way


def enumerate_candidates(
    *,
    thread_budget: int = 256,
    smem_budget_bytes: int = 48 * 1024,
    bk: int = 32,
    bm_choices: tuple[int, ...] = (16, 32, 64, 128, 256),
    bn_choices: tuple[int, ...] = (16, 32, 64, 128, 256),
    f_choices: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> list[TileCandidate]:
    """All ``(BM, BN, F_M, F_N)`` combos that satisfy the thread-budget
    and smem-footprint constraints."""
    out: list[TileCandidate] = []
    for bm in bm_choices:
        for bn in bn_choices:
            for f_m in f_choices:
                if bm % f_m or f_m > bm:
                    continue
                for f_n in f_choices:
                    if bn % f_n or f_n > bn:
                        continue
                    cand = TileCandidate(bm=bm, bn=bn, f_m=f_m, f_n=f_n)
                    if cand.threads != thread_budget:
                        continue
                    if cand.smem_bytes(bk) > smem_budget_bytes:
                        continue
                    out.append(cand)
    return out


def effective_b_conflict_cost(cand: TileCandidate, bk: int, **kw) -> float:
    """Bank conflict cost normalized by LDS vector width — LDS.128
    inherently takes 4 cycles to drain a warp regardless of conflict
    pattern, so a 4-way conflict on F_N=4 (which NVCC vectorizes to
    LDS.128) costs the same as a 1-way conflict on F_N=1 (LDS.32).
    Empirically validated on k_add_5_reduce: 4-way @ F_N=4 = 673us
    while 1-way @ F_N=2 = 1253us.

    Cost model: ``raw_conflict_cycles = conflict_way × 1`` (for LDS.32)
    or ``conflict_way × ceil(F_N/4)`` (for LDS-wide). LDS.128 absorbs
    the first 4-way for free since the 4 cycles are needed anyway.
    """
    raw = predict_b_bank_conflicts(cand, bk, **kw)
    # LDS.128 lower bound: 4 cycles per warp regardless. Wider F_N
    # issues multiple LDS instructions, multiplying.
    lds_cycles_per_iter = max(4, cand.f_n)  # 4 cycles minimum
    # If conflict ≤ vector_width, no extra cost. Otherwise scale.
    vector_width = min(4, cand.f_n)  # LDS.128 width in elements
    extra_cycles = max(0, raw - vector_width)
    return lds_cycles_per_iter + extra_cycles


def score(cand: TileCandidate, bk: int, *, weight_layout: str = "KN", pad: int = 0) -> tuple[float, float, int]:
    """Ranking key for analytical pruning. Lower is better.
    ``(conflict_cost, -reuse, smem_bytes)``."""
    eff = effective_b_conflict_cost(cand, bk, weight_layout=weight_layout, pad=pad)
    return (eff, -cand.reuse, cand.smem_bytes(bk))


def candidates_to_try(
    m: int,
    k: int,
    n: int,
    *,
    bk_choices: tuple[int, ...] = (16, 32, 64),
    thread_budget: int = 256,
    smem_budget_bytes: int = 96 * 1024,
    max_results: int = 12,
    min_reuse: float = 1.0,
) -> list[tuple[TileCandidate, int]]:
    """Analytical filter — returns a small list of ``(tile, BK)`` pairs
    worth empirically benching for the given ``(M, K, N)`` matmul shape.

    Filters applied:
    - Smem footprint ≤ ``smem_budget_bytes``
    - Thread count == ``thread_budget`` (256)
    - Per-axis F divides per-axis B
    - ``reuse ≥ min_reuse`` (FMAs / smem-load ratio — anything below
      ~1 is bandwidth-bound and not worth trying)
    - Tile divides ``M`` and ``N`` cleanly (no boundary checks needed)

    Sort: by ``(effective_b_conflict_cost, -reuse, smem_bytes)``. The
    LDS.128-aware cost recognizes that 4-way @ F_N=4 is "free"
    (absorbed by LDS.128's natural 4-cycle warp drain), so the top of
    the list isn't dominated by F_N=1 / low-reuse degenerate configs.

    Returns up to ``max_results`` entries — short enough to micro-bench
    each (~5 sec/config via ``deplodock run --bench --ir <kernel>.json``)
    and pick the empirical winner."""
    seen: set[tuple[int, int, int, int, int]] = set()
    ranked: list[tuple[float, float, int, TileCandidate, int]] = []
    for bk in bk_choices:
        for cand in enumerate_candidates(thread_budget=thread_budget, smem_budget_bytes=smem_budget_bytes, bk=bk):
            if m % cand.bm or n % cand.bn:
                continue
            if cand.reuse < min_reuse:
                continue
            key = (cand.bm, cand.bn, cand.f_m, cand.f_n, bk)
            if key in seen:
                continue
            seen.add(key)
            cost, neg_reuse, smem = score(cand, bk)
            ranked.append((cost, neg_reuse, smem, cand, bk))
    ranked.sort(key=lambda x: x[:3])  # only sort by the cost / reuse / smem keys
    _ = k  # K affects only K_outer iter count, captured by BK choice
    return [(c, bk) for _, _, _, c, bk in ranked[:max_results]]


def find_best_tile(
    *,
    bk: int = 32,
    thread_budget: int = 256,
    smem_budget_bytes: int = 48 * 1024,
    weight_layout: str = "KN",
    pad: int = 0,
) -> tuple[TileCandidate, list[tuple[TileCandidate, tuple[int, float, int, int]]]]:
    """Return the best candidate plus the full ranked list. Best is
    sorted by (B conflicts, -reuse, A conflicts, smem)."""
    candidates = enumerate_candidates(thread_budget=thread_budget, smem_budget_bytes=smem_budget_bytes, bk=bk)
    ranked = sorted(((c, score(c, bk, weight_layout=weight_layout, pad=pad)) for c in candidates), key=lambda x: x[1])
    return ranked[0][0], ranked
