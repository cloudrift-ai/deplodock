"""Analytic prior — a stateless, hand-weighted :class:`Prior` over
``knob.knob_features``.

This is the *untrained* prior: the cold-start ranking the search uses before any
tuning data exists. It replaces the old hand-coded matmul heuristic
(``score_matmul_thread`` + the ``_priority_matmul_*`` enumeration sort) — same
features, now expressed as a fixed linear model over the one shared feature dict
``knob.knob_features`` produces, so there is a SINGLE ranking path: a config is
scored by a ``Prior`` (this one cold, ``CatBoostPrior`` once trained), composed
behind :class:`~deplodock.compiler.pipeline.search.prior.fallback.FallbackPrior`.

``score`` returns a positive latency *proxy* (``exp(-scale · wᵀfeatures)``),
**lower is better** — matching ``CatBoostPrior``'s polarity. The proxy is not
calibrated µs; only its ordering (greedy argmin / PUCT relative ``P``) matters.
The weights :data:`_W_A` are fit offline by ``scripts/golden_knob_heuristics.py``
jointly over EVERY kernel regime — fp32-scalar / fp16-warp matmul, cooperative
reduce, and pointwise goldens — so one un-gated linear model over the shared
``D_*`` features (plus ``MMA_tier``) ranks them all (the warp tier rides tier-aware
targets in ``_geom_feats`` plus a positive ``MMA_tier`` weight — fp16/bf16 prefer
the tensor-core tile over the scalar one, the warp-first default that used to live
in enumeration order; the reduce signal rides thread-count / occupancy as
cooperative ``BR`` raises the thread count). It replaces the old per-mode
``_priority_*`` enumeration sorts (matmul / reduce / pointwise), which were the
cold ranking before.
"""

from __future__ import annotations

import math

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.prior.base import Prior

# Linear weights over ``knob.knob_features`` (``D_*`` geometry keys + ``MMA_tier``),
# fit offline by ``scripts/golden_knob_heuristics.py`` jointly over ALL kernel
# regimes — fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise
# goldens — tier-balanced (each regime weighted equally so the sparse
# reduce/pointwise tiers aren't drowned by the matmul shapes), minimizing the
# goldens' tier-weighted mean ``log2(rank+1)``. Dominant terms: occupancy
# (``D_ctas_ge_sm``/``D_near_waves`` — keep #CTAs ≈ 2 waves over the SMs), the
# ``D_bm_band`` thread-tile band, the tier-split warp BK target (``D_w_near_bk`` —
# BK≈2 on the TMA tile), the positive ``MMA_tier`` (fp16/bf16 prefer the warp
# tensor-core tile — the fp16 cases enumerate BOTH tiers, so the fit must rank the
# warp golden over the scalar candidates), and the reduce signal rides
# ``D_threads``/occupancy (cooperative ``BR`` raises thread count toward the
# target). One un-gated linear model serves every regime, so some band weights
# compromise across regimes (e.g. ``D_bn_band`` is mildly negative — matmul wants
# the band but reduce wants BN=1; the fit trades it for occupancy).
_W_A: dict[str, float] = {
    "D_splitk_le2": 8.407426536981845,
    "D_ctas_ge_sm": -7.948709567336204,
    "D_bn_band": -7.821722985417555,
    "D_tilen_clean": 6.416225259656883,
    "D_w_near_bk": 6.211004717311999,
    "MMA_tier": 5.106063905982595,
    "D_bm_band": 4.79269048276045,
    "D_bn_ge_bm": 4.1047225430800305,
    "D_w_l2_bk": 3.6467210113988764,
    "D_near_waves": 3.3396197822131195,
    "D_square": 3.0592744803555165,
    "D_near_area": 2.5706036479303176,
    "D_bk_ge32": -2.250612760379462,
    "D_near_intensity": -2.040435217053873,
    "D_near_tilen": -1.8722293684790128,
    "D_log2_area": -1.5867630330619453,
    "D_l2_reuse": 1.2797850818471639,
    "D_l2_bk": 0.9757523358723075,
    # Per-role split of the former ``D_neg_overhang`` (same weight on each role —
    # the sum reproduces the old ``-count`` contribution exactly; a future refit
    # can separate them). See plans/drop-overhang-knob-structural-masked-feature.md.
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_neg_masked_k": -0.8646029028646691,
    "D_l2_bm": -0.8611753178680309,
    "D_aspect": -0.700694025296909,
    "D_l2_bn": -0.6882108423260582,
    "D_splitk": -0.5742697587615958,
    "D_pow2_threads": -0.4401714077219094,
    "D_log2_waves": -0.3096053721652958,
    "D_near_kchunks": -0.24376538495140618,
    "D_log2_ctas": 0.23685242485122074,
    "D_l2_threads": -0.15663789439639914,
    "D_near_threads": 0.10459439545995662,
    "D_cells_cap": -0.10275127787239871,
    "D_reuse": -0.056740960131700914,
    "D_near_cells": -0.03521970822845533,
    "D_cells": -0.007608072063392554,
    "D_tile_n": -0.005719609457676535,
    "D_threads": -0.004295893049161819,
    "D_tile_m": -0.003968686543542194,
}


# Masked-tier (symbolic-axis) weights — fit by the same script over the dynamic
# (``.dynM``) goldens only. A masked-tile kernel prices differently from its
# static twin: the boundary guard taxes small tiles, the staged prologues the
# static weights reward are locked out on symbolic rows, and the occupancy terms
# see a free-dim product that EXCLUDES the symbolic axis (the 992 stamp), so the
# static weights systematically under-size masked tiles (``BM 8/16``,
# ``SPLITK 1/2`` — the dynM seed report's finding 4). Selected at score time on
# the stamped ``S_ext_n_symbolic_axis`` flag.
_W_A_DYN: dict[str, float] = {
    "D_splitk_le2": 2.335205221027407,
    "D_near_kchunks": 2.0411714964370526,
    "D_l2_bn": -1.9721307282729108,
    "D_l2_bk": -1.9378821830668715,
    "D_ctas_ge_sm": -1.6368130082756995,
    "D_splitk": 1.506972617874152,
    "D_near_waves": -1.4885713149137085,
    # Per-role split of the former ``D_neg_overhang`` (see ``_W_A``).
    "D_neg_masked_m": 1.4232013571064015,
    "D_neg_masked_n": 1.4232013571064015,
    "D_neg_masked_k": 1.4232013571064015,
    "D_bm_band": -1.328509340005289,
    "D_l2_threads": -1.3114718972707482,
    "MMA_tier": 1.1625757242541237,
    "D_l2_bm": -1.1044983303644318,
    "D_near_intensity": -1.0373173103115754,
    "D_pow2_threads": 0.9477121154344588,
    "D_bn_ge_bm": -0.7675791001039283,
    "D_log2_area": 0.7283876807192595,
    "D_bn_band": -0.7001195998838845,
    "D_w_near_bk": -0.6919543937787341,
    "D_square": 0.6742826935596792,
    "D_near_area": 0.664416975841559,
    "D_near_tilen": 0.6216444551999621,
    "D_w_l2_bk": 0.47136282725736045,
    "D_tilen_clean": 0.461865876692245,
    "D_bk_ge32": -0.3082928739467462,
    "D_near_threads": -0.2164612881543422,
    "D_l2_cells_occ": 0.17912858998990044,
    "D_log2_ctas": 0.1586980948517343,
    "D_log2_waves": 0.12902916253215332,
    "D_near_cells": 0.09205423871906947,
    "D_l2_reuse": -0.08454193988132098,
    "D_cells": -0.05183793166434414,
    "D_reuse": 0.040639341150390126,
    "D_cells_cap": -0.026621293900614396,
    "D_tile_n": 0.0025264619706496572,
    "D_tile_m": -0.0024300935430789845,
    "D_threads": -0.00089339705959862,
    "D_aspect": 0.000349748592144826,
}


class AnalyticPrior(Prior):
    """Fixed linear ranker over ``knob_features`` — the cold-start prior.

    Stateless: ``fitted`` is always ``True`` (it has nothing to learn), and the
    training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``) are
    no-ops so it composes cleanly under :class:`FallbackPrior`. Two weight sets:
    ``weights`` for static shapes, ``weights_dynamic`` for symbolic-axis
    (masked-tile) kernels — picked per score on ``S_ext_n_symbolic_axis``."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        weights_dynamic: dict[str, float] | None = None,
        scale: float = 0.1,
        atomic_free_split_threshold: float = 4.0,
        atomic_free_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self._w = weights if weights is not None else _W_A
        self._w_dyn = weights_dynamic if weights_dynamic is not None else _W_A_DYN
        # exp() argument scale — keeps the proxy in a finite, sane range; does not
        # affect ranking (monotone), only the proxy's magnitude.
        self._scale = scale
        # Atomic-free split-K preference (see plans/atomic-free-monoid-combine.md).
        # Hardcoded — NOT fit into ``_W_A`` (a plain linear weight can't express the
        # "good when split wide, bad when split narrow" interaction). The learned
        # CatBoostPrior takes over once real atomic-vs-free ``H_opt=3`` rows exist.
        self._atomic_free_split_threshold = atomic_free_split_threshold
        self._atomic_free_weight = atomic_free_weight

    @property
    def fitted(self) -> bool:
        return True

    def fit(self) -> None:  # nothing to learn
        return None

    def add_rows(self, rows) -> None:  # noqa: ARG002 — stateless, ignores observations
        return None

    def maybe_refit(self, *, force: bool = False) -> bool:  # noqa: ARG002
        return False

    def to_json(self) -> dict | None:  # not persisted
        return None

    def score(self, knobs: dict) -> float:
        """Latency proxy (``exp(-scale · quality)``), lower is better. A config the
        weights have no opinion on (no ``D_*`` features — e.g. a non-tiled kernel)
        scores the neutral ``1.0``, so ties fall to enumeration order. Symbolic-axis
        (masked-tile) kernels rank under the dynamic weight set."""
        feats = knob.knob_features(knobs)
        w_set = self._w_dyn if feats.get("S_ext_n_symbolic_axis", 0.0) > 0 else self._w
        quality = sum(w * feats.get(k, 0.0) for k, w in w_set.items())
        # Atomic-free split-K gate (local term — see __init__). ``NOATOMIC`` is the
        # canonical knob name of ``ATOMIC_FREE_SPLITK`` (1 when the workspace fork is
        # on). The ``af_on · (±1)`` product is the interaction a plain weight can't
        # express: above the split threshold REWARD atomic-free (higher quality →
        # lower latency proxy), below it PENALIZE so a narrow split keeps the cheap
        # atomicAdd fast-path. NOATOMIC=False scores zero either way (af_on = 0), so
        # the atomic path keeps its geometry-driven rank.
        af_on = feats.get("NOATOMIC", 0.0)
        if af_on:
            many_splits = feats.get("SPLITK", 1.0) >= self._atomic_free_split_threshold
            quality += self._atomic_free_weight * af_on * (1.0 if many_splits else -1.0)
        return math.exp(-self._scale * max(min(quality, 80.0), -80.0))

    def mean_score(self, knobs: dict) -> float:
        return self.score(knobs)
