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
    "D_bm_band": 10.869206451133332,
    "D_splitk_le2": 8.407426536981845,
    "D_ctas_ge_sm": -7.948709567336204,
    "D_w_near_bk": 6.8518793473315265,
    "D_bn_ge_bm": 4.1047225430800305,
    "D_bn_band": -3.6793127460255177,
    "D_near_waves": 3.3396197822131195,
    "D_square": 3.0592744803555165,
    "D_near_area": 2.5706036479303176,
    "D_tilen_clean": -2.5517002129957875,
    "D_splitk": -2.2594440169222545,
    "D_bk_ge32": -2.250612760379462,
    "MMA_tier": 2.2010287724932924,
    "D_w_l2_bk": 2.0808985855150666,
    "D_near_tilen": -1.8722293684790134,
    "D_l2_bm": -1.8244728321406853,
    "D_log2_area": -1.5182763690806524,
    "D_l2_reuse": 1.4867241939733629,
    "D_near_intensity": -1.3805736365293138,
    "D_l2_bk": 0.8774891884764401,
    "D_neg_overhang": -0.8646029028646692,
    "D_near_kchunks": -0.8403134313789603,
    "D_l2_bn": -0.7907502921783296,
    "D_aspect": -0.700694025296909,
    "D_pow2_threads": -0.4401714077219094,
    "D_log2_waves": -0.3096053721652958,
    "D_log2_ctas": 0.23685242485122074,
    "D_l2_threads": -0.15663789439639914,
    "D_near_threads": 0.10459439545995662,
    "D_cells_cap": -0.10275127787239871,
    "D_reuse": -0.0606460824714877,
    "D_near_cells": -0.03521970822845533,
    "D_cells": -0.007608072063392554,
    "D_tile_n": -0.006832526644562641,
    "D_threads": -0.004295893049161819,
    "D_tile_m": -0.002604940803026191,
}


class AnalyticPrior(Prior):
    """Fixed linear ranker over ``knob_features`` — the cold-start prior.

    Stateless: ``fitted`` is always ``True`` (it has nothing to learn), and the
    training surface (``fit`` / ``add_rows`` / ``maybe_refit`` / ``to_json``) are
    no-ops so it composes cleanly under :class:`FallbackPrior`."""

    def __init__(self, *, weights: dict[str, float] | None = None, scale: float = 0.1) -> None:
        super().__init__()
        self._w = weights if weights is not None else _W_A
        # exp() argument scale — keeps the proxy in a finite, sane range; does not
        # affect ranking (monotone), only the proxy's magnitude.
        self._scale = scale

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
        scores the neutral ``1.0``, so ties fall to enumeration order."""
        feats = knob.knob_features(knobs)
        quality = sum(w * feats.get(k, 0.0) for k, w in self._w.items())
        return math.exp(-self._scale * max(min(quality, 80.0), -80.0))

    def mean_score(self, knobs: dict) -> float:
        return self.score(knobs)
