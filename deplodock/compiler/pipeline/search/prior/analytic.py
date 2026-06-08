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
``D_*`` features ranks them all (the warp tier rides tier-aware targets in
``_geom_feats``; the reduce signal rides thread-count / occupancy as cooperative
``BR`` raises the thread count). It replaces the old per-mode ``_priority_*``
enumeration sorts (matmul / reduce / pointwise), which were the cold ranking
before.
"""

from __future__ import annotations

import math

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.prior.base import Prior

# Linear weights over ``knob.knob_features`` ``D_*`` keys, fit offline by
# ``scripts/golden_knob_heuristics.py`` jointly over ALL kernel regimes —
# fp32-scalar + fp16/bf16-warp matmul, cooperative reduce, and pointwise goldens —
# tier-balanced (each regime weighted equally so the sparse reduce/pointwise tiers
# aren't drowned by the matmul shapes), minimizing the goldens' tier-weighted mean
# ``log2(rank+1)``. Dominant terms: occupancy (``D_ctas_ge_sm``/``D_near_waves`` —
# keep #CTAs ≈ 2 waves over the SMs), the ``D_bm_band`` thread-tile band, the
# tier-split warp BK target (``D_w_near_bk`` — BK≈2 on the TMA tile), and the
# reduce signal rides ``D_threads``/occupancy (cooperative ``BR`` raises thread
# count toward the target). One un-gated linear model serves every regime, so some
# band weights compromise across regimes (e.g. ``D_bn_band`` is mildly negative —
# matmul wants the band but reduce wants BN=1; the fit trades it for occupancy).
_W_A: dict[str, float] = {
    "D_bm_band": 14.920552348240792,
    "D_splitk_le2": 8.407426536981845,
    "D_ctas_ge_sm": -7.948709567336204,
    "D_w_near_bk": 6.8518793473315265,
    "D_bn_ge_bm": 6.278827239141573,
    "D_near_waves": 3.3396197822131195,
    "D_square": 3.0592744803555165,
    "D_near_area": 2.5706036479303176,
    "D_tilen_clean": -2.5517002129957875,
    "D_splitk": -2.2594440169222545,
    "D_bk_ge32": -2.2506127603794623,
    "D_l2_bm": -2.1021570661314115,
    "D_near_tilen": -1.8722293684790134,
    "D_log2_area": -1.819577558854622,
    "D_bn_band": -1.6254704053387636,
    "D_l2_reuse": 1.4867241939733629,
    "D_l2_bk": 1.259242110612971,
    "D_near_kchunks": -0.8403134313789603,
    "D_aspect": -0.700694025296909,
    "D_w_l2_bk": 0.5978498582272562,
    "D_log2_ctas": 0.542881044276033,
    "D_pow2_threads": -0.4401714077219094,
    "D_near_intensity": -0.35192753800024107,
    "D_log2_waves": -0.3096053721652958,
    "D_l2_bn": -0.18853218023203946,
    "D_neg_overhang": 0.1648645127840344,
    "D_l2_threads": -0.15663789439639914,
    "D_near_threads": 0.10459439545995661,
    "D_cells_cap": -0.10275127787239871,
    "D_near_cells": -0.05255466763042605,
    "D_reuse": -0.03051986517890071,
    "D_cells": -0.022208323805983393,
    "D_threads": -0.004295893049161815,
    "D_tile_m": -0.0026049408030261903,
    "D_tile_n": -0.0015936106921647275,
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
