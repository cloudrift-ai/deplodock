"""Analytic prior — a stateless, hand-weighted :class:`Prior` over
``features.knob_features``.

This is the *untrained* prior: the cold-start ranking the search uses before any
tuning data exists. It replaces the old hand-coded matmul heuristic
(``score_matmul_thread`` + the ``_priority_matmul_*`` enumeration sort) — same
features, now expressed as a fixed linear model over the one shared feature dict
``features.knob_features`` produces, so there is a SINGLE ranking path: a config is
scored by a ``Prior`` (this one cold, ``CatBoostPrior`` once trained), composed
behind :class:`~emmy.compiler.pipeline.search.prior.fallback.FallbackPrior`.

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

from emmy.compiler.pipeline.search.features import knob_features
from emmy.compiler.pipeline.search.prior.base import Prior

# Linear weights over ``features.knob_features`` (``D_*`` geometry keys + ``MMA_tier``),
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
    "D_pow2_threads": 21.028373153022503,
    "D_splitk_le2": 12.423479314133088,
    "D_ctas_ge_sm": -7.948709567336203,
    "D_bn_band": -7.821722985417555,
    "MMA_tier": 7.172491368387892,
    "D_bm_band": 6.8615559657079475,
    "D_tilen_clean": 6.416225259656883,
    "D_bn_ge_bm": 4.1047225430800305,
    "D_w_l2_bk": 3.8318001959254397,
    "D_near_waves": 3.3396197822131195,
    "D_w_near_bk": 2.788315108732825,
    "D_near_area": 2.570603647930318,
    "D_square": 2.4682813427691133,
    "D_bk_ge32": -2.250612760379462,
    "D_near_intensity": -2.040435217053873,
    "D_l2_bm": -1.8789585273911558,
    "D_near_tilen": -1.8722293684790128,
    "D_near_threads": 1.610234563591009,
    "D_log2_area": -1.5867630330619453,
    "D_l2_reuse": 1.2797850818471639,
    "D_l2_bk": 0.9757523358723075,
    "D_neg_masked_k": -0.8646029028646691,
    "D_neg_masked_m": -0.8646029028646691,
    "D_neg_masked_n": -0.8646029028646691,
    "D_l2_cells_occ": 0.6315974578405096,
    "D_l2_bn": -0.5480949828814328,
    "D_splitk_excess": -0.5464648932550946,
    "D_near_kchunks": -0.42097617675856036,
    "D_splitk": -0.38586117939940745,
    "D_log2_waves": -0.3096053721652958,
    "D_log2_ctas": -0.195971369583823,
    "D_l2_threads": -0.15663789439639914,
    "D_aspect": 0.1452188967685775,
    "D_cells_cap": -0.10275127787239871,
    "D_reuse": -0.05674096013170091,
    "D_near_cells": -0.03521970822845533,
    "D_tile_n": -0.014204261399731513,
    "D_cells": -0.007608072063392555,
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
    "D_tilen_clean": 14.742589019163587,
    "D_bn_ge_bm": 14.511156339086375,
    "D_bn_band": -9.538827014524168,
    "D_pow2_threads": -7.150597557048656,
    "D_stage_ring": 6.745928314041857,
    "D_l2_bn": -4.00479195646865,
    "D_near_intensity": -3.259364460492787,
    "D_l2_reuse": 2.9132578004646428,
    "D_bm_band": -2.568184097080697,
    "D_ctas_ge_sm": -1.9462192907579767,
    "D_stage_async": 1.8834908323381063,
    "D_log2_waves": -1.7528995570518198,
    "D_splitk_excess": 1.7027100468896357,
    "D_l2_bm": -1.5327855127310666,
    "D_neg_masked_n": -1.4668126972062272,
    "D_near_area": 1.4158926817835582,
    "D_finalize_kernel": -1.3965113194734327,
    "D_near_threads": -1.3844620912584973,
    "D_bk_ge32": 1.326219856390027,
    "MMA_tier": -1.2534173382382152,
    "D_near_tilen": 1.2470208178221367,
    "D_l2_bk": -1.1926773964354764,
    "D_near_waves": -1.039843966378179,
    "D_splitk_le2": 0.9867238400326539,
    "D_aspect": 0.9042342460189189,
    "D_neg_masked_m": 0.7592693851163446,
    "D_w_l2_bk": 0.7321116149309171,
    "D_l2_cells_occ": 0.7122682887515748,
    "D_stage_tma": 0.6981159102868509,
    "D_near_kchunks": 0.5909235484526739,
    "D_square": -0.5632015743257406,
    "D_splitk": -0.45343397768078963,
    "D_neg_masked_k": 0.43858477080877745,
    "D_log2_ctas": 0.37533044561331635,
    "D_w_near_bk": 0.3442724287325934,
    "D_log2_area": 0.2833357073787948,
    "D_l2_threads": -0.20603274081292097,
    "D_cells_cap": -0.19740500851481346,
    "D_stage_reg_depth": 0.1791660246430764,
    "D_stage_depth": -0.17391526278300756,
    "D_near_cells": -0.16143971180120867,
    "D_tile_m": -0.07181251219230106,
    "D_cells": 0.058818028512549095,
    "D_reuse": 0.043592289527736194,
    "D_threads": -0.004719911837716548,
    "D_tile_n": -0.0005224290385496793,
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
        # Atomic-free split-K preference.
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
        feats = knob_features(knobs)
        w_set = self._w_dyn if feats.get("S_ext_n_symbolic_axis", 0.0) > 0 else self._w
        quality = sum(w * feats.get(k, 0.0) for k, w in w_set.items())
        # Deferred-kernel split-K finalize gate (local term — see __init__). The
        # cross-CTA finalize is the REDUCE codec's ``c`` letter, featurized as
        # ``D_finalize_kernel`` (1 when the deferred ``c<cta>k`` combine kernel is on).
        # The ``af_on · (±1)`` product is the interaction a plain weight can't express:
        # above the split threshold REWARD the deferred fold (higher quality → lower
        # latency proxy), below it PENALIZE so a narrow split keeps the cheap atomicAdd
        # fast-path. The atomic finalize scores zero either way (af_on = 0), so it keeps
        # its geometry-driven rank.
        af_on = feats.get("D_finalize_kernel", 0.0)
        if af_on:
            splitk = feats.get("D_splitk", 1.0)  # the split-K count (REDUCE@<k>.cta)
            many_splits = splitk >= self._atomic_free_split_threshold
            quality += self._atomic_free_weight * af_on * (1.0 if many_splits else -1.0)
        return math.exp(-self._scale * max(min(quality, 80.0), -80.0))

    def mean_score(self, knobs: dict) -> float:
        return self.score(knobs)
