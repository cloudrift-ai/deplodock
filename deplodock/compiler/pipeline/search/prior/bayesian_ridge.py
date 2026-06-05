"""Bayesian-ridge inner-tune prior — Thompson-sampled, fit in memory from one
tune run's own benches.

Replaces the nuked hand-written ``TileOp.score`` tiebreak with a model that
learns which knobs are good. Regresses ``y = log(best_reward)`` (= −log median
latency, so a few catastrophic configs don't capture the fit) on
:func:`knob.knob_features`. Standardized features, ``y`` centered; ridge
posterior ``θ̄ = (XᵀX + λI)⁻¹ Xᵀy_c``, ``Σ = σ²(XᵀX + λI)⁻¹`` (σ² from
residuals). :meth:`resample` draws one ``θ̃ ~ N(θ̄, Σ)`` per descent; scores are
``θ̃·φ`` — scale-free ranking, absolute value irrelevant.

Missing knobs (a partial state lacks deeper knobs) are 0-filled then
standardized, mapping "absent" to a fixed coordinate distinct from any real
(positive) knob value — an implicit presence encoding. Within one sibling set
the absent pattern is constant, so it never perturbs the ranking that matters.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.prior.base import Prior


class BayesianRidgePrior(Prior):
    """In-memory Bayesian-ridge prior over knob features, Thompson-sampled."""

    def __init__(self, *, seed: int = 0, ridge: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ridge = ridge
        self._rng = np.random.default_rng(seed)
        self._cols: list[str] | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._theta: np.ndarray | None = None  # posterior mean
        self._cov_L: np.ndarray | None = None  # Cholesky of Σ
        self._y_mean = 0.0
        self._theta_sample: np.ndarray | None = None  # current Thompson draw

    @property
    def fitted(self) -> bool:
        return self._theta is not None

    def fit(self, rows: list[tuple[dict, float]]) -> None:
        """Refit the posterior from value-of-position rows ``(knobs, label)``.

        Standardizes features (zero-variance columns — e.g. the single op's
        constant structural features — are neutralized to 0), centers ``y``,
        ridge-solves the mean, and forms Σ for Thompson draws."""
        feats = [knob.knob_features(k) for k, _ in rows]
        cols = sorted({c for f in feats for c in f})
        if not cols:
            return
        X = np.array([[f.get(c, 0.0) for c in cols] for f in feats], dtype=float)
        y = np.array([lab for _, lab in rows], dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-9] = 1.0  # constant column → standardized to 0 (inert)
        Xs = (X - mean) / std
        y_mean = float(y.mean())
        yc = y - y_mean
        d = Xs.shape[1]
        A = Xs.T @ Xs + self._ridge * np.eye(d)
        Ainv = np.linalg.inv(A)
        theta = Ainv @ Xs.T @ yc
        resid = yc - Xs @ theta
        dof = max(len(rows) - d, 1)
        sigma2 = max(float(resid @ resid) / dof, 1e-9)
        cov = sigma2 * Ainv
        cov = 0.5 * (cov + cov.T)  # symmetrize against round-off before Cholesky
        try:
            cov_L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov_L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
        self._cols, self._mean, self._std = cols, mean, std
        self._theta, self._cov_L, self._y_mean = theta, cov_L, y_mean
        self._note_fit()
        self.resample()

    def resample(self) -> None:
        """Draw a fresh Thompson sample ``θ̃ ~ N(θ̄, Σ)`` — one per descent."""
        if self._theta is None or self._cov_L is None:
            self._theta_sample = None
            return
        z = self._rng.standard_normal(self._theta.shape[0])
        self._theta_sample = self._theta + self._cov_L @ z

    def score(self, knobs: dict) -> float:
        if not self.active or self._theta_sample is None:
            return 0.0
        return float(self._vec(knobs) @ self._theta_sample + self._y_mean)

    def mean_score(self, knobs: dict) -> float:
        if self._theta is None:
            return 0.0
        return float(self._vec(knobs) @ self._theta + self._y_mean)

    def _vec(self, knobs: dict) -> np.ndarray:
        assert self._cols is not None and self._mean is not None and self._std is not None
        f = knob.knob_features(knobs)
        x = np.array([f.get(c, 0.0) for c in self._cols], dtype=float)
        return (x - self._mean) / self._std
