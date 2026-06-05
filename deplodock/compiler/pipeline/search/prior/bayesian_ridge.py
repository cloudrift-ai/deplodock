"""Bayesian-ridge inner-tune prior — fit in memory from one tune run's own
benches, backed by scikit-learn.

Replaces the nuked hand-written ``TileOp.score`` tiebreak with a model that
learns which knobs are good. Regresses ``y = log(best_reward)`` (= −log median
latency, so a few catastrophic configs don't capture the fit) on
:func:`knob.knob_features`, pipelined through a :class:`StandardScaler` (so the
ridge sees unit-scale features and an absent knob — 0-filled — maps to a fixed
coordinate) into :class:`sklearn.linear_model.BayesianRidge` (which fits the
regularization + the noise level itself and exposes a per-point predictive
mean + std).

:meth:`score` is a Thompson draw — ``mean + std·z`` per call — so the MCTS PUCT
softmax over a sibling set samples each sibling once; :meth:`mean_score` is the
noise-free posterior mean used for the greedy argmax + calibration. Magnitude is
irrelevant, only the ranking.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.prior.base import Prior


class BayesianRidgePrior(Prior):
    """In-memory scikit-learn Bayesian-ridge prior over knob features."""

    def __init__(self, *, seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(seed)
        self._cols: list[str] | None = None
        self._scaler: StandardScaler | None = None
        self._model: BayesianRidge | None = None

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def fit(self, rows: list[tuple[dict, float]]) -> None:
        """Refit from value-of-position rows ``(knobs, label)``: featurize, fit
        the standardizer + ``BayesianRidge`` (which infers its own ridge α and
        noise λ, and the intercept)."""
        feats = [knob.knob_features(k) for k, _ in rows]
        cols = sorted({c for f in feats for c in f})
        if not cols:
            return
        X = np.array([[f.get(c, 0.0) for c in cols] for f in feats], dtype=float)
        y = np.array([lab for _, lab in rows], dtype=float)
        scaler = StandardScaler().fit(X)  # zero-variance columns → scale_=1 (left unchanged)
        model = BayesianRidge().fit(scaler.transform(X), y)
        self._cols, self._scaler, self._model = cols, scaler, model
        self._note_fit()

    def resample(self) -> None:
        # No-op: the Thompson draw happens per :meth:`score` call (sklearn gives
        # a per-point predictive std, so each sibling is sampled independently).
        pass

    def score(self, knobs: dict) -> float:
        if self._model is None:
            return 0.0
        mean, std = self._model.predict(self._x(knobs), return_std=True)
        return float(mean[0] + std[0] * self._rng.standard_normal())

    def mean_score(self, knobs: dict) -> float:
        if self._model is None:
            return 0.0
        return float(self._model.predict(self._x(knobs))[0])

    def _x(self, knobs: dict) -> np.ndarray:
        assert self._cols is not None and self._scaler is not None
        f = knob.knob_features(knobs)
        x = np.array([[f.get(c, 0.0) for c in self._cols]], dtype=float)
        return self._scaler.transform(x)

    # --- persistence ------------------------------------------------------

    def to_bytes(self) -> bytes | None:
        """Serialize the fitted model (``None`` if unfit). Pickles the sklearn
        ``StandardScaler`` + ``BayesianRidge`` (both importable classes — no
        bare-stem module refs), so the blob loads in a fresh ``compile``."""
        if self._model is None:
            return None
        return pickle.dumps({"cols": self._cols, "scaler": self._scaler, "model": self._model})

    @classmethod
    def from_bytes(cls, blob: bytes) -> BayesianRidgePrior:
        """Reconstruct a fitted prior for inference from a :meth:`to_bytes` blob."""
        st = pickle.loads(blob)
        p = cls()
        p._cols, p._scaler, p._model = st["cols"], st["scaler"], st["model"]
        return p
