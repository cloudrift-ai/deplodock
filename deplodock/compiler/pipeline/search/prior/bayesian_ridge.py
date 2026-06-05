"""Bayesian-ridge tuning prior — one GLOBAL model across every kernel, trained
online during ``tune`` and checkpointed to the DB, backed by scikit-learn.

Replaces the nuked hand-written ``TileOp.score`` tiebreak with a model that
learns which knobs are good. It is *global*: each op's inner search trains it on
``archived + that op's tree`` (:attr:`_archived_rows` holds the finished ops'
rows, this run + any loaded checkpoint), so the ``S_*`` structural features vary
across ops → the model learns op-structure → knob quality and generalizes to
kernels it has not tuned. :meth:`to_bytes` / :meth:`from_bytes` round-trip the
model + the archived rows, so a follow-up ``tune`` keeps accumulating and a
``compile`` / ``run`` loads it to pick knobs.

Regresses ``y = log(best_reward)`` (= −log median
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
        # GLOBAL prior: final value-of-position rows from kernels tuned so far
        # (this run + any loaded checkpoint). Each refit trains on
        # ``archived + current-op tree``, so the ``S_*`` structural features
        # vary across ops → the model learns op-structure → knob quality and
        # generalizes to untuned kernels. ``archive`` freezes a finished op's
        # rows; the live op's rows arrive via :meth:`fit`'s ``rows`` argument.
        self._archived_rows: list[tuple[dict, float]] = []

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def archive(self, rows: list[tuple[dict, float]]) -> None:
        """Freeze a completed kernel's value-of-position rows into the global
        training set (called once the op's inner search finishes)."""
        self._archived_rows.extend((dict(k), float(v)) for k, v in rows)

    def fit(self, rows: list[tuple[dict, float]]) -> None:
        """Refit from ``archived + rows`` ((knobs, label) — the live op's tree
        snapshot plus every finished op): featurize, fit the standardizer +
        ``BayesianRidge`` (which infers its own ridge α and noise λ, and the
        intercept)."""
        all_rows = self._archived_rows + list(rows)
        feats = [knob.knob_features(k) for k, _ in all_rows]
        cols = sorted({c for f in feats for c in f})
        if not cols:
            return
        X = np.array([[f.get(c, 0.0) for c in cols] for f in feats], dtype=float)
        y = np.array([lab for _, lab in all_rows], dtype=float)
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
        """Serialize the model **and the archived rows** (``None`` if there's
        nothing yet). Pickles the sklearn ``StandardScaler`` + ``BayesianRidge``
        (importable classes — no bare-stem refs); the archived rows let a
        follow-up tune keep training the same global prior without forgetting,
        and a ``compile`` / ``run`` load just uses the model."""
        if self._model is None and not self._archived_rows:
            return None
        return pickle.dumps({"cols": self._cols, "scaler": self._scaler, "model": self._model, "archived_rows": self._archived_rows})

    @classmethod
    def from_bytes(cls, blob: bytes) -> BayesianRidgePrior:
        """Reconstruct a checkpointed global prior — model (for inference /
        warm-start) plus the archived rows (so a tune keeps accumulating)."""
        st = pickle.loads(blob)
        p = cls()
        p._cols, p._scaler, p._model = st["cols"], st["scaler"], st["model"]
        p._archived_rows = st.get("archived_rows", [])
        if p._model is not None:
            p._first_fit_idx = 0  # warm from the start — no cold warmup this run
        return p

    @classmethod
    def load(cls, regime_key: str, *, seed: int = 0, path=None) -> BayesianRidgePrior:
        """The global prior for ``regime_key`` — warm from its checkpoint file if
        present, else fresh — bound so :meth:`checkpoint` saves it back. ``path``
        defaults to ``config.prior_path()``."""
        from deplodock import config  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.prior import store  # noqa: PLC0415

        path = path or config.prior_path()
        blob = store.load(path, regime_key)
        p = cls.from_bytes(blob) if blob is not None else cls(seed=seed)
        p.regime_key, p._path = regime_key, path
        return p
