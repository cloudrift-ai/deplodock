"""CatBoost tuning prior — ONE global model across every kernel, GPU and nvcc
setting, refit in batches from a bounded dataset and checkpointed to JSON.

**Why CatBoost**: an offline evaluation script measured
linear / rf / hgb / xgb / lgbm / catboost over a multi-op tuning dataset, and
CatBoost performed the best.

Regresses ``y = log(median latency µs)`` on :func:`knob.knob_features` — the
``S_*`` structural + ``H_*`` hardware/nvcc-regime features let one model tell
kernels and regimes apart from the feature vector (log space keeps the RMSE fit
from being dominated by the slow configs, where we least care). :meth:`score` /
:meth:`mean_score` return the predicted latency in µs (``exp`` of the regressed
log-latency); the search converts that to reward. **Lower is better** — greedy
picks the ``mean_score`` *argmin*, and PUCT (see :mod:`policy.mcts`) inverts the
prediction to a reward. Predictions are deterministic, so :meth:`score` (PUCT
ranking) and :meth:`mean_score` (greedy + calibration) coincide; PUCT explores via
its own exploration term rather than a Thompson draw. :meth:`to_json` / :meth:`from_json`
round-trip the CatBoost model (its native ``cbm`` blob, base64'd) plus the
reservoir dataset, so a follow-up ``tune`` keeps accumulating and a ``compile`` /
``run`` loads it to pick knobs.
"""

from __future__ import annotations

import base64
import os
import tempfile

import numpy as np

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.prior.base import Prior


class CatBoostPrior(Prior):
    """CatBoost-backed global prior over knob features."""

    ITERATIONS = 400
    DEPTH = 6
    LEARNING_RATE = 0.05

    def __init__(self, *, iterations: int = ITERATIONS, **kwargs) -> None:
        super().__init__(**kwargs)
        self._iterations = iterations
        self._cols: list[str] | None = None
        self._model = None

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def fit(self) -> None:
        """Refit a fresh CatBoostRegressor on the whole bounded dataset (RMSE on
        ``log(latency µs)``). Columns are the sorted union of feature keys; an
        absent knob fills to ``NaN`` (NOT ``0.0``) so CatBoost's native
        missing-value handling (``nan_mode="Min"``) can split "this knob isn't
        present on this row" off from a knob that is present *and* legitimately
        zero (a BOOL ``False`` → ``0.0``, ``STAGE="00"`` → ``popcount 0.0``).
        With a 0-fill those two collapsed to the same coordinate; ``NaN`` keeps
        them distinct (cross-op missing knobs — ``MMA`` on an fp32 row — and the
        partial-prefix descent rows are the cases that differ). Non-positive
        labels (a stale pre-latency checkpoint stored log-reward rows) are
        dropped so ``log`` never sees a non-positive latency — rebuild such a
        prior with ``tune --clean``."""
        from catboost import CatBoostRegressor  # noqa: PLC0415

        rows = [(k, lab) for k, lab in self._dataset if lab > 0]
        feats = [knob.knob_features(k) for k, _ in rows]
        cols = sorted({c for f in feats for c in f})
        if not cols:
            return
        x = np.array([[f.get(c, np.nan) for c in cols] for f in feats], dtype=float)
        y = np.log(np.array([lab for _, lab in rows], dtype=float))
        model = CatBoostRegressor(
            iterations=self._iterations,
            depth=self.DEPTH,
            learning_rate=self.LEARNING_RATE,
            loss_function="RMSE",
            random_seed=self._seed,
            thread_count=-1,
            nan_mode="Min",  # treat absent (NaN) features as their own split-off bucket
            verbose=False,
            allow_writing_files=False,  # don't litter a catboost_info/ dir at the cwd
        )
        model.fit(x, y)
        self._cols, self._model = cols, model

    def score(self, knobs: dict) -> float:
        return self.mean_score(knobs)

    def mean_score(self, knobs: dict) -> float:
        """Predicted median latency in µs (``exp`` of the regressed log-latency);
        ``0.0`` until the first fit. Lower is better."""
        if self._model is None:
            return 0.0
        f = knob.knob_features(knobs)
        x = np.array([[f.get(c, np.nan) for c in self._cols]], dtype=float)  # absent → NaN, matching fit (see fit docstring)
        return float(np.exp(self._model.predict(x)[0]))

    # --- persistence ------------------------------------------------------

    def to_json(self) -> dict | None:
        """Serialize the CatBoost model (native ``cbm`` blob, base64'd) + the
        reservoir dataset + counters (``None`` when there's nothing yet)."""
        if self._model is None and not self._dataset:
            return None
        return {
            "model": self._model_b64(),
            "cols": self._cols,
            "dataset": [[dict(k), float(v)] for k, v in self._dataset],
            "seen": self._seen,
            "since_fit": self._since_fit,
        }

    @classmethod
    def from_json(cls, obj: dict) -> CatBoostPrior:
        """Reconstruct a checkpointed prior from :meth:`to_json` — model (for
        inference / warm-start) plus the reservoir dataset (so a tune keeps
        accumulating). Tolerant of a stale checkpoint: an incompatible / corrupt
        model blob is dropped (the rows are still salvaged and a refit rebuilds
        the model), so e.g. a pre-CatBoost prior file migrates instead of crashing."""
        p = cls()
        p._cols = obj.get("cols")
        # ``dataset`` is the current key; ``archived_rows`` the legacy one — keep
        # either so an old checkpoint's accumulated rows survive the migration.
        raw = obj.get("dataset") or obj.get("archived_rows") or []
        p._dataset = [(dict(k), float(v)) for k, v in raw]
        p._seen = int(obj.get("seen", len(p._dataset)))
        p._since_fit = int(obj.get("since_fit", 0))
        # Only a base64 string is a CatBoost ``cbm`` blob; anything else (e.g. a
        # dict — the former BayesianRidge prior's sklearn estimator state) is
        # discarded, and the next refit rebuilds the model from the rows.
        blob = obj.get("model")
        if isinstance(blob, str) and blob:
            try:
                p._model = cls._model_from_b64(blob)
                p._first_fit_idx = 0  # loaded → warm, no cold warmup this run
            except Exception:  # noqa: BLE001 — corrupt/incompatible blob → refit fresh
                p._model = None
        return p

    @classmethod
    def load(cls, *, seed: int = 0, path=None) -> CatBoostPrior:
        """The one global prior — warm from its JSON checkpoint if present, else
        fresh — bound so :meth:`checkpoint` saves it back. ``path`` defaults to
        ``config.prior_path()``. Best-effort: an unreadable / incompatible
        checkpoint falls back to a fresh prior rather than failing the compile."""
        from deplodock import config, storage  # noqa: PLC0415

        path = path or config.prior_path()
        obj = storage.read_json(path)
        try:
            p = cls.from_json(obj) if isinstance(obj, dict) else cls(seed=seed)
        except Exception:  # noqa: BLE001 — never let a bad checkpoint break tune/compile
            p = cls(seed=seed)
        p._path = path
        return p

    def _model_b64(self) -> str | None:
        """CatBoost has no in-memory to-bytes API, so round-trip the native
        ``cbm`` file through a tempfile and base64 it for the JSON."""
        if self._model is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            tmp = f.name
        try:
            self._model.save_model(tmp, format="cbm")
            blob = open(tmp, "rb").read()  # noqa: SIM115
        finally:
            os.unlink(tmp)
        return base64.b64encode(blob).decode("ascii")

    @staticmethod
    def _model_from_b64(b64: str):
        from catboost import CatBoostRegressor  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            f.write(base64.b64decode(b64))
            tmp = f.name
        try:
            model = CatBoostRegressor()
            model.load_model(tmp, format="cbm")
        finally:
            os.unlink(tmp)
        return model
