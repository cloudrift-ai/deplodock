"""CatBoost tuning prior — ONE global model across every kernel, GPU and nvcc
setting, refit in batches from a bounded dataset and checkpointed to JSON.

**Why CatBoost** (chosen by an offline bake-off, `scripts/prior_bakeoff.py`, over
a multi-op tuning dataset; metrics = argmax latency-ratio in/out of sample +
per-op Spearman + off-manifold safety):

    model     in-sample  leave-1-op-out  spearman  off-manifold-safe
    linear      1.01        1.13            +0.66     NO (argmax → a degenerate
                                                          corner, e.g. BR=1/512:
                                                          4us → 232us / invalid)
    rf          1.00        1.31            +0.85     yes
    hgb         1.01        1.04            +0.79     yes
    xgb/lgbm    1.00        1.18            +0.85     ~ (slight creep)
    catboost    1.01        1.01            +0.82     yes   ◀ winner

The linear model (the old ``BayesianRidge`` prior) is monotone in every knob, so
its greedy argmax is always a *corner* of the candidate box regardless of how
well the interior is sampled — that is the `BR=1` blow-up. Any **bounded** tree
ensemble is off-manifold-safe (an un-benched extreme inherits the nearest leaf's
value, so it can't outrank the true optimum). Among them CatBoost uniquely also
**generalizes to an untuned op near-perfectly** (leave-one-op-out argmax 1.01 vs
xgb/lgbm 1.18, rf 1.31) — its ordered boosting + symmetric/oblivious trees
regularize hard against per-op overfitting. So one global CatBoost prior is good
enough on a *new* op that we no longer refit within an op's own search (the model
is fixed per run; see :mod:`base`).

Regresses ``y = log(best_reward)`` (= −log median latency) on
:func:`knob.knob_features` — the ``S_*`` structural + ``H_*`` hardware/nvcc-regime
features let one model tell kernels and regimes apart from the feature vector.
Predictions are deterministic, so :meth:`score` (PUCT ranking) and
:meth:`mean_score` (greedy argmax + calibration) coincide; PUCT explores via its
own exploration term rather than a Thompson draw. :meth:`to_json` / :meth:`from_json`
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
        the value-of-position labels). Columns are the sorted union of feature
        keys; an absent knob 0-fills to a fixed coordinate."""
        from catboost import CatBoostRegressor  # noqa: PLC0415

        feats = [knob.knob_features(k) for k, _ in self._dataset]
        cols = sorted({c for f in feats for c in f})
        if not cols:
            return
        x = np.array([[f.get(c, 0.0) for c in cols] for f in feats], dtype=float)
        y = np.array([lab for _, lab in self._dataset], dtype=float)
        model = CatBoostRegressor(
            iterations=self._iterations,
            depth=self.DEPTH,
            learning_rate=self.LEARNING_RATE,
            loss_function="RMSE",
            random_seed=self._seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,  # don't litter a catboost_info/ dir at the cwd
        )
        model.fit(x, y)
        self._cols, self._model = cols, model

    def score(self, knobs: dict) -> float:
        return self.mean_score(knobs)

    def mean_score(self, knobs: dict) -> float:
        if self._model is None:
            return 0.0
        f = knob.knob_features(knobs)
        x = np.array([[f.get(c, 0.0) for c in self._cols]], dtype=float)
        return float(self._model.predict(x)[0])

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
        accumulating from where it left off)."""
        p = cls()
        p._cols = obj.get("cols")
        p._dataset = [(dict(k), float(v)) for k, v in obj.get("dataset", [])]
        p._seen = int(obj.get("seen", len(p._dataset)))
        p._since_fit = int(obj.get("since_fit", 0))
        blob = obj.get("model")
        if blob:
            p._model = cls._model_from_b64(blob)
            p._first_fit_idx = 0  # loaded → warm, no cold warmup this run
        return p

    @classmethod
    def load(cls, *, seed: int = 0, path=None) -> CatBoostPrior:
        """The one global prior — warm from its JSON checkpoint if present, else
        fresh — bound so :meth:`checkpoint` saves it back. ``path`` defaults to
        ``config.prior_path()``."""
        from deplodock import config, storage  # noqa: PLC0415

        path = path or config.prior_path()
        obj = storage.read_json(path)
        p = cls.from_json(obj) if isinstance(obj, dict) else cls(seed=seed)
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
