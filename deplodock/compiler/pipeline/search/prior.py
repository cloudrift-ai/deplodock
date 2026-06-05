"""Online learned prior for the inner ``tune`` MCTS — a Bayesian ridge model
fit *within a single tune run*, in memory, discarded at the end.

Replaces the hand-written ``TileOp.score`` tiebreaker (nuked from selection)
with a model that learns which knobs are good from the run's own benches. The
search consults it only as the unvisited-sibling tiebreaker (no UCB-arithmetic
change), and it is Thompson-sampled so exploration stays honest.

Training signal — leaf vs non-leaf
-----------------------------------
Real benches exist only for **leaves** (fully-specified TileParams). But the
MCTS ranks **partial-knob siblings** at every fork level, so the label for any
node is its **value-of-position = max reward over benched descendants** — which
``SearchTree.record_terminal`` already maintains as ``SearchNode.best_reward``.
The search hands us ``(knobs, best_reward)`` rows for every node with a benched
descendant (leaves *and* branches); we regress ``y = log(best_reward)``
(= −log median latency, so a few catastrophic configs don't capture the fit)
on :func:`knob.knob_features`.

The labels are non-stationary (``best_reward`` only rises), so we **refit from a
tree snapshot** each time rather than a streaming update — at this scale (≤
hundreds of nodes, d ~ tens) the ridge solve is trivial and re-reading the
labels sidesteps the moving-target problem.

Model
-----
Standardized features, ``y`` centered. Ridge posterior
``θ̄ = (XᵀX + λI)⁻¹ Xᵀy_c``, ``Σ = σ²(XᵀX + λI)⁻¹`` (σ² from residuals).
Thompson: one draw ``θ̃ ~ N(θ̄, Σ)`` per descent (``resample``), scores are
``θ̃·φ`` — scale-free ranking, absolute value irrelevant.

Missing knobs (a partial state lacks deeper knobs) are 0-filled then
standardized, which maps "absent" to a fixed coordinate distinct from any real
(positive) knob value — an implicit presence encoding. Within one sibling set
the absent pattern is constant, so it never perturbs the ranking that matters.
"""

from __future__ import annotations

import math

import numpy as np

from deplodock.compiler.pipeline import knob


class OnlinePrior:
    """In-memory Bayesian-ridge prior over knob features, Thompson-sampled.

    Lifecycle = one inner :class:`TuningSearch` (one kernel). ``refit_every`` /
    ``min_rows`` gate how often the search calls :meth:`fit`; ``resample`` is
    called once per descent for a fresh Thompson draw."""

    def __init__(self, *, seed: int = 0, ridge: float = 1.0, refit_every: int = 4, min_rows: int = 6, active: bool = True) -> None:
        # ``active=False`` (shadow mode): the model still fits and records the
        # trajectory for the end-of-run stats, but :meth:`score` returns 0 so
        # selection is pure UCB — the baseline arm of the online-vs-UCB A/B.
        self.active = active
        self.refit_every = refit_every
        self.min_rows = min_rows
        self._ridge = ridge
        self._rng = np.random.default_rng(seed)
        self._cols: list[str] | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._theta: np.ndarray | None = None  # posterior mean
        self._cov_L: np.ndarray | None = None  # Cholesky of Σ
        self._y_mean = 0.0
        self._theta_sample: np.ndarray | None = None  # current Thompson draw
        # Bench trajectory for the end-of-run sanity stats: (knobs, median, status)
        # in bench order. ``_first_fit_idx`` marks the warmup/post boundary (the
        # bench count at the first successful fit — before that the prior is 0 =
        # pure UCB).
        self.trajectory: list[tuple[dict, float, str]] = []
        self._first_fit_idx: int | None = None

    @property
    def fitted(self) -> bool:
        return self._theta is not None

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        """Append a benched leaf to the trajectory (for the summary stats)."""
        self.trajectory.append((dict(knobs), median, status))

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
        if self._first_fit_idx is None:
            self._first_fit_idx = len(self.trajectory)
        self.resample()

    def resample(self) -> None:
        """Draw a fresh Thompson sample ``θ̃ ~ N(θ̄, Σ)`` — one per descent."""
        if self._theta is None or self._cov_L is None:
            self._theta_sample = None
            return
        z = self._rng.standard_normal(self._theta.shape[0])
        self._theta_sample = self._theta + self._cov_L @ z

    def score(self, knobs: dict) -> float:
        """Thompson-sampled prediction for ranking. ``0.0`` until the first
        fit (warmup = pure UCB), or always in shadow mode (``active=False``)."""
        if not self.active or self._theta_sample is None:
            return 0.0
        return float(self._vec(knobs) @ self._theta_sample + self._y_mean)

    def mean_score(self, knobs: dict) -> float:
        """Posterior-mean prediction (no Thompson noise) — for calibration."""
        if self._theta is None:
            return 0.0
        return float(self._vec(knobs) @ self._theta + self._y_mean)

    def _vec(self, knobs: dict) -> np.ndarray:
        assert self._cols is not None and self._mean is not None and self._std is not None
        f = knob.knob_features(knobs)
        x = np.array([f.get(c, 0.0) for c in self._cols], dtype=float)
        return (x - self._mean) / self._std

    # --- end-of-run sanity stats ------------------------------------------

    def summary(self, label: str) -> str:
        """A compact stats block judging the prior by *which picks* it made —
        silly-pick rate before vs after the model warmed up (should drop), and
        the prior's self-calibration on its post-warmup picks."""
        oks = [(k, us) for k, us, st in self.trajectory if st == "ok" and us > 0]
        n = len(self.trajectory)
        if not oks:
            return f"[prior] {label} — {n} benches, no ok measurements"
        best = min(us for _, us in oks)
        best_idx = next(i for i, (_, us, st) in enumerate(self.trajectory) if st == "ok" and us == best)
        warm = self._first_fit_idx if self._first_fit_idx is not None else n
        mode = "online" if self.active else "shadow/UCB"
        lines = [
            f"[prior] {label} ({mode}) — {n} benches (warmup {warm} / post {n - warm})",
            f"  best:        {best:.3f} us @ bench #{best_idx}",
            f"  silly (>=2x best):  {self._silly(0, warm, 2 * best)}   {self._silly(warm, n, 2 * best, post=True)}",
        ]
        calib = self._calibration(warm, n)
        if calib is not None:
            lines.append(f"  calibration (post Spearman pred vs reward): {calib:+.2f}")
        return "\n".join(lines)

    def _silly(self, lo: int, hi: int, thresh: float, *, post: bool = False) -> str:
        tag = "post" if post else "warmup"
        oks = [us for _, us, st in self.trajectory[lo:hi] if st == "ok" and us > 0]
        if not oks:
            return f"{tag} 0/0"
        silly = sum(1 for us in oks if us >= thresh)
        return f"{tag} {silly}/{len(oks)} ({100 * silly / len(oks):.0f}%)"

    def _calibration(self, lo: int, hi: int) -> float | None:
        """Spearman ρ between the prior's posterior-mean prediction and the
        measured reward (−log us) over post-warmup ok picks."""
        if not self.fitted:
            return None
        pred, reward = [], []
        for knobs, us, st in self.trajectory[lo:hi]:
            if st != "ok" or us <= 0:
                continue
            pred.append(self.mean_score(knobs))
            reward.append(-math.log(us))
        if len(pred) < 3:
            return None
        return _spearman(np.array(pred), np.array(reward))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation = Pearson on ranks. Returns 0.0 when either
    side is constant (rank std 0)."""
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(((ra - ra.mean()) @ (rb - rb.mean())) / (len(a) * sa * sb))


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # Average tied groups so equal values get equal rank.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
