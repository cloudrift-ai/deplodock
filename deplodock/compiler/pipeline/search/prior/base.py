"""Base class for the inner-``tune`` learned prior + shared diagnostics.

A :class:`Prior` is fit *within one inner ``TuningSearch``* (per kernel,
discarded after the op is tuned) and consulted by the MCTS to rank unvisited
siblings. Subclasses implement the model — :meth:`fit` / :meth:`resample` /
:meth:`score` / :meth:`mean_score` / :attr:`fitted`; the base owns the bench
trajectory and the end-of-run sanity block.

Mode flags (set by the command, carried on the object so the search reads them
off the prior rather than via extra constructor args):

- ``active`` — :meth:`score` steers selection (``False`` = *shadow*: still fit +
  report, but score ``0`` so selection is pure UCB — the A/B baseline).
- ``acquisition`` — depth-2 PUCT (the prior enters the UCB *arithmetic* as a
  softmax policy) vs the depth-1 unvisited tiebreak; read by ``TuningSearch``.
- ``mode`` — display label for the summary block.

Training signal is **value-of-position**: real benches exist only at leaves, but
the prior ranks partial-knob siblings at every fork level, so the label for any
node is the max reward over its benched descendants (``SearchNode.best_reward``).
The search hands :meth:`fit` ``(knobs, log(best_reward))`` rows for leaves *and*
branches. Labels are non-stationary (they only rise), so the model **refits from
a tree snapshot** rather than streaming.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class Prior(ABC):
    """Abstract inner-tune prior. ``refit_every`` / ``min_rows`` gate how often
    the search calls :meth:`fit`; :meth:`resample` is called once per descent
    for a fresh draw (Thompson)."""

    def __init__(
        self, *, active: bool = True, acquisition: bool = False, mode: str = "online", refit_every: int = 4, min_rows: int = 6
    ) -> None:
        self.active = active
        self.acquisition = acquisition
        self.mode = mode
        self.refit_every = refit_every
        self.min_rows = min_rows
        # Bench trajectory for the end-of-run stats: (knobs, median, status) in
        # bench order. ``_first_fit_idx`` marks the warmup/post boundary (the
        # bench count at the first successful fit — before that score() is 0 =
        # pure UCB); subclasses set it in :meth:`fit` via :meth:`_note_fit`.
        self.trajectory: list[tuple[dict, float, str]] = []
        self._first_fit_idx: int | None = None

    @property
    @abstractmethod
    def fitted(self) -> bool: ...

    @abstractmethod
    def fit(self, rows: list[tuple[dict, float]]) -> None:
        """Refit the model from value-of-position rows ``(knobs, label)``."""

    @abstractmethod
    def resample(self) -> None:
        """Draw a fresh sample for ranking — one per descent."""

    @abstractmethod
    def score(self, knobs: dict) -> float:
        """Sampled prediction for ranking. ``0.0`` until the first fit, or
        always in shadow mode (``active=False``)."""

    @abstractmethod
    def mean_score(self, knobs: dict) -> float:
        """Posterior-mean prediction (no sampling noise) — for calibration."""

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        """Append a benched leaf to the trajectory (for the summary stats)."""
        self.trajectory.append((dict(knobs), median, status))

    def _note_fit(self) -> None:
        """Subclasses call this at the end of a successful :meth:`fit` to stamp
        the warmup/post boundary on the first fit."""
        if self._first_fit_idx is None:
            self._first_fit_idx = len(self.trajectory)

    # --- end-of-run sanity stats ------------------------------------------

    def summary(self, label: str) -> str:
        """A compact stats block judging the prior by *which picks* it made —
        silly-pick rate before vs after the model warmed up, and the prior's
        self-calibration on its post-warmup picks."""
        oks = [(k, us) for k, us, st in self.trajectory if st == "ok" and us > 0]
        n = len(self.trajectory)
        if not oks:
            return f"[prior] {label} — {n} benches, no ok measurements"
        best = min(us for _, us in oks)
        best_idx = next(i for i, (_, us, st) in enumerate(self.trajectory) if st == "ok" and us == best)
        warm = self._first_fit_idx if self._first_fit_idx is not None else n
        lines = [
            f"[prior] {label} ({self.mode}) — {n} benches (warmup {warm} / post {n - warm})",
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
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
