"""Base class for the learned tuning prior + shared dataset / diagnostics.

A :class:`Prior` is **one global model** consulted by the MCTS to rank candidates
via PUCT (``tune``) and by the greedy driver to pick knobs (``compile`` / ``run``).
Unlike the earlier online-refit-every-few-benches design, it is trained in
**batches**: every tuned op's value-of-position rows are streamed into a bounded
dataset (:meth:`add_rows`), and the model is refit (:meth:`maybe_refit`) on a
**dataset-size-tiered cadence** (:data:`REFIT_SCHEDULE`) — frequently while the
model is data-poor, then progressively less often as returns diminish: every
100 rows up to 1k, every 1k up to 10k, every 10k from there on. The dataset is
capped at :data:`MAX_ROWS` by **reservoir sampling** (Algorithm R) — a uniform
random sample of every row ever seen, across runs — so a long-lived global prior
neither grows without bound nor over-weights recent ops. The model is therefore
*fixed during a single op's search* (the global prior already generalizes to a
new op — see :class:`CatBoostPrior`), and exploration comes from PUCT's own term,
not per-bench refitting.

Training signal is **value-of-position**: real benches exist only at leaves, but
the prior ranks partial-knob siblings at every fork level, so the label for any
node is the max reward over its benched descendants (``SearchNode.best_reward``).
The search hands :meth:`add_rows` ``(knobs, log(best_reward))`` rows for leaves
*and* branches. :meth:`score` / :meth:`mean_score` return ``0`` until the first
fit, so a cold prior gives a uniform PUCT policy (exploration via the PUCT term).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

# Dataset-size-tiered refit cadence: ``(size_threshold, interval)`` bands, ordered
# by threshold. The refit interval is the first band whose threshold the current
# dataset size is still below, else the last band's interval — so refits coarsen
# as data accumulates: every 100 rows up to 1k, every 1k up to 10k, every 10k
# from 10k on (and beyond the cap, as the reservoir churns).
REFIT_SCHEDULE = ((1_000, 100), (10_000, 1_000), (100_000, 10_000))
# Hard cap on the training set — reservoir-sampled down to a uniform sample of all
# rows ever seen. Bounds memory + fit time for a long-lived global prior.
MAX_ROWS = 100_000


class Prior(ABC):
    """Abstract global tuning prior over a bounded, reservoir-sampled dataset.

    ``max_rows`` / ``min_rows`` are instance params (defaulting to the module
    constants); ``refit_every`` overrides the tiered :data:`REFIT_SCHEDULE` with a
    single fixed interval (used by tests)."""

    def __init__(self, *, seed: int = 0, min_rows: int = 50, refit_every: int | None = None, max_rows: int = MAX_ROWS) -> None:
        self._seed = seed
        self.min_rows = min_rows
        self.refit_every = refit_every  # None → tiered REFIT_SCHEDULE
        self.max_rows = max_rows
        self._rng = np.random.default_rng(seed)
        # The bounded training set + reservoir bookkeeping. ``_seen`` is the total
        # rows ever offered (for Algorithm R); ``_since_fit`` is rows added since
        # the last fit (the refit trigger). Both persist in the checkpoint.
        self._dataset: list[tuple[dict, float]] = []
        self._seen = 0
        self._since_fit = 0
        # Bench trajectory for the end-of-run stats: (knobs, median, status) in
        # bench order. ``_first_fit_idx`` marks the warmup/post boundary.
        self.trajectory: list[tuple[dict, float, str]] = []
        self._first_fit_idx: int | None = None
        # Checkpoint binding — set by ``load``; lets :meth:`checkpoint` persist
        # without the caller threading a path.
        self._path = None

    @property
    @abstractmethod
    def fitted(self) -> bool: ...

    @abstractmethod
    def fit(self) -> None:
        """Refit the model on the current :attr:`_dataset`."""

    @abstractmethod
    def score(self, knobs: dict) -> float:
        """Prediction for ranking a candidate. ``0.0`` until the first fit (cold
        prior → uniform PUCT policy)."""

    @abstractmethod
    def mean_score(self, knobs: dict) -> float:
        """Prediction for the greedy argmax + calibration (same as :meth:`score`
        for a deterministic model)."""

    # --- dataset + batched refit ------------------------------------------

    def add_rows(self, rows: list[tuple[dict, float]]) -> None:
        """Stream value-of-position rows into the bounded dataset via reservoir
        sampling (Algorithm R): under the cap they append; at the cap each new row
        replaces a uniformly-random existing one with the correct probability, so
        ``_dataset`` stays a uniform sample of every row ever seen."""
        for k, v in rows:
            self._seen += 1
            self._since_fit += 1
            row = (dict(k), float(v))
            if len(self._dataset) < self.max_rows:
                self._dataset.append(row)
            else:
                j = int(self._rng.integers(self._seen))
                if j < self.max_rows:
                    self._dataset[j] = row

    def _refit_interval(self) -> int:
        """Rows that must accumulate before the next refit — the fixed
        ``refit_every`` override if set, else the :data:`REFIT_SCHEDULE` band for
        the current dataset size (coarsening as data grows)."""
        if self.refit_every is not None:
            return self.refit_every
        n = len(self._dataset)
        for threshold, interval in REFIT_SCHEDULE:
            if n < threshold:
                return interval
        return REFIT_SCHEDULE[-1][1]

    def maybe_refit(self, *, force: bool = False) -> bool:
        """Refit (and stamp the warmup boundary) iff enough rows have streamed in
        since the last fit (:meth:`_refit_interval`) and the dataset clears
        ``min_rows``. ``force`` (used at end-of-run) bypasses the interval so a
        small tune that never crossed a tier threshold still gets a fitted model —
        but only when there's new data or no model yet, never a redundant re-fit.
        Returns whether a fit happened (caller checkpoints on ``True``)."""
        if len(self._dataset) < self.min_rows:
            return False
        if force:
            if self.fitted and self._since_fit == 0:
                return False  # nothing new since the last fit
        elif self._since_fit < self._refit_interval():
            return False
        self.fit()
        self._since_fit = 0
        self._note_fit()
        return True

    def record_bench(self, knobs: dict, median: float, status: str) -> None:
        """Append a benched leaf to the trajectory (for the summary stats)."""
        self.trajectory.append((dict(knobs), median, status))

    def checkpoint(self) -> None:
        """Persist this prior to its bound JSON file (a no-op if unbound or there's
        nothing to save). The binding is set by ``load``."""
        if self._path is None:
            return
        obj = self.to_json()
        if obj is None:
            return
        from deplodock import storage  # noqa: PLC0415

        storage.write_json(self._path, obj)

    def to_json(self) -> dict | None:
        """Serialize the model + dataset for persistence (``None`` when there's
        nothing yet). Override in subclasses; see :meth:`CatBoostPrior.to_json`."""
        return None

    def _note_fit(self) -> None:
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
            f"[prior] {label} — {n} benches (warmup {warm} / post {n - warm}), dataset {len(self._dataset)}/{self.max_rows}",
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
        """Spearman ρ between the prior's prediction and the measured reward
        (−log us) over post-warmup ok picks (``scipy.stats.spearmanr``)."""
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
        from scipy.stats import spearmanr  # noqa: PLC0415

        rho = float(spearmanr(pred, reward).statistic)
        return None if math.isnan(rho) else rho
