"""Offline model bake-off for the learned tuning prior.

Compares candidate regression models on the metrics that matter for the prior's
two consumers, over a grouped dataset of accumulated tune rows (see
``prior_data.py`` / ``scripts/gen_prior_data.sh``):

  argmax ratio   — fit, then for each op pick the model's argmax over that op's
                   *leaf* configs; ratio = us_picked / us_best (1.00 = optimal).
                   What GREEDY compile cares about. Reported IN-SAMPLE (train on
                   all ops) and LEAVE-ONE-OP-OUT (train on the others — does it
                   generalize to an untuned op?).
  spearman ρ     — median per-op rank correlation over all rows. What tune's
                   PUCT cares about.
  hazard         — greedy-corner safety: fraction of ops where pushing some
                   single knob to an UNSEEN extreme (0 or 2x the max) is scored
                   above the best measured config — i.e. the model would lure
                   greedy off the cliff (this is the failure the linear prior hit:
                   argmax extrapolated to BR=1, 4us -> 232us / invalid kernels).

Research-only tooling — needs the comparison libs (not project deps):
    pip install scikit-learn xgboost lightgbm catboost

Usage:
    python scripts/prior_bakeoff.py [PRIOR_JSON ...]   # default: the bakeoff cache
"""

from __future__ import annotations

import math
import sys

import numpy as np
import prior_data
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deplodock.compiler.pipeline import knob

# Optional third-party gradient boosters — added to the bake-off when installed.
_OPTIONAL = {}
try:
    from xgboost import XGBRegressor

    _OPTIONAL["xgb"] = lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=0)
except ImportError:
    pass
try:
    from lightgbm import LGBMRegressor

    _OPTIONAL["lgbm"] = lambda: LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05, n_jobs=-1, random_state=0, verbose=-1)
except ImportError:
    pass
try:
    from catboost import CatBoostRegressor

    _OPTIONAL["catboost"] = lambda: CatBoostRegressor(
        iterations=400, depth=6, learning_rate=0.05, random_seed=0, verbose=False, allow_writing_files=False
    )
except ImportError:
    pass

MODELS = ["linear", "logquad", "rf", "hgb", "gp", *_OPTIONAL]
_GP_CAP = 500  # GP is O(n^3): subsample training to this many rows


def augment(feat: dict) -> dict:
    """log-quadratic basis: keep base features and, for every tunable feature, add
    ``log2(1+|v|)`` and its square, so a linear-in-weights model can fit a concave
    sweet spot (interior optimum) instead of running off to a monotone corner."""
    out = dict(feat)
    for k, v in feat.items():
        if k.startswith(("S_", "H_")):
            continue
        lv = math.log2(1.0 + abs(v))
        out[f"{k}__log"], out[f"{k}__log2"] = lv, lv * lv
    return out


def _feats(rows, aug):
    return [augment(knob.knob_features(k)) if aug else knob.knob_features(k) for k, _ in rows]


def _matrix(feat_dicts, cols):
    return np.array([[f.get(c, 0.0) for c in cols] for f in feat_dicts], dtype=float)


def _model(name):
    if name in ("linear", "logquad"):
        return make_pipeline(StandardScaler(), BayesianRidge())
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=31, random_state=0)
    if name == "gp":
        k = ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(0.5)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0))
    if name in _OPTIONAL:
        return _OPTIONAL[name]()
    raise ValueError(name)


def _fit(name, rows_train):
    aug = name == "logquad"
    feats = _feats(rows_train, aug)
    cols = sorted({c for f in feats for c in f})
    X = _matrix(feats, cols)
    y = np.array([lab for _, lab in rows_train], dtype=float)
    if name == "gp" and len(X) > _GP_CAP:
        idx = np.random.default_rng(0).choice(len(X), _GP_CAP, replace=False)
        X, y = X[idx], y[idx]
    m = _model(name).fit(X, y)
    return m, cols, aug


def _predict(m, cols, aug, feat_dicts):
    return m.predict(_matrix([augment(f) if aug else f for f in feat_dicts], cols))


def _score_op(rows, pred, ratios, rhos):
    labels = np.array([lab for _, lab in rows])
    rho = spearmanr(pred, labels).statistic
    if not math.isnan(rho):
        rhos.append(rho)
    kmax = max(prior_data.n_tunable(k) for k, _ in rows)
    leaf = [i for i, (k, _) in enumerate(rows) if prior_data.n_tunable(k) == kmax]
    if len(leaf) >= 2:
        best = max(labels[i] for i in leaf)
        picked = max(leaf, key=lambda i: pred[i])
        ratios.append(math.exp(best - labels[picked]))


def evaluate(name, groups, loo):
    """(mean argmax ratio, worst argmax ratio, median per-op Spearman). In-sample
    fits ONCE (every op shares the same all-rows training set); LOO refits per
    held-out op."""
    ratios, rhos = [], []
    all_rows = [r for g in groups.values() for r in g]
    if not loo:
        m, cols, aug = _fit(name, all_rows)
        for rows in groups.values():
            _score_op(rows, _predict(m, cols, aug, [knob.knob_features(k) for k, _ in rows]), ratios, rhos)
    else:
        for sig, rows in groups.items():
            train = [r for s, g in groups.items() if s != sig for r in g]
            if not train:
                continue
            m, cols, aug = _fit(name, train)
            _score_op(rows, _predict(m, cols, aug, [knob.knob_features(k) for k, _ in rows]), ratios, rhos)
    return float(np.mean(ratios)), float(np.max(ratios)), float(np.median(rhos))


def grid_safety(name, groups, n=4000, seed=0):
    """Greedy-faithful safety: train in-sample, then for each op sample a realistic
    candidate grid (each knob drawn from its SEEN-in-leaves values — on-manifold,
    no invented extremes), take the model's argmax, and report the TRUE latency
    ratio of the nearest MEASURED leaf to that pick. ~1.0 = the model's favorite
    region is genuinely good; large = it prefers a region that actually benches
    badly (what sinks greedy). Returns (mean, worst)."""
    rng = np.random.default_rng(seed)
    all_rows = [r for g in groups.values() for r in g]
    m, cols, aug = _fit(name, all_rows)
    base = [c for c in cols if not c.endswith(("__log", "__log2"))]
    tunable = [c for c in base if not c.startswith(("S_", "H_"))]
    ratios = []
    for rows in groups.values():
        kmax = max(prior_data.n_tunable(k) for k, _ in rows)
        leaf = [(knob.knob_features(k), lab) for k, lab in rows if prior_data.n_tunable(k) == kmax]
        if len(leaf) < 4:
            continue
        feats = [f for f, _ in leaf]
        labs = np.array([lab for _, lab in leaf])
        ctx = {c: feats[int(labs.argmax())].get(c, 0.0) for c in base if c not in tunable}  # S_/H_ fixed per op
        seen = {c: sorted({f.get(c, 0.0) for f in feats}) for c in tunable}
        cand = [dict(ctx, **{c: seen[c][rng.integers(len(seen[c]))] for c in tunable}) for _ in range(n)]
        gstar = cand[int(np.argmax(_predict(m, cols, aug, cand)))]
        L = np.array([[f.get(c, 0.0) for c in base] for f in feats])
        std = L.std(0)
        std[std == 0] = 1.0
        gv = np.array([gstar.get(c, 0.0) for c in base])
        nn = int(np.argmin((((L - gv) / std) ** 2).sum(1)))  # nearest measured leaf
        ratios.append(math.exp(labs.max() - labs[nn]))
    return float(np.mean(ratios)), float(np.max(ratios))


def main(argv):
    paths = argv or ["/tmp/bakeoff/prior.json"]
    rows = prior_data.load_rows(paths)
    groups = prior_data.group_by_op(rows)
    leafy = sum(1 for g in groups.values() if len(g) >= 2)
    print(f"dataset: {len(rows)} rows, {len(groups)} op-structures ({leafy} usable)\n")
    rows_out = []
    for i, name in enumerate(MODELS, 1):
        try:
            print(f"[{i}/{len(MODELS)}] {name:8} fitting:", end="", flush=True)
            im, iw, rho = evaluate(name, groups, loo=False)
            print(" in-sample✓", end="", flush=True)
            lm, lw, _ = evaluate(name, groups, loo=True)
            print(" loo✓", end="", flush=True)
            sm, sw = grid_safety(name, groups)
            print(" grid-safety✓", flush=True)
            rows_out.append(f"{name:8} | {im:8.2f} {iw:8.1f} | {lm:8.2f} {lw:8.1f} |   {rho:+.2f}   | {sm:5.2f} {sw:5.1f}")
        except Exception as e:  # noqa: BLE001
            rows_out.append(f"{name:8} | ERROR: {type(e).__name__}: {e}")

    print()
    print(f"{'model':8} | {'in-sample':>17} | {'leave-1-op-out':>17} | spearman | grid-safety")
    print(f"{'':8} | {'mean':>8} {'worst':>8} | {'mean':>8} {'worst':>8} | (median) | mean  worst")
    print("-" * 80)
    print("\n".join(rows_out))


if __name__ == "__main__":
    main(sys.argv[1:])
