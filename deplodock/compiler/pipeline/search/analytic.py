"""Golden-config evaluation harness — reconstructs a matmul shape's enumeration
and ranks it with a ``Prior``.

Ranking itself now lives in :mod:`deplodock.compiler.pipeline.search.prior`: the
hand-coded :class:`AnalyticPrior` (the cold-start linear model over
``knob.knob_features``) and the learned ``CatBoostPrior`` are the ONE ranking
path. This module is just the offline *evaluation* glue — given a recorded golden
it enumerates the shape's candidate rows and reports the golden's rank under a
scorer (the ``AnalyticPrior`` by default; ``eval prior`` passes the learned one).
Used by ``deplodock eval analytic`` / ``eval prior`` and the prior diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.context import Context
from deplodock.gpu import DEFAULT_GPU

# Occupancy reference fallback when a context carries no SM count (GPU-less hosts) —
# the default card's memorized SM count (RTX 5090 / sm_120 = 170; the golden set was
# measured there). Single source: the common GPU registry (:mod:`deplodock.gpu`).
DEFAULT_SM_COUNT = DEFAULT_GPU.sm_count

# Knobs the thread-tier / warp-tier enumeration decides — the projection a golden
# is matched on (STAGE / RING / WARPSPEC / NOATOMIC are stamped by later passes,
# not chosen here). FK is omitted from the thread set: the matmul enumeration
# always emits FK=1, so it never distinguishes a golden.
THREAD_KNOBS = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
WARP_KNOBS = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> tuple[list[dict], tuple[str, ...]]:  # noqa: ARG001
    """Golden-eval enumeration — REMOVED with the cartesian enumerator (invalid
    under the move-composer architecture). Returns no rows so the golden-rank
    diagnostics degrade to "nothing enumerates" until the eval harness is rebuilt
    over the move tree (``plans/tile-ir-block-dag.md``)."""
    return [], (THREAD_KNOBS if dtype == "fp32" else WARP_KNOBS)


def _analytic_scorer(M: int, N: int, K: int, ctx: Context, *, dynamic: bool = False) -> Callable[[dict], float]:
    """Default ranker for :func:`evaluate_golden` — the :class:`AnalyticPrior`
    folded into the higher-is-better convention ``evaluate_golden`` ranks on
    (``-latency``). Merges the shape / regime features the prior featurizes
    (``S_ext_*`` from ``M``/``N``/``K`` + ``H_*`` from the context) into each row,
    mirroring what the planner stamps in the live pipeline. ``dynamic`` mirrors
    the 992 stamp for a symbolic-M shape: M drops out of the free-dim product and
    ``S_ext_n_symbolic_axis`` is set — the flag the prior selects its masked-tier
    weight set on."""
    from deplodock.compiler.pipeline.search.prior import AnalyticPrior  # noqa: PLC0415

    ap = AnalyticPrior()
    free = float(N) if dynamic else float(M * N)
    base = {**ctx.features(), "S_ext_free_prod": free, "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    if dynamic:
        base["S_ext_n_symbolic_axis"] = 1.0
    return lambda r: -ap.score({**base, **r})


def evaluate_golden(
    M: int,
    N: int,
    K: int,
    dtype: str,
    golden_knobs: dict,
    ctx: Context,
    sm_count: int | None = None,  # noqa: ARG001 — kept for call-site compat; the analytic prior reads ctx.sm_count
    scorer: Callable[[dict], float] | None = None,
    dynamic: bool = False,
) -> tuple[dict, int | None, int]:
    """Score a matmul shape's full enumeration and return ``(pick, rank, pool)``:
    the argmax pick (higher score = better), the recorded golden's 0-based rank in
    the scored order (``None`` if the golden isn't in the enumeration — pin / dtype
    mismatch), and the enumeration size. The rank — not whether the #1 pick equals
    the golden — is the metric that matters: it's where the tuner's patience budget
    has to reach. ``scorer`` (``row → float``, higher better) defaults to the
    :class:`AnalyticPrior` (negated latency); the learned-prior diagnostics pass
    ``-prior.mean_score`` instead. Returns ``({}, None, 0)`` if nothing
    enumerates."""
    rows, match_keys = _enumerate(M, N, K, dtype, ctx)
    if not rows:
        return {}, None, 0
    if scorer is None:
        scorer = _analytic_scorer(M, N, K, ctx, dynamic=dynamic)
    want = tuple(golden_knobs.get(k) for k in match_keys)
    gidx = next((i for i, r in enumerate(rows) if tuple(r.get(k) for k in match_keys) == want), None)
    scores = [scorer(r) for r in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank = sum(1 for s in scores if s > scores[gidx]) if gidx is not None else None
    return rows[best], rank, len(rows)


def pick_matmul(M: int, N: int, K: int, dtype: str, ctx: Context, sm_count: int | None = None) -> dict:
    """Best knob row for an ``(M, K) @ (K, N)`` matmul under the analytic prior —
    no learned data, no measurements. Thin wrapper over :func:`evaluate_golden`
    (no golden to match). Returns ``{}`` if nothing enumerates."""
    return evaluate_golden(M, N, K, dtype, {}, ctx, sm_count)[0]
