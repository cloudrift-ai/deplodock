"""Golden-config evaluation harness — reconstructs a matmul shape's enumeration
and ranks it with a ``Prior``.

Ranking itself now lives in :mod:`deplodock.compiler.pipeline.search.prior`: the
hand-coded :class:`AnalyticPrior` (the cold-start linear model over
``features.knob_features``) and the learned ``CatBoostPrior`` are the ONE ranking
path. This module is just the offline *evaluation* glue — given a recorded golden
it enumerates the shape's candidate rows and reports the golden's rank under a
scorer (the ``AnalyticPrior`` by default; ``eval prior`` passes the learned one).
Used by ``deplodock eval analytic`` / ``eval prior`` and the prior diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.search.features import tile_signature


def _matmul_thread_gate(r: dict, dag, reduce_key: str) -> bool:
    """The heuristic-plausible band for thread-tier matmul tiles — pruned the
    enumeration so the cold ``AnalyticPrior`` couldn't argmin onto an *unbenched*
    degenerate tile."""
    raise NotImplementedError("tile lowering demolished — pending rebuild")


def _matmul_dag(M: int, N: int, K: int, dtype: str, ctx: Context):
    """Build a single ``(M, K) @ (K, N)`` matmul ``LoopOp`` and return its
    ``IterDag`` — the shape the per-family enumeration offers were composed over."""
    raise NotImplementedError("tile lowering demolished — pending rebuild")


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> tuple[list[dict], tuple[str, ...]]:
    """Reconstruct the planner's matmul enumeration for a shape + the knob tuple to
    match a golden on."""
    raise NotImplementedError("tile lowering demolished — pending rebuild")


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
    rows, _ = _enumerate(M, N, K, dtype, ctx)
    if not rows:
        return {}, None, 0
    if scorer is None:
        scorer = _analytic_scorer(M, N, K, ctx, dynamic=dynamic)
    # Match the recorded golden against the native candidate rows by schema-agnostic
    # structural signature (free slots + reduce decomp + atom + stage) — robust to either
    # spelling, so it still joins a legacy-recorded DB row against the native enumeration.
    want = tile_signature(golden_knobs) if golden_knobs else None
    gidx = next((i for i, r in enumerate(rows) if tile_signature(r) == want), None) if want else None
    scores = [scorer(r) for r in rows]
    best = max(range(len(rows)), key=scores.__getitem__)
    rank = sum(1 for s in scores if s > scores[gidx]) if gidx is not None else None
    return rows[best], rank, len(rows)


def pick_matmul(M: int, N: int, K: int, dtype: str, ctx: Context) -> dict:
    """Best knob row for an ``(M, K) @ (K, N)`` matmul under the analytic prior —
    no learned data, no measurements. Thin wrapper over :func:`evaluate_golden`
    (no golden to match). Returns ``{}`` if nothing enumerates."""
    return evaluate_golden(M, N, K, dtype, {}, ctx)[0]
