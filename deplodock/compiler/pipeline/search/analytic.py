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
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import enumerate_cartesian

# Occupancy reference fallback when a context carries no SM count (GPU-less hosts).
# RTX 5090 / sm_120 has 170 SMs; the golden set was measured there.
DEFAULT_SM_COUNT = 170

# Knobs the thread-tier / warp-tier enumeration decides — the projection a golden
# is matched on (STAGE / RING / WARPSPEC / NOATOMIC are stamped by later passes,
# not chosen here). FK is omitted from the thread set: the matmul enumeration
# always emits FK=1, so it never distinguishes a golden.
THREAD_KNOBS = ("BN", "BM", "FM", "FN", "BK", "SPLITK", "BR")
WARP_KNOBS = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> tuple[list[dict], tuple[str, ...]]:
    """Reconstruct the planner's enumeration for a matmul shape + the knob tuple to
    match a golden on. Rows are in enumeration (construction) order — ranking is the
    caller's ``scorer`` (the :class:`AnalyticPrior` by default), not an enumeration
    sort."""
    if dtype == "fp32":
        rows = enumerate_cartesian(E_M=M, E_N=N, E_K=K, ctx=ctx, priority_mode="matmul", m_axis_name="m", n_axis_name="n")
        return rows, THREAD_KNOBS

    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get({"fp16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(dtype, ""))
    if atom is None:
        return [], WARP_KNOBS
    rows = enumerate_cartesian(E_M=M, E_N=N, E_K=K, ctx=ctx, priority_mode=("matmul", "warp"), atoms=(atom,))
    rows = [r for r in rows if r["WM"] * r["WN"] != 1]
    return rows, WARP_KNOBS


def _analytic_scorer(M: int, N: int, K: int, ctx: Context) -> Callable[[dict], float]:
    """Default ranker for :func:`evaluate_golden` — the :class:`AnalyticPrior`
    folded into the higher-is-better convention ``evaluate_golden`` ranks on
    (``-latency``). Merges the shape / regime features the prior featurizes
    (``S_ext_*`` from ``M``/``N``/``K`` + ``H_*`` from the context) into each row,
    mirroring what the planner stamps in the live pipeline."""
    from deplodock.compiler.pipeline.search.prior import AnalyticPrior  # noqa: PLC0415

    ap = AnalyticPrior()
    base = {**ctx.features(), "S_ext_free_prod": float(M * N), "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
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
        scorer = _analytic_scorer(M, N, K, ctx)
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
