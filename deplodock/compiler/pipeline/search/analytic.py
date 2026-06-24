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


def _matmul_thread_gate(r: dict) -> bool:
    """The heuristic-plausible band for thread-tier matmul tiles, distilled from
    the measured ``GOLDEN_CONFIGS`` (every recorded golden satisfies it). Used to
    prune the enumeration so the cold ``AnalyticPrior`` can't argmin onto an
    *unbenched* degenerate tile (e.g. ``BN=8`` from the now-default wide candidate
    menu) and override the golden-shaped option. Coalesced wide inner axis, short
    outer axis, large K-chunk, light split-K, clean output-column width. The caller
    falls back to the ungated set when this empties (tiny / unusual shapes with no
    in-band candidate), so it only ever *narrows*, never strands a shape."""
    bn, bm = r["BN"], r["BM"]
    threads = bn * bm
    tile_n = bn * r["FN"]
    return (
        16 <= bn <= 64
        and 8 <= bm <= 16
        and bn >= bm
        and r["BK"] >= 32
        and r["SPLITK"] <= 2
        and threads in (128, 256, 512, 1024)
        and tile_n in (32, 64, 128)
    )


def _matmul_dag(M: int, N: int, K: int, dtype: str, ctx: Context):
    """Build a single ``(M, K) @ (K, N)`` matmul ``LoopOp`` and return its
    ``IterDag`` — the shape the per-family enumeration offers are composed over.

    Reuses the real frontend → loop lowering (``LOOP_PASSES``, option-0 greedy
    resolve, no GPU) so the dag's axes / extents / carrier match what the live
    pipeline tiles. Returns ``None`` if nothing lowers (a degenerate shape)."""
    from deplodock.compiler import dtype as _dt  # noqa: PLC0415
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from deplodock.compiler.pipeline.fork import Fork  # noqa: PLC0415
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag  # noqa: PLC0415
    from deplodock.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    dt = _dt.get({"fp32": "f32", "fp16": "f16", "bf16": "bf16"}.get(dtype, dtype))
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dt), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dt), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N), dt), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]

    def _option0(fp):
        o = fp.options[0]
        while isinstance(o, Fork) and not o.is_leaf:
            o = o.expand()[0]
        return o

    terminal, _ = Run(pipeline=Pipeline.build(LOOP_PASSES), ctx=ctx).resolve(g, _option0)
    loops = [n.op for n in terminal.nodes.values() if isinstance(n.op, LoopOp)]
    return iter_dag(loops[0]) if loops else None


def _enumerate(M: int, N: int, K: int, dtype: str, ctx: Context) -> tuple[list[dict], tuple[str, ...]]:
    """Reconstruct the planner's matmul enumeration for a shape + the knob tuple to
    match a golden on. Composes the per-family enumeration offers
    (``enumeration/_moves``) into the cartesian of legal knob rows — the thread
    tier for ``fp32`` (reduce K-tiling × free thread tile × register tile), the
    warp tier for ``fp16``/``bf16`` (atom × warp counts × register cells × K-chunk).
    Rows are in enumeration (construction) order; ranking is the caller's
    ``scorer`` (the :class:`AnalyticPrior` by default), not an enumeration sort."""
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _moves  # noqa: PLC0415

    dag = _matmul_dag(M, N, K, dtype, ctx)
    if dag is None:
        return [], (THREAD_KNOBS if dtype == "fp32" else WARP_KNOBS)
    budget = _moves.Budget()

    if dtype == "fp32":
        rows: list[dict] = []
        threads = _moves.thread_offers(dag, budget)
        for bk, fk, sk in _moves.reduce_offers(dag):
            red = _moves.reduce_knobs((bk, fk, sk))
            regs = _moves.reduce_reg_offers(dag, budget, fk)
            for t in threads:
                thr = _moves.thread_knobs(dag, t)
                for r in regs:
                    # BR=1 is the scalar-tier seal (110_seal_scalar_tier) — a SEMIRING
                    # matmul has no cooperative-K lane; stamp it so the row carries the
                    # complete THREAD_KNOBS projection a golden is matched on.
                    rows.append({**red, **thr, **_moves.reg_knobs(dag, r), "BR": 1})
        # Narrow to the golden-plausible band so the cold prior can't argmin onto a
        # degenerate tile; fall back to the ungated set if no in-band candidate exists.
        gated = [r for r in rows if _matmul_thread_gate(r)]
        return (gated or rows), THREAD_KNOBS

    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get({"fp16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(dtype, ""))
    if atom is None:
        return [], WARP_KNOBS
    rows = []
    bks = _moves.warp_bk_offers(dag, atom)
    regs = _moves.warp_reg_offers(atom)
    for wm, wn in _moves.warp_offers(atom):  # wm·wn ≥ 2 already (single-warp mma pruned)
        geom = _moves.warp_geom_knobs(wm, wn)
        for fm, fn in regs:
            reg = _moves.warp_reg_knobs(fm, fn)
            for bk in bks:
                rows.append({**geom, **reg, **_moves.warp_bk_knobs(atom, bk)})
    return rows, WARP_KNOBS


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
