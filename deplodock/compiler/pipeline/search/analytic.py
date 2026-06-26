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
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam

# Free-axis knobs the thread-tier / warp-tier enumeration decides — the projection a
# golden is matched on (the reduce decomposition is now the native ``REDUCE@<axis>``
# value, dynamic per kernel, so it is matched separately, not as a static column).
THREAD_KNOBS = ("BN", "BM", "FM", "FN")
WARP_KNOBS = ("WN", "WM", "FM", "FN", "MMA")


def _matmul_thread_gate(r: dict, dag, reduce_key: str) -> bool:
    """The heuristic-plausible band for thread-tier matmul tiles, distilled from
    the measured ``GOLDEN_CONFIGS`` (every recorded golden satisfies it). Used to
    prune the enumeration so the cold ``AnalyticPrior`` can't argmin onto an
    *unbenched* degenerate tile and override the golden-shaped option. Coalesced
    wide inner axis, short outer axis, large K-chunk, light split-K, clean
    output-column width. The caller falls back to the ungated set when this empties
    (tiny / unusual shapes with no in-band candidate), so it only ever *narrows*."""
    bn, fn = fam.dec_split(r[fam.split_key(dag.inner_n.axis.name)])
    bm, _ = fam.dec_split(r[fam.split_key(dag.outer_m.axis.name)])
    threads = bn * bm
    tile_n = bn * fn
    decomp = fam.dec_reduce(r[reduce_key])
    return (
        16 <= bn <= 64
        and 8 <= bm <= 16
        and bn >= bm
        and decomp.serial >= 32
        and decomp.cta <= 2
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

    rk = fam.reduce_key(dag.k_node.loop.axis.name)
    if dtype == "fp32":
        rows: list[dict] = []
        threads = _moves.thread_offers(dag, budget)
        for bk, fk, sk in _moves.reduce_offers(dag):
            red = _moves.reduce_knobs(dag, (bk, fk, sk))
            regs = _moves.reduce_reg_offers(dag, budget, fk)
            for t in threads:
                base = {**red, **_moves.thread_knobs(dag, t)}  # par-only SPLIT
                for r in regs:
                    rows.append({**base, **_moves.reg_knobs(dag, base, r)})  # complete SPLIT
        # Narrow to the golden-plausible band so the cold prior can't argmin onto a
        # degenerate tile; fall back to the ungated set if no in-band candidate exists.
        gated = [r for r in rows if _matmul_thread_gate(r, dag, rk)]
        return (gated or rows), THREAD_KNOBS

    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get({"fp16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(dtype, ""))
    if atom is None:
        return [], WARP_KNOBS
    rows = []
    bks = _moves.warp_bk_offers(dag, atom)
    regs = _moves.warp_reg_offers(dag, atom)
    for wm, wn in _moves.warp_offers(dag, atom):  # wm·wn ≥ 2 already (single-warp mma pruned)
        geom = _moves.warp_geom_knobs(dag, wm, wn)  # par-only SPLIT
        for fm, fn in regs:
            row = {**geom, **_moves.warp_reg_knobs(dag, geom, fm, fn)}  # complete SPLIT
            for bk in bks:
                rows.append({**row, **_moves.warp_bk_knobs(dag, atom, bk)})
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


def pick_matmul(M: int, N: int, K: int, dtype: str, ctx: Context) -> dict:
    """Best knob row for an ``(M, K) @ (K, N)`` matmul under the analytic prior —
    no learned data, no measurements. Thin wrapper over :func:`evaluate_golden`
    (no golden to match). Returns ``{}`` if nothing enumerates."""
    return evaluate_golden(M, N, K, dtype, {}, ctx)[0]
