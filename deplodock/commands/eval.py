"""``deplodock eval <knobs|prior|heuristic>`` — evaluate the tuning machinery.

Three subcommands:

- ``eval knobs``     — print the registered knob schema, then (with a tune DB)
  per-knob **regret** + a knob-interaction matrix (the analysis below).
- ``eval heuristic`` — evaluate the hardcoded prior-free heuristic
  (``search/heuristic``) on the golden configs: the golden's **rank** in the
  heuristic's enumeration order (the position the tuner's patience must reach).
- ``eval prior``     — evaluate the learned ``CatBoostPrior`` on the golden
  configs: the greedy pipeline pick vs golden (per-knob ``found/golden``), the
  golden's rank under the prior, and (``--features``) the regressor input vector.

The ``eval knobs`` regret analysis: for each kernel (grouped by the kernel C
identifier extracted from ``cuda_op.pretty``), compute per-knob regret:

    regret[K] = max(best_us | K=v) / min(best_us | K=v)

where ``best_us | K=v`` is the minimum measured latency over variants
pinning ``K=v`` (marginalizing the other knobs by taking min). Aggregate
across kernels with median / p90 / geometric mean and print a sorted
table.

The intended use is to decide knob ordering for a hierarchical Fork tree
in the planner: high-regret knobs go at the root of the tree (commit
first), low-regret knobs go at the leaves. A second table shows
pairwise knob interaction so coupled knobs (where the optimal value of
K2 depends on K1) can be kept in the same Fork rather than split across
levels.

Grouping caveat: the analysis groups variants by kernel C identifier
only — different shapes of the same kernel collapse into one group.
Same-kernel-different-shape variants are comparable in *relative* knob
impact even when absolute latencies differ, so the rank order of knobs
is the load-bearing output here.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from deplodock.commands.compile import resolve_tune_db

logger = logging.getLogger(__name__)

_KERNEL_NAME_RE = re.compile(r"void\s+(\w+)\s*\(")


def register_eval_command(subparsers) -> None:
    """``deplodock eval <knobs|prior|heuristic>`` — evaluate the tuning knobs, the
    learned prior, or the hardcoded heuristic against the golden configs."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate tuning knobs / the learned prior / the heuristic against golden configs",
    )
    sub = parser.add_subparsers(dest="eval_target", required=True)

    pk = sub.add_parser("knobs", help="Print the registered knob schema + (with a tune DB) per-knob regret + interactions")
    pk.add_argument("--db", help="Path to tune DB for the regret analysis. Default: DEPLODOCK_TUNE_DB or ~/.cache/deplodock/autotune.db.")
    pk.add_argument("--min-variants", type=int, default=8, help="Skip kernels with fewer than this many measured variants (default: 8).")
    pk.add_argument("--kernel", help="Only include kernels whose C identifier contains this substring (e.g. 'matmul').")
    pk.set_defaults(func=handle_eval_knobs)

    ph = sub.add_parser(
        "heuristic",
        help="Evaluate the hardcoded prior-free heuristic on the golden configs (golden's rank in its enumeration order)",
    )
    ph.add_argument("--kernel", help="Filter golden configs by name substring (e.g. 'square', 'q_proj').")
    ph.set_defaults(func=handle_eval_heuristic)

    pp = sub.add_parser(
        "prior",
        help="Evaluate the learned prior on the golden configs (greedy pick vs golden + golden's rank under the prior)",
    )
    pp.add_argument(
        "--prior",
        help="Path to the learned-prior JSON to load. Default: DEPLODOCK_PRIOR_FILE or ~/.cache/deplodock/prior.json. "
        "(`deplodock tune` writes this file; it is NOT the tune DB.)",
    )
    pp.add_argument("--kernel", help="Filter golden configs by name substring.")
    pp.add_argument(
        "--features",
        action="store_true",
        help="Also print the exact feature vector the prior (CatBoost) regresses on per golden config (knob.knob_features).",
    )
    pp.set_defaults(func=handle_eval_prior)

    pg = sub.add_parser(
        "golden",
        help="Greedy pipeline pick vs recorded golden, per golden config (the reproduction check; no heuristic/rank diagnostics)",
    )
    pg.add_argument("--prior", help="Learned-prior JSON to load (default: DEPLODOCK_PRIOR_FILE or ~/.cache/deplodock/prior.json).")
    pg.add_argument("--kernel", help="Filter golden configs by name substring.")
    pg.add_argument("--features", action="store_true", help="Also print the prior's regressor feature vector per golden config.")
    pg.set_defaults(func=handle_eval_golden)


def handle_eval_knobs(args) -> None:
    """``eval knobs`` — the registered knob schema, then (with a tune DB) per-knob
    regret + the knob-interaction matrix."""
    _emit_registry()

    db_path = Path(args.db) if args.db else resolve_tune_db()
    if not db_path.exists():
        logger.info("")
        logger.info("No tune DB at %s — skipping the measured per-knob regret analysis.", db_path)
        return
    logger.info("")
    logger.info("Reading: %s", db_path)

    variants_by_kernel = _load_variants(db_path, kernel_filter=args.kernel)
    kernels = {k: v for k, v in variants_by_kernel.items() if len(v) >= args.min_variants}
    logger.info(
        "Kernels with ≥%d measured variants: %d (of %d total)",
        args.min_variants,
        len(kernels),
        len(variants_by_kernel),
    )
    if not kernels:
        return

    rows = _compute_knob_regret(kernels)
    if not rows:
        logger.info("No knob varied across ≥2 values in any kernel — nothing to rank.")
        return
    _emit_regret_table(rows)

    interactions = _compute_interactions(kernels, [r.knob for r in rows])
    _emit_interaction_matrix([r.knob for r in rows], interactions)


def handle_eval_heuristic(args) -> None:
    """``eval heuristic`` — the hardcoded prior-free heuristic's rank of each golden."""
    _emit_heuristic_eval(args.kernel)


def handle_eval_prior(args) -> None:
    """``eval prior`` — the learned prior on the golden configs: the greedy pick vs
    golden, the golden's rank under the prior, and (with ``--features``) the
    regressor input vector."""
    if args.prior:
        from deplodock import config  # noqa: PLC0415

        os.environ[config.PRIOR_FILE] = str(Path(args.prior).expanduser())
    if args.features:
        _emit_golden_features(args.kernel)
    _emit_prior_eval(args.kernel)


def handle_eval_golden(args) -> None:
    """``eval golden`` — the greedy pipeline pick vs recorded golden per config (the
    actionable "did the pipeline reproduce the golden knobs?" view). Watch it while
    iteratively tuning golden shapes one at a time (``deplodock tune --golden
    <name>``). Use ``eval heuristic`` / ``eval prior`` for the heuristic rank and the
    rank-under-prior diagnostics."""
    if args.prior:
        from deplodock import config  # noqa: PLC0415

        os.environ[config.PRIOR_FILE] = str(Path(args.prior).expanduser())
    if args.features:
        _emit_golden_features(args.kernel)
    configs, nw = _golden_configs(args.kernel)
    _emit_prior_golden_check(configs, nw, title=False)


def _emit_registry() -> None:
    """List every registered :class:`~deplodock.compiler.pipeline.knob.Knob` — the
    canonical tuning schema (name, type, candidate hints, aliases, help) collected
    by ``knob.registry`` from all loaded passes, regardless of any DB."""
    from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline, knob  # noqa: PLC0415

    # ``registry`` only sees passes already imported into ``sys.modules``; build
    # the full pipeline once so every Knob-bearing rule module is loaded first.
    Pipeline.build(CUDA_PASSES)
    import textwrap  # noqa: PLC0415

    reg = knob.registry()
    names = sorted(reg)
    kw = max((len(n) for n in names), default=4)  # knob-name column width
    tw = max((len(reg[n].type.value) for n in names), default=4)  # type column width
    hw = 33  # hints column width (truncated past this)
    help_w = 64  # help wraps to this width; continuation lines indent under it
    indent = " " * (kw + 2 + tw + 2 + hw + 2)

    logger.info("Registered tuning knobs (%d) — the canonical schema:", len(reg))
    logger.info(f"{'knob':<{kw}}  {'type':<{tw}}  {'candidates':<{hw}}  help")
    logger.info("-" * (kw + 2 + tw + 2 + hw + 2 + help_w))
    for name in names:
        k = reg[name]
        hints = ", ".join(str(h) for h in k.hints) if k.hints else "-"
        if len(hints) > hw:
            hints = hints[: hw - 1] + "…"
        help_txt = " ".join((k.help or "").split())  # collapse whitespace/newlines
        if k.aliases:
            help_txt += f" [aliases: {', '.join(k.aliases)}]"
        lines = textwrap.wrap(help_txt, width=help_w) or [""]
        logger.info(f"{name:<{kw}}  {k.type.value:<{tw}}  {hints:<{hw}}  {lines[0]}")
        for cont in lines[1:]:
            logger.info(indent + cont)


_WARP_KNOBS = ("WN", "WM", "FM", "FN", "BK", "SPLITK", "MMA")

# ANSI colours, only when stdout is a tty (piped / logged output stays plain).
_TTY = sys.stdout.isatty()
_GREEN, _YELLOW, _RED, _RESET = ("\033[32m", "\033[33m", "\033[31m", "\033[0m") if _TTY else ("", "", "", "")


def _ratio_color(matched: int, total: int) -> str:
    """Green (all match) / yellow (>80%) / red (otherwise)."""
    frac = matched / total if total else 1.0
    return _GREEN if matched == total else (_YELLOW if frac > 0.8 else _RED)


def _cell(visible: str, width: int, color: str = "") -> str:
    """Left-justify ``visible`` to ``width`` columns, padding by its *visible*
    length so embedded ANSI colour codes don't throw off the alignment."""
    body = f"{color}{visible}{_RESET}" if color else visible
    return body + " " * max(1, width - len(visible))


# Canonical knob display order — the tile geometry knobs first (block / register
# tile, split, pipeline), then everything else alphabetically. Keeps the
# ``found/golden`` columns in a stable, readable order across kernels.
_KNOB_ORDER = ("BM", "BN", "BK", "BR", "FM", "FN", "FK", "WM", "WN", "SPLITK", "BUFFER_COUNT", "STAGE", "MMA")
_KNOB_RANK = {k: i for i, k in enumerate(_KNOB_ORDER)}


def _knob_sort_key(k: str) -> tuple[int, str]:
    """Sort knobs by :data:`_KNOB_ORDER`, unknown knobs last (alphabetically)."""
    return (_KNOB_RANK.get(k, len(_KNOB_ORDER)), k)


def _aligned_knob_cells(rows: list[tuple[dict, dict]]) -> list[str]:
    """Render the ``K=found/golden`` knob columns for a set of ``(gold, got)`` rows,
    aligned: the union of knobs across all rows in canonical order, each knob padded
    to its widest cell so the columns line up vertically; mismatches red, blanks
    where a row lacks the knob. Returns one string per input row, in order."""
    keys = sorted({k for gold, _ in rows for k in gold}, key=_knob_sort_key)
    width = {k: max(len(f"{k}={got.get(k, '-')}/{gold[k]}") for gold, got in rows if k in gold) for k in keys}
    lines = []
    for gold, got in rows:
        cells = []
        for k in keys:
            if k not in gold:
                cells.append(" " * width[k])
                continue
            gv, fv = gold[k], got.get(k, "-")
            vis = f"{k}={fv}/{gv}"
            body = f"{_RED}{vis}{_RESET}" if fv != gv else vis
            cells.append(body + " " * (width[k] - len(vis)))
        lines.append("  ".join(cells).rstrip())
    return lines


def _emit_golden_features(kernel_filter: str | None) -> None:
    """Print, per golden config, the exact feature vector the learned
    :class:`CatBoostPrior` regresses on — ``knob.knob_features(merged)`` where
    ``merged`` is the ``H_*`` host/regime features + the ``S_*`` structural/shape
    features (obtained by compiling the shape to the loop dialect, where
    ``992_stamp_structural_features`` runs) + the golden tuning knobs. This is
    the model's *input* for that shape+config — note the shape enters only as the
    coarse ``S_ext_*`` extent products/maxes; the occupancy / CTA-count / reuse
    terms that drive matmul perf (and that the heuristic computes) are NOT here."""
    import logging as _logging  # noqa: PLC0415

    from deplodock.commands.trace import graph_from_code  # noqa: PLC0415
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline, knob  # noqa: PLC0415
    from deplodock.compiler.pipeline.knob import CTX_PREFIX, STRUCT_PREFIX  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: PLC0415

    configs = [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)]
    if kernel_filter:
        configs = [g for g in configs if kernel_filter in g.name]

    logger.info("")
    logger.info("Learned-prior feature vector (knob.knob_features) — the CatBoost regressor's input per golden config:")
    quiet = [_logging.getLogger(n) for n in ("deplodock.compiler", "deplodock.commands.trace")]
    prev = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(_logging.WARNING)
    try:
        for g in configs:
            try:
                graph, _, _ = graph_from_code(g.snippet())
                compiled = Pipeline.build(LOOP_PASSES).run(graph)  # loop dialect — S_* stamped, no codegen
                s_feats: dict = {}
                for n in compiled.nodes.values():
                    s_feats.update({k: v for k, v in (getattr(n.op, "knobs", {}) or {}).items() if k.startswith(STRUCT_PREFIX)})
                merged = {**Context.from_target(g.compute_cap).features(), **s_feats, **g.knobs}
                feats = knob.knob_features(merged)
            except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
                logger.info("  %-26s  ERR  %s", g.name, " ".join(f"{type(e).__name__}: {e}".split())[:100])
                continue
            logger.info("  %s  (%d features):", g.name, len(feats))
            tuning = {k: v for k, v in feats.items() if not k.startswith((STRUCT_PREFIX, CTX_PREFIX))}
            for label, sel in (
                ("S_", {k: v for k, v in feats.items() if k.startswith(STRUCT_PREFIX)}),
                ("H_", {k: v for k, v in feats.items() if k.startswith(CTX_PREFIX)}),
                ("knob", tuning),
            ):
                if sel:
                    logger.info("    %-5s %s", label, " ".join(f"{k}={v:g}" for k, v in sorted(sel.items())))
    finally:
        for lg, lv in zip(quiet, prev, strict=True):
            lg.setLevel(lv)


def _golden_configs(kernel_filter: str | None):
    """The matmul golden configs, optionally filtered by name substring, plus the
    kernel-name column width for aligned tables."""
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: PLC0415

    configs = [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)]
    if kernel_filter:
        configs = [g for g in configs if kernel_filter in g.name]
    return configs, max((len(g.name) for g in configs), default=10) + 2


def _emit_heuristic_eval(kernel_filter: str | None) -> None:
    """``eval heuristic`` body: the hardcoded prior-free heuristic
    (``search/heuristic``) — no prior, no GPU, no measurements. One streamed line
    per golden config with the golden's **rank** in the heuristic's enumeration
    order (the position the tuner's patience must reach) + per-knob ``found/golden``
    (mismatches red), summarized as median + top-k coverage."""
    from statistics import median  # noqa: PLC0415

    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.heuristic import THREAD_KNOBS, evaluate_golden  # noqa: PLC0415

    configs, nw = _golden_configs(kernel_filter)
    logger.info("  " + _cell("kernel", nw) + _cell("m/t", 6) + _cell("rank", 8) + _cell("pool", 9) + "knobs (found/golden; red = mismatch)")
    ranks: list[int] = []
    entries: list[tuple] = []  # ("row", prefix, gold, got) | ("err", text)
    for g in configs:
        gold = {k: v for k, v in g.knobs.items() if k in (THREAD_KNOBS if g.dtype == "fp32" else _WARP_KNOBS)}
        try:
            got, rank, pool = evaluate_golden(g.M, g.N, g.K, g.dtype, gold, Context.from_target(g.compute_cap))
        except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
            entries.append(("err", _cell(g.name, nw) + "ERR  " + " ".join(f"{type(e).__name__}: {e}".split())[:100]))
            continue
        matched = sum(1 for k in gold if got.get(k) == gold[k])
        prefix = (
            _cell(g.name, nw)
            + _cell(f"{matched}/{len(gold)}", 6, _ratio_color(matched, len(gold)))
            + _cell(str(rank) if rank is not None else "?", 8)
            + _cell(str(pool), 9)
        )
        entries.append(("row", prefix, gold, got))
        if rank is not None:
            ranks.append(rank)
    knob_lines = iter(_aligned_knob_cells([(p[2], p[3]) for p in entries if p[0] == "row"]))
    for p in entries:
        logger.info("  " + (p[1] if p[0] == "err" else p[1] + next(knob_lines)))
    if ranks:
        n = len(ranks)
        cov = "  ".join(f"top{k}={sum(r < k for r in ranks)}/{n}" for k in (1, 10, 25, 50, 100))
        logger.info("")
        logger.info("  heuristic golden rank — median=%d  %s", int(median(ranks)), cov)


def _emit_prior_eval(kernel_filter: str | None) -> None:
    """``eval prior`` body: the learned ``CatBoostPrior`` on the golden configs —
    the golden's rank under the prior over the full enumeration (offline) followed
    by the greedy tile-pipeline pick vs golden (the real selection). Reads the
    prior JSON (``DEPLODOCK_PRIOR_FILE`` / ``--prior``; option-0 when none loaded)."""
    from deplodock import config  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.prior import CatBoostPrior, diagnostics  # noqa: PLC0415

    prior = CatBoostPrior.load()
    logger.info("")
    if prior.fitted:
        logger.info(diagnostics.golden_prior_eval(prior, kernel_filter))
    else:
        logger.info("No fitted prior at %s — greedy falls to option-0 (run `deplodock tune`).", config.prior_path())

    configs, nw = _golden_configs(kernel_filter)
    _emit_prior_golden_check(configs, nw)


def _emit_prior_golden_check(configs: list, nw: int, *, title: bool = True) -> None:
    """Greedy fork pick through the tile pipeline vs recorded golden. The pick reads
    the learned-prior JSON (``config.prior_path()``: ``DEPLODOCK_PRIOR_FILE`` /
    ``--prior``); option-0 with no fitted prior. Stops at the tile dialect (every
    knob fork resolves there: no codegen / nvcc). Rows are collected, then printed
    with column-aligned ``found/golden`` knobs (canonical order). ``title`` prints
    the ``Golden reproduction — … prior: <path>`` banner (``eval prior``);
    ``eval golden`` passes ``title=False`` for just the table."""
    import logging as _logging  # noqa: PLC0415

    from deplodock import config  # noqa: PLC0415
    from deplodock.commands.trace import graph_from_code  # noqa: PLC0415
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline  # noqa: PLC0415

    def tunable(knobs: dict) -> dict:
        return {k: v for k, v in knobs.items() if not k.startswith(("S_", "H_"))}

    def picked(snippet: str) -> dict:
        graph, _, _ = graph_from_code(snippet)
        compiled = Pipeline.build(TILE_PASSES).run(graph)  # tile dialect only — no codegen/nvcc
        knobs: dict = {}
        for node in compiled.nodes.values():
            k = getattr(node.op, "knobs", None)
            if k:
                knobs.update(k)
        return tunable(knobs)

    if title:
        prior_path = config.prior_path()
        logger.info("")
        logger.info(
            "Golden reproduction — greedy pipeline pick vs recorded golden; prior: %s (%s):",
            prior_path,
            "loaded" if prior_path.exists() else "MISSING → option-0",
        )
    logger.info("  " + _cell("kernel", nw) + _cell("m/t", 6) + "knobs (found/golden)")
    # Silence the trace/compile chatter (different logger subtrees) so this
    # function's own ``logger`` can stream one clean result line per config.
    quiet = [_logging.getLogger(n) for n in ("deplodock.compiler", "deplodock.commands.trace")]
    prev = [lg.level for lg in quiet]
    for lg in quiet:
        lg.setLevel(_logging.WARNING)
    n_match = 0
    entries: list[tuple] = []  # ("row", prefix, gold, got) | ("err", text)
    try:
        for g in configs:
            gold = tunable(g.knobs)
            try:
                got = picked(g.snippet())
            except Exception as e:  # noqa: BLE001 — one shape's error shouldn't abort the report
                entries.append(("err", _cell(g.name, nw) + "ERR  " + " ".join(f"{type(e).__name__}: {e}".split())[:100]))
                continue
            matched = sum(1 for k in gold if got.get(k) == gold[k])
            n_match += matched == len(gold)
            prefix = _cell(g.name, nw) + _cell(f"{matched}/{len(gold)}", 6, _ratio_color(matched, len(gold)))
            entries.append(("row", prefix, gold, got))
    finally:
        for lg, lv in zip(quiet, prev, strict=True):
            lg.setLevel(lv)
    knob_lines = iter(_aligned_knob_cells([(p[2], p[3]) for p in entries if p[0] == "row"]))
    for p in entries:
        logger.info("  " + (p[1] if p[0] == "err" else p[1] + next(knob_lines)))
    logger.info("")
    logger.info("  pipeline reproduced golden knobs exactly: %d/%d", n_match, len(configs))


@dataclass(frozen=True)
class KnobRow:
    knob: str
    n_kernels: int
    median_values: int
    median_regret: float
    p90_regret: float
    geomean_regret: float


def _load_variants(db_path: Path, *, kernel_filter: str | None) -> dict[str, list[tuple[dict, float]]]:
    """Return ``{kernel_name: [(knobs_dict, latency_us), ...]}``.

    Joins ``perf`` (measured latencies) with ``cuda_op`` (for the kernel
    source we parse the C identifier from). Opens the DB read-only so a
    concurrent ``deplodock tune`` doesn't block on the schema check.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT cuda_op.pretty, perf.knobs, perf.latency_us_median "
            "FROM perf JOIN cuda_op ON perf.op_key = cuda_op.key "
            "WHERE perf.status='ok' AND perf.latency_us_median > 0"
        )
        rows = cur.fetchall()
    finally:
        con.close()

    out: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for pretty, knobs_json, us in rows:
        m = _KERNEL_NAME_RE.search(pretty)
        if not m:
            continue
        name = m.group(1)
        if kernel_filter and kernel_filter not in name:
            continue
        try:
            knobs = json.loads(knobs_json)
        except (TypeError, json.JSONDecodeError):
            continue
        out[name].append((knobs, us))
    return out


def _compute_knob_regret(kernels: dict[str, list[tuple[dict, float]]]) -> list[KnobRow]:
    per_knob_regret: dict[str, list[float]] = defaultdict(list)
    per_knob_n_values: dict[str, list[int]] = defaultdict(list)
    for variants in kernels.values():
        all_knobs: set[str] = set()
        for knobs, _ in variants:
            all_knobs.update(knobs.keys())
        for K in all_knobs:
            best_by_value: dict = {}
            for knobs, us in variants:
                v = knobs.get(K)
                if v is None:
                    continue
                if v not in best_by_value or us < best_by_value[v]:
                    best_by_value[v] = us
            if len(best_by_value) < 2:
                # Knob took only one distinct value across this kernel's
                # variants — no choice to evaluate.
                continue
            latencies = list(best_by_value.values())
            per_knob_regret[K].append(max(latencies) / min(latencies))
            per_knob_n_values[K].append(len(best_by_value))

    rows = [
        KnobRow(
            knob=K,
            n_kernels=len(per_knob_regret[K]),
            median_values=int(median(per_knob_n_values[K])),
            median_regret=median(per_knob_regret[K]),
            p90_regret=_percentile(per_knob_regret[K], 0.90),
            geomean_regret=_geomean(per_knob_regret[K]),
        )
        for K in per_knob_regret
    ]
    rows.sort(key=lambda r: -r.geomean_regret)
    return rows


def _compute_interactions(
    kernels: dict[str, list[tuple[dict, float]]],
    knobs: list[str],
) -> dict[tuple[str, str], float | None]:
    """For each ordered pair (K1, K2): fraction of kernels where the
    argmin K2 value changes across different K1 values."""
    out: dict[tuple[str, str], float | None] = {}
    for K1 in knobs:
        for K2 in knobs:
            if K1 == K2:
                continue
            n_changes = 0
            n_total = 0
            for variants in kernels.values():
                argmin_by_v1: dict = {}
                for knobs_dict, us in variants:
                    v1 = knobs_dict.get(K1)
                    v2 = knobs_dict.get(K2)
                    if v1 is None or v2 is None:
                        continue
                    prev = argmin_by_v1.get(v1)
                    if prev is None or us < prev[1]:
                        argmin_by_v1[v1] = (v2, us)
                if len(argmin_by_v1) < 2:
                    continue
                n_total += 1
                if len({entry[0] for entry in argmin_by_v1.values()}) > 1:
                    n_changes += 1
            out[(K1, K2)] = (n_changes / n_total) if n_total else None
    return out


def _emit_regret_table(rows: list[KnobRow]) -> None:
    header = f"{'knob':<10} {'n_kernels':>10} {'median_n_vals':>14} {'median_regret':>14} {'p90_regret':>11} {'geomean_regret':>15}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows:
        logger.info(
            "%-10s %10d %14d %13.2fx %10.2fx %14.2fx",
            r.knob,
            r.n_kernels,
            r.median_values,
            r.median_regret,
            r.p90_regret,
            r.geomean_regret,
        )


def _emit_interaction_matrix(knobs: list[str], interactions: dict[tuple[str, str], float | None]) -> None:
    logger.info("")
    logger.info("knob interaction — frac of kernels where argmin(K2) changes across K1 values")
    logger.info("(high value = knobs are coupled; can't commit to K1 then search K2 independently)")
    logger.info("K1\\K2".ljust(10) + "".join(f"{k:>10}" for k in knobs))
    for K1 in knobs:
        cells = []
        for K2 in knobs:
            if K1 == K2:
                cells.append(f"{'-':>10}")
                continue
            v = interactions.get((K1, K2))
            cells.append(f"{v:>10.2f}" if v is not None else f"{'-':>10}")
        logger.info(f"{K1:<10}" + "".join(cells))


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[int(round((len(s) - 1) * p))]


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
