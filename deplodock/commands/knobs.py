"""``deplodock knobs`` — knob-impact analysis from the tune DB.

For each kernel (grouped by the kernel C identifier extracted from
``cuda_op.pretty``), compute per-knob regret:

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
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from deplodock.commands.compile import resolve_tune_db

logger = logging.getLogger(__name__)

_KERNEL_NAME_RE = re.compile(r"void\s+(\w+)\s*\(")


def register_knobs_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "knobs",
        help="List the registered knob schema, then (if a tune DB exists) per-knob regret + interactions",
    )
    parser.add_argument(
        "--db",
        help="Path to tune DB. Default uses DEPLODOCK_TUNE_DB or ~/.cache/deplodock/autotune.db.",
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=8,
        help="Skip kernels with fewer than this many measured variants (default: 8).",
    )
    parser.add_argument(
        "--kernel",
        help="Only include kernels whose C identifier contains this substring (e.g. 'matmul').",
    )
    parser.set_defaults(func=handle_knobs)


def handle_knobs(args) -> None:
    # The canonical schema from the knob registry — always available (no DB).
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
