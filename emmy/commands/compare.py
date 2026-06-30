"""``emmy compare <dumpA> <dumpB>`` — diff the bench results two compiler
dump dirs recorded.

Three sections, each present when both dumps carry the artifact:

- **Full model** (``60_bench_compare.json``): per backend, A vs B latency + ratio.
- **Per-kernel, torch-comparable** (``62_kernel_bench.json``, written by
  ``tune --bench``): each kernel's emmy latency A vs B, matched by
  provenance name — exact (hash-suffixed) name first, then base name (hash
  stripped) in order of appearance, so a re-tuned kernel whose content hash
  moved still pairs up. Kernels present on one side only are listed as
  added / removed — the kernel-set-change (structural fork / fusion) signal.
- **Per-launch emmy** (``60_benchmark.json``): the same matching over the
  raw per-launch times — the fallback when a dump has no per-kernel bench.

Ratios outside ``--tol`` are colored (green faster, red slower) and flagged, so
the before/after of a compiler change reads off one table instead of two
terminal scrollbacks. Per-kernel rows, not the full-model total, are the stable
signal across tunes (greedy picks vary run to run on scalar-tier kernels).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from pathlib import Path

from emmy.commands.table import GREEN as _GREEN
from emmy.commands.table import RED as _RED
from emmy.commands.table import Col, render_table

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"_[0-9a-f]{6}$")


def register_compare_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Diff two dump dirs' bench results (full-model backends + per-kernel latencies)",
    )
    parser.add_argument("dump_a", help="Baseline dump dir (the 'before').")
    parser.add_argument("dump_b", help="Comparison dump dir (the 'after').")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.10,
        help="Relative change beyond which a row is colored + flagged (default: 0.10).",
    )
    parser.set_defaults(func=handle_compare)


def handle_compare(args) -> None:
    a, b = Path(args.dump_a), Path(args.dump_b)
    for d in (a, b):
        if not d.is_dir():
            logger.error("not a dump dir: %s", d)
            raise SystemExit(2)
    printed = _compare_full_model(a, b, args.tol)
    printed |= _compare_kernel_bench(a, b, args.tol)
    printed |= _compare_per_launch(a, b, args.tol)
    if not printed:
        logger.error(
            "no comparable bench artifacts in both dumps — looked for 60_bench_compare.json, 62_kernel_bench.json, 60_benchmark.json"
        )
        raise SystemExit(2)


def _load(dump: Path, name: str):
    path = dump / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable %s (%s) — skipping its section", path, exc)
        return None


def _ratio_cell(a_us: float | None, b_us: float | None, tol: float) -> tuple[str, str]:
    """``B/A`` as a colored cell: green = B faster than A beyond tol, red =
    slower beyond tol, plain inside the band; ``-`` when either side is absent."""
    if not a_us or not b_us:
        return ("-", "")
    ratio = b_us / a_us
    color = _GREEN if ratio < 1 - tol else (_RED if ratio > 1 + tol else "")
    return (f"{ratio:.2f}x", color)


def _fmt_us(us: float | None) -> str:
    return f"{us:.1f}" if us is not None else "-"


def _emit(title: str, rows: list[list]) -> None:
    print()
    print(title)
    for line in render_table([Col("name"), Col("A us", "r"), Col("B us", "r"), Col("B/A", "r")], rows, rule=True):
        print(line)


def _compare_full_model(a: Path, b: Path, tol: float) -> bool:
    da, db = _load(a, "60_bench_compare.json"), _load(b, "60_bench_compare.json")
    if da is None or db is None:
        return False
    ba, bb = da.get("backends", {}), db.get("backends", {})
    rows = []
    for name in list(ba) + [n for n in bb if n not in ba]:
        a_us = (ba.get(name) or {}).get("latency_us")
        b_us = (bb.get(name) or {}).get("latency_us")
        rows.append([name, _fmt_us(a_us), _fmt_us(b_us), _ratio_cell(a_us, b_us, tol)])
    _emit("Full model (60_bench_compare.json):", rows)
    return True


def _match_kernels(rows_a: list[tuple[str, float | None]], rows_b: list[tuple[str, float | None]]):
    """Pair the two sides' ``(name, us)`` rows: exact name first, then base name
    (trailing ``_<hash6>`` stripped) in order of appearance — a re-tuned kernel
    whose content hash moved still pairs with its counterpart. Returns
    ``(matched, only_a, only_b)`` with ``matched`` in side-A order."""
    exact_b: dict[str, deque] = defaultdict(deque)
    for name, us in rows_b:
        exact_b[name].append((name, us))
    base_b: dict[str, deque] = defaultdict(deque)

    def base(name: str) -> str:
        return _HASH_RE.sub("", name)

    matched, only_a, deferred = [], [], []
    for name, us in rows_a:  # exact pass
        if exact_b[name]:
            nb, usb = exact_b[name].popleft()
            matched.append((name, us, nb, usb))
        else:
            deferred.append((name, us))
    for q in exact_b.values():  # leftovers become base-name candidates
        for nb, usb in q:
            base_b[base(nb)].append((nb, usb))
    for name, us in deferred:  # base-name pass, in order of appearance
        q = base_b[base(name)]
        if q:
            nb, usb = q.popleft()
            matched.append((name, us, nb, usb))
        else:
            only_a.append((name, us))
    only_b = [item for q in base_b.values() for item in q]
    return matched, only_a, only_b


def _emit_kernel_diff(title: str, rows_a, rows_b, tol: float) -> None:
    matched, only_a, only_b = _match_kernels(rows_a, rows_b)
    rows = []
    for name_a, us_a, name_b, us_b in sorted(matched, key=lambda m: -(m[3] or 0.0)):
        label = name_a if name_a == name_b else f"{name_a} -> {name_b}"
        rows.append([label, _fmt_us(us_a), _fmt_us(us_b), _ratio_cell(us_a, us_b, tol)])
    tot_a = sum(us for _, us, _, _ in matched if us) or None
    tot_b = sum(us for _, _, _, us in matched if us) or None
    rows.append(["TOTAL (matched)", _fmt_us(tot_a), _fmt_us(tot_b), _ratio_cell(tot_a, tot_b, tol)])
    _emit(title, rows)
    for tag, extras in (("only in A", only_a), ("only in B", only_b)):
        for name, us in extras:
            print(f"  {tag}: {name} ({_fmt_us(us)} us) — kernel-set change (structural fork / fusion difference)")


def _compare_kernel_bench(a: Path, b: Path, tol: float) -> bool:
    da, db = _load(a, "62_kernel_bench.json"), _load(b, "62_kernel_bench.json")
    if da is None or db is None:
        return False

    def rows(d) -> list[tuple[str, float | None]]:
        return [(r["kernel"], (r.get("backends") or {}).get("Emmy")) for r in d]

    _emit_kernel_diff("Per-kernel emmy -O3 (62_kernel_bench.json, tune --bench):", rows(da), rows(db), tol)
    return True


def _compare_per_launch(a: Path, b: Path, tol: float) -> bool:
    da, db = _load(a, "60_benchmark.json"), _load(b, "60_benchmark.json")
    if da is None or db is None or not (da.get("per_launch") and db.get("per_launch")):
        return False

    def rows(d) -> list[tuple[str, float | None]]:
        return [(lt["kernel_name"], lt["time_ms"] * 1000.0 if lt.get("time_ms") is not None else None) for lt in d["per_launch"]]

    _emit_kernel_diff("Per-launch emmy (60_benchmark.json):", rows(da), rows(db), tol)
    return True
