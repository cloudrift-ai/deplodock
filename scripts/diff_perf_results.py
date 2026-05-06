"""Diff two ``tests/perf/.results/*.json`` files and surface kernels that
got slower (or faster) under different pass-gate configs.

Usage::

    python scripts/diff_perf_results.py BASELINE.json DEGRADED.json [--top N]

Prints a table sorted by relative slowdown (degraded_us / baseline_us)
descending — biggest losers first. Cases that exist in only one run
appear with an annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _by_name(d: dict) -> dict[str, dict]:
    return {row["name"]: row for row in d.get("rows", [])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("degraded")
    p.add_argument("--top", type=int, default=20, help="rows to print on each side (default 20)")
    p.add_argument("--threshold", type=float, default=1.05, help="relative-slowdown threshold to flag (default 1.05x)")
    args = p.parse_args()

    base = _by_name(_load(args.baseline))
    deg = _by_name(_load(args.degraded))
    common = sorted(set(base) & set(deg))

    rows = []
    for name in common:
        b_us = base[name].get("deplodock_us")
        d_us = deg[name].get("deplodock_us")
        if not b_us or not d_us:
            continue
        slowdown = d_us / b_us
        rows.append((slowdown, name, b_us, d_us, base[name].get("op", "")))

    rows.sort(reverse=True)

    print(f"{'slowdown':>9}  {'op':10}  {'case':45}  {'base_us':>10}  {'deg_us':>10}")
    print("-" * 95)
    flagged = [r for r in rows if r[0] >= args.threshold]
    print(f"# {len(flagged)} cases with slowdown ≥ {args.threshold}x  (showing top {args.top})")
    for slowdown, name, b_us, d_us, op in rows[: args.top]:
        marker = "  ←" if slowdown >= args.threshold else ""
        print(f"{slowdown:>8.2f}x  {op:10}  {name:45}  {b_us:>10.1f}  {d_us:>10.1f}{marker}")
    if len(rows) > args.top:
        print(f"... ({len(rows) - args.top} more, smallest slowdown {rows[-1][0]:.2f}x)")

    only_base = sorted(set(base) - set(deg))
    only_deg = sorted(set(deg) - set(base))
    if only_base:
        print(f"\nin baseline only: {only_base[:5]}{' …' if len(only_base) > 5 else ''}")
    if only_deg:
        print(f"in degraded only: {only_deg[:5]}{' …' if len(only_deg) > 5 else ''}")

    sys.exit(0)


if __name__ == "__main__":
    main()
