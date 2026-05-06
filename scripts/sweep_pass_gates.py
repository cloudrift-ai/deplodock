"""Sweep the perf-suite cases under (chunk_reduce, pad_smem) gate matrix.

For each case in ``tests/perf/cases.py`` and each combo of
``DEPLODOCK_DISABLE_CHUNK_REDUCE`` / ``DEPLODOCK_DISABLE_PAD_SMEM``,
runs the tile pipeline and sums broadcast-corrected
``conflict_events`` across all (Stage, Load) bindings. Used to surface
cases where one pass dominates the conflict reduction.

Output: a CSV-ish table on stdout, plus ``--json`` to dump full
per-stage records for downstream report rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Make tests.perf importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deplodock.compiler.diagnostics.bank_conflicts import find_all_bindings, simulate  # noqa: E402
from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline  # noqa: E402
from tests.perf.cases import FUSED_CASES, PRIMITIVE_CASES, build_deplodock_graph  # noqa: E402

CONFIGS = [
    ("baseline", {}),
    ("no_chunk_reduce", {"DEPLODOCK_DISABLE_CHUNK_REDUCE": "1"}),
    ("no_pad_smem", {"DEPLODOCK_DISABLE_PAD_SMEM": "1"}),
    ("no_chunk_no_pad", {"DEPLODOCK_DISABLE_CHUNK_REDUCE": "1", "DEPLODOCK_DISABLE_PAD_SMEM": "1"}),
]

GATES = ("DEPLODOCK_DISABLE_CHUNK_REDUCE", "DEPLODOCK_DISABLE_PAD_SMEM")


def with_env(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in GATES}

    class Ctx:
        def __enter__(self_):
            for k in GATES:
                os.environ[k] = overrides.get(k, "0")
            return self_

        def __exit__(self_, *_):
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return Ctx()


def total_events_per_kernel(graph) -> dict[str, dict[str, int]]:
    """Map kernel_name → {stage_name → total_conflict_events}."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen = set()
    for b in find_all_bindings(graph):
        key = (b.tile_op_name, b.stage.name, b.load.name, tuple(b.enclosing_loop_axes))
        if key in seen:
            continue
        seen.add(key)
        r = simulate(b)
        if r is None:
            continue
        out[b.tile_op_name][r.stage_name] += r.conflict_events
    return {k: dict(v) for k, v in out.items()}


def run_case(case, configs=CONFIGS) -> dict:
    """Run ``case`` under each gate config; return per-config results."""
    record = {"case": case.name, "op": case.op, "shapes": [list(s) for s in case.shapes]}
    for cfg_name, env in configs:
        try:
            with with_env(env):
                g = build_deplodock_graph(case)
                run_pipeline(g, TILE_PASSES)
                kernels = total_events_per_kernel(g)
                total = sum(sum(s.values()) for s in kernels.values())
                record[cfg_name] = {"total": total, "per_kernel": kernels}
        except Exception as e:  # noqa: BLE001 — diagnostic helper
            record[cfg_name] = {"error": f"{type(e).__name__}: {e}"}
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include", nargs="*", default=[], help="substring filters; empty = all")
    p.add_argument("--exclude", nargs="*", default=[], help="substring filters")
    p.add_argument("--max-cases", type=int, default=0, help="cap number of cases (0 = unlimited)")
    p.add_argument("--json", default=None, help="write full records to this path")
    args = p.parse_args()

    cases = PRIMITIVE_CASES + FUSED_CASES
    if args.include:
        cases = [c for c in cases if any(t in c.name for t in args.include)]
    if args.exclude:
        cases = [c for c in cases if not any(t in c.name for t in args.exclude)]
    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"{'case':45} {'baseline':>10} {'-chunk':>10} {'-pad':>10} {'-both':>10}")
    print("-" * 90)
    records = []
    for case in cases:
        rec = run_case(case)
        records.append(rec)

        def cell(cfg, rec=rec):
            r = rec.get(cfg, {})
            return r.get("total", "ERR") if isinstance(r, dict) else "ERR"

        print(
            f"{case.name:45} {cell('baseline'):>10} {cell('no_chunk_reduce'):>10} {cell('no_pad_smem'):>10} {cell('no_chunk_no_pad'):>10}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
