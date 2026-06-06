#!/usr/bin/env python
"""Tune every shape in the golden set, then ask: does greedy-with-prior find the
golden (or better) config?

Pipeline, all on an isolated prior + tune DB so the user's caches are untouched
(``DEPLODOCK_PRIOR_FILE`` / ``DEPLODOCK_TUNE_DB`` default to /tmp paths here):

  1. TUNE — for each ``GOLDEN_CONFIGS`` matmul, ``deplodock tune --code <snippet>``
     (idempotent: ops already tuned to >= patience are skipped, and per-op results
     transfer across shapes that share kernel structure). This trains the one
     global learned prior.
  2. BENCH — for each shape, ``deplodock run --code <snippet> --bench`` does the
     greedy single-shot pick *from the trained prior* and benches it at -O3.
     Parse the Deplodock latency and compare to the recorded golden ``deplodock_us``
     (same kernel path, same device → apples-to-apples). "golden or better" ==
     greedy_us <= golden_us.
  3. KNOBS — a final ``deplodock knobs --golden`` shows the greedy-with-prior knob
     pick vs the recorded golden, per knob.

Results stream to stdout and a JSON sidecar so a kill leaves partial data.

Run:  ./venv/bin/python scripts/tune_golden_set.py            # tune + bench + report
      EXP_PATIENCE=50 ./venv/bin/python scripts/tune_golden_set.py
      ./venv/bin/python scripts/tune_golden_set.py --skip-tune # re-report only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

os.environ.setdefault("DEPLODOCK_PRIOR_FILE", "/tmp/golden_exp_prior.json")
os.environ.setdefault("DEPLODOCK_TUNE_DB", "/tmp/golden_exp_autotune.db")

from deplodock.compiler.pipeline.search.golden_configs import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: E402

_DD = "./venv/bin/deplodock"
_DEPLODOCK_US = re.compile(r"Deplodock\s+([\d.]+)")
_EAGER_US = re.compile(r"Eager PyTorch\s+([\d.]+)")
_RESULTS = Path("/tmp/tune_golden_results.json")


def _run(args: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(args, env=os.environ, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "") + (e.stderr or "") + "\n<timeout>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patience", type=int, default=int(os.environ.get("EXP_PATIENCE", "50")))
    ap.add_argument("--skip-tune", action="store_true", help="reuse the existing isolated prior; bench + report only")
    ap.add_argument("--tune-timeout", type=int, default=3600)
    ap.add_argument("--bench-timeout", type=int, default=900)
    args = ap.parse_args()

    configs = [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)]
    print(f"Prior:  {os.environ['DEPLODOCK_PRIOR_FILE']}")
    print(f"DB:     {os.environ['DEPLODOCK_TUNE_DB']}")
    print(f"{len(configs)} golden shapes | patience={args.patience}\n", flush=True)

    # ---- Phase 1: tune ---- #
    if not args.skip_tune:
        for i, g in enumerate(configs, 1):
            t = time.time()
            rc, _ = _run([_DD, "tune", "--code", g.snippet(), "--patience", str(args.patience)], args.tune_timeout)
            tag = "ok" if rc == 0 else f"rc={rc}"
            print(f"[tune {i:2d}/{len(configs)}] {g.name:24s} {time.time() - t:6.0f}s  {tag}", flush=True)

    # ---- Phase 2: greedy-with-prior bench ---- #
    print("\n== greedy-with-prior bench vs recorded golden ==", flush=True)
    print(f"  {'shape':24s} {'dtype':5s} {'golden_us':>9s} {'greedy_us':>9s} {'ratio':>6s}  verdict", flush=True)
    rows = []
    for g in configs:
        rc, out = _run([_DD, "run", "--code", g.snippet(), "--bench"], args.bench_timeout)
        m = _DEPLODOCK_US.search(out)
        greedy = float(m.group(1)) if m else None
        eager = float(me.group(1)) if (me := _EAGER_US.search(out)) else None
        ratio = greedy / g.deplodock_us if greedy else None
        # "golden or better": greedy within 5% of the recorded golden deplodock latency.
        verdict = "—" if greedy is None else ("GOLDEN+" if ratio <= 1.05 else f"{ratio:.2f}x slower")
        rows.append(
            {
                "name": g.name,
                "dtype": g.dtype,
                "golden_us": g.deplodock_us,
                "golden_cublas_us": g.cublas_us,
                "greedy_us": greedy,
                "eager_us": eager,
                "ratio": ratio,
                "verdict": verdict,
            }
        )
        _RESULTS.write_text(json.dumps(rows, indent=2))
        rs = f"{ratio:.2f}" if ratio else "  —"
        gs = f"{greedy:9.1f}" if greedy else "      err"
        print(f"  {g.name:24s} {g.dtype:5s} {g.deplodock_us:9.1f} {gs} {rs:>6s}  {verdict}", flush=True)

    hits = sum(1 for r in rows if r["ratio"] is not None and r["ratio"] <= 1.05)
    print(f"\n  greedy found golden-or-better (<=1.05x): {hits}/{len(rows)}", flush=True)

    # ---- Phase 3: greedy knob reproduction ---- #
    print("\n== knobs --golden (greedy-with-prior knob pick vs golden) ==", flush=True)
    _, out = _run([_DD, "knobs", "--golden"], args.bench_timeout)
    # Forward just the two golden-reproduction blocks.
    keep = False
    for line in out.splitlines():
        if "Golden reproduction" in line:
            keep = True
        if keep:
            print(line, flush=True)


if __name__ == "__main__":
    main()
