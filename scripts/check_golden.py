"""Compile each golden config with the current learned prior (greedy) and report
whether greedy reproduced the golden knobs — no benching, no tuning.

For every ``GoldenConfig`` it traces the recorded snippet, runs the deterministic
greedy compile (which loads the global prior from ``DEPLODOCK_PRIOR_FILE``), reads
the realized kernel knobs off the compiled graph, and diffs them against the
golden's recorded knobs. Use it after tuning the golden shapes to see how much of
the golden ground truth the prior has actually learned to reproduce.

Usage:
    DEPLODOCK_PRIOR_FILE=… python scripts/check_golden.py [--kernel SUBSTR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from deplodock.commands.trace import graph_from_code  # noqa: E402
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline  # noqa: E402
from deplodock.compiler.pipeline.search.golden_configs import GOLDEN_CONFIGS, MatmulGoldenConfig  # noqa: E402


def _tunable(knobs: dict) -> dict:
    return {k: v for k, v in knobs.items() if not k.startswith(("S_", "H_"))}


def picked_knobs(snippet: str) -> dict:
    """Greedy-compile the snippet (prior-driven) and union the realized op knobs."""
    graph, _, _ = graph_from_code(snippet)
    compiled = Pipeline.build(CUDA_PASSES).run(graph)
    knobs: dict = {}
    for node in compiled.nodes.values():
        k = getattr(node.op, "knobs", None)
        if k:
            knobs.update(k)
    return _tunable(knobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", help="only golden configs whose name contains this substring")
    args = ap.parse_args()

    configs = [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig)]
    if args.kernel:
        configs = [g for g in configs if args.kernel in g.name]

    n_match = 0
    print(f"{'golden':28} {'match':>6}  diff (golden→greedy on golden's keys)")
    print("-" * 90)
    for g in configs:
        gold = _tunable(g.knobs)
        try:
            got = picked_knobs(g.snippet())
        except Exception as e:  # noqa: BLE001
            print(f"{g.name:28} {'ERR':>6}  {type(e).__name__}: {e}")
            continue
        diff = {k: (gold[k], got.get(k)) for k in gold if got.get(k) != gold[k]}
        if not diff:
            n_match += 1
        shown = ", ".join(f"{k}:{a}→{b}" for k, (a, b) in sorted(diff.items())) if diff else "—"
        print(f"{g.name:28} {'✓' if not diff else '✗':>6}  {shown}")
    print("-" * 90)
    print(f"reproduced golden knobs: {n_match}/{len(configs)}")


if __name__ == "__main__":
    main()
