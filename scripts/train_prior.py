"""Offline-fit the global learned prior from accumulated tune datasets.

Refits a fresh :class:`CatBoostPrior` on the union of the value-of-position rows
from one or more prior checkpoints and writes a deployable prior JSON (the same
format ``compile`` / ``run`` load via :meth:`CatBoostPrior.load`). Decouples
fitting from a live ``tune`` run — consolidate several machines' / sessions' tune
data into one prior, or re-fit after changing the featurizer, without re-benching.
Rows beyond the reservoir cap (``max_rows``) are uniformly subsampled.

Usage:
    python scripts/train_prior.py PRIOR_JSON [PRIOR_JSON ...] --out OUT.json
"""

from __future__ import annotations

import argparse

import prior_data

from deplodock.compiler.pipeline.search.prior import CatBoostPrior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="prior checkpoint JSON file(s) to read rows from")
    ap.add_argument("--out", required=True, help="output prior JSON (deployable; point DEPLODOCK_PRIOR_FILE here)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = prior_data.load_rows(args.inputs)
    groups = prior_data.group_by_op(rows)
    p = CatBoostPrior(seed=args.seed)
    p.add_rows(rows)  # reservoir-bounded to max_rows
    p.fit()
    if not p.fitted:
        raise SystemExit("fit produced no model (empty / featureless dataset?)")
    p._path = args.out
    p.checkpoint()
    print(f"fit {len(p._dataset)}/{len(rows)} rows over {len(groups)} op-structures -> {args.out} ({len(p._cols)} feature cols)")


if __name__ == "__main__":
    main()
