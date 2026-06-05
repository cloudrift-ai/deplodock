"""Shared dataset extraction for offline learned-prior training / evaluation.

The learned tuning prior (``deplodock.compiler.pipeline.search.prior``) checkpoints
its training set as ``archived_rows`` — value-of-position ``(knobs, label)`` pairs
(``label = log reward = -log median_us``) spanning every op tuned so far, with the
op's ``S_*`` structural features and ``H_*`` hardware/nvcc-regime features baked into
each knob dict. This module loads one or more such checkpoints into a grouped
dataset, used by ``train_prior.py`` (offline fit) and ``prior_bakeoff.py`` (model
comparison).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from deplodock import storage  # noqa: E402


def load_rows(paths: list[str | Path]) -> list[tuple[dict, float]]:
    """All ``(knobs, label)`` rows from the given prior checkpoint JSON files,
    concatenated. Accepts both the current ``dataset`` key (CatBoost prior) and
    the legacy ``archived_rows`` key."""
    rows: list[tuple[dict, float]] = []
    for p in paths:
        obj = storage.read_json(p)
        if obj is None:
            raise SystemExit(f"no readable prior dataset at {p}")
        rows.extend((dict(k), float(v)) for k, v in obj.get("dataset", obj.get("archived_rows", [])))
    return rows


def op_sig(knobs: dict) -> tuple:
    """An op's identity for grouping — its ``S_*`` structural feature signature
    (same op structure → same key, regardless of tunable-knob choices)."""
    return tuple(sorted((k, v) for k, v in knobs.items() if k.startswith("S_")))


def n_tunable(knobs: dict) -> int:
    """Count of real tuning knobs (excludes ``S_*`` structural / ``H_*`` regime
    features) — a leaf config has the most, a value-of-position branch fewer."""
    return sum(1 for k in knobs if not k.startswith(("S_", "H_")))


def group_by_op(rows: list[tuple[dict, float]]) -> dict[tuple, list[tuple[dict, float]]]:
    """Bucket rows by :func:`op_sig` so each group is one op-structure's configs."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[op_sig(r[0])].append(r)
    return dict(groups)
