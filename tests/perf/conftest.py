"""Conftest for ``tests/perf/``: marker gating, ``bench_pair`` fixture,
session-end summary table, and JSON dump.

Tests in this directory carry ``pytestmark = [pytest.mark.perf,
requires_cuda]``. The ``perf`` marker is **deselected by default** —
plain ``pytest tests/`` skips them so ``make test`` stays fast. Run
explicitly with ``pytest -m perf`` (or ``make bench-kernels``).

The ``bench_pair`` fixture builds a torch reference and a Deplodock
graph from a ``Case``, times each with the same iteration count, and
records a ``PerfRow`` into a session-scoped collector. After the
session, ``pytest_terminal_summary`` prints one table sorted by ratio
(worst losses first) and writes the same data to
``tests/perf/.results/<utc-timestamp>.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.compiler.conftest import requires_cuda  # noqa: F401  (re-exported)
from tests.perf.cases import Case, build_deplodock_graph, build_torch_ref

_RESULTS_DIR = Path(__file__).resolve().parent / ".results"


# ---------------------------------------------------------------------------
# PerfRow + session collector
# ---------------------------------------------------------------------------


@dataclass
class PerfRow:
    name: str
    op: str
    shape: str
    dtype: str
    torch_us: float
    deplodock_us: float
    ratio: float  # torch_us / deplodock_us — >1 means deplodock wins
    launches: int
    tags: tuple[str, ...]


def _collector(config) -> list[PerfRow]:
    if not hasattr(config, "_perf_rows"):
        config._perf_rows = []
    return config._perf_rows


# ---------------------------------------------------------------------------
# Marker gating: deselect `perf` unless explicitly requested
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("-m") or ""
    if "perf" in selected:
        return
    skip_perf = pytest.mark.skip(reason="perf marker not selected; run with `pytest -m perf`")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)


# ---------------------------------------------------------------------------
# bench_pair fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_pair(request):
    """Return a callable ``run(case, *, warmup, iters) -> PerfRow``.

    Times the PyTorch eager reference and the Deplodock-compiled graph
    on the same shape, records a row, returns it. Does not assert on
    the ratio — the suite tracks performance, it doesn't gate on it.
    """

    def _run(case: Case, *, warmup: int = 5, iters: int = 50) -> PerfRow:
        import torch

        from deplodock.compiler.backend.cuda.backend import CudaBackend

        # --- torch side ---
        torch_fn = build_torch_ref(case)
        for _ in range(warmup):
            torch_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch_fn()
        end.record()
        torch.cuda.synchronize()
        torch_us = (start.elapsed_time(end) / iters) * 1000.0

        # --- deplodock side ---
        graph = build_deplodock_graph(case)
        backend = CudaBackend()
        compiled = backend.compile(graph)
        bench = backend.benchmark(compiled, warmup=warmup, num_iters=iters)
        deplodock_us = bench.time_ms * 1000.0
        launches = bench.num_launches

        ratio = (torch_us / deplodock_us) if deplodock_us > 0 else 0.0
        row = PerfRow(
            name=case.name,
            op=case.op,
            shape=case.shape_str,
            dtype=case.dtype,
            torch_us=torch_us,
            deplodock_us=deplodock_us,
            ratio=ratio,
            launches=launches,
            tags=case.tags,
        )
        _collector(request.config).append(row)
        return row

    return _run


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def _format_table(rows: list[PerfRow]) -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r.ratio)  # losses first
    headers = ("op", "case", "shape", "torch_us", "depl_us", "ratio", "launches")
    widths = [
        max(len(headers[0]), max(len(r.op) for r in rows)),
        max(len(headers[1]), max(len(r.name) for r in rows)),
        max(len(headers[2]), max(len(r.shape) for r in rows)),
        len("torch_us"),
        len("depl_us"),
        len("ratio"),
        len("launches"),
    ]

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    lines = [_fmt(headers), _fmt(tuple("-" * w for w in widths))]
    for r in rows:
        lines.append(
            _fmt(
                (
                    r.op,
                    r.name,
                    r.shape,
                    f"{r.torch_us:>8.1f}",
                    f"{r.deplodock_us:>7.1f}",
                    f"{r.ratio:>5.2f}x",
                    f"{r.launches:>8d}",
                )
            )
        )
    return "\n".join(lines)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rows: list[PerfRow] = getattr(config, "_perf_rows", [])
    if not rows:
        return
    tw = terminalreporter
    tw.write_sep("=", "perf summary (sorted by ratio; >1.00x means deplodock wins)")
    tw.write_line(_format_table(rows))

    # Persist JSON for cross-run diffing.
    _RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    out = _RESULTS_DIR / f"{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "git_rev": os.environ.get("DEPLODOCK_GIT_REV", ""),
        "rows": [{**asdict(r), "tags": list(r.tags)} for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2))
    tw.write_line(f"perf results saved to {out}")
