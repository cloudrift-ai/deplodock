"""Conftest for ``tests/perf/``: marker gating, ``bench_pair`` fixture,
session-end summary table, and JSON dump.

Tests in this directory carry ``pytestmark = [pytest.mark.perf,
requires_cuda]``. The ``perf`` marker is **deselected by default** —
plain ``pytest tests/`` skips them so ``make test`` stays fast. Run
explicitly with ``pytest -m perf`` (or ``make bench-kernels``).

The ``bench_pair`` fixture drives ``deplodock run --bench`` (and
``--profile`` when ``DEPLODOCK_BENCH_NCU=1``) as a subprocess per
``Case`` and parses the dump files (``DEPLODOCK_DUMP_DIR``-rooted) to
build a ``PerfRow``. Reusing the CLI's bench infra keeps the
torch / torch.compile / deplodock comparison and the ncu metrics on
the same code path users invoke directly. Iteration count comes from
``case.iters`` (heavy cases at 20, others at 100).

After the session, ``pytest_terminal_summary`` prints a table sorted
by ratio (worst losses first) and writes the same data to
``tests/perf/.results/<utc-timestamp>.json``. With ncu enabled, extra
columns (occupancy, bank conflicts, SM/DRAM/FMA throughput, regs) are
appended.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.compiler.conftest import requires_cuda  # noqa: F401  (re-exported)
from tests.perf.cases import Case

_RESULTS_DIR = Path(__file__).resolve().parent / ".results"

# Cross-process advisory lock around the GPU iter loop in the
# subprocess-driven bencher. Set on conftest import so every spawned
# ``deplodock run --bench`` (across xdist workers and inside each
# worker's per-case subprocess) coordinates on the same path. Trace,
# compile, dump-write all run unlocked — only the kernel-launch phase
# serializes. Override with ``DEPLODOCK_GPU_LOCK`` before invoking
# ``make bench-kernels`` if a different path is desired.
os.environ.setdefault("DEPLODOCK_GPU_LOCK", "/tmp/deplodock-gpu.lock")


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
    iters: int = 100
    torch_compile_us: float | None = None
    # Per-kernel ncu metrics keyed by kernel name. Populated only when
    # the optional ncu pass runs (``DEPLODOCK_BENCH_NCU=1``). Each entry
    # maps metric-name → numeric value (units are baked into the metric
    # name; see ``deplodock.commands.run._NCU_METRICS``). ``None`` when
    # not collected.
    ncu: dict[str, dict[str, float]] | None = None


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


def _ncu_enabled() -> bool:
    if os.environ.get("DEPLODOCK_NCU_CHILD"):
        return False
    if os.environ.get("DEPLODOCK_BENCH_NCU", "") not in ("1", "true", "True"):
        return False
    return shutil.which("ncu") is not None


@pytest.fixture
def bench_pair(request):
    """Return a callable ``run(case, *, warmup, iters) -> PerfRow``.

    Each call spawns ``deplodock run --code <case.code> --bench
    [--profile]`` with a fresh ``DEPLODOCK_DUMP_DIR`` and parses the
    resulting JSON / CSV. The CLI's ``--bench`` path interleaves
    ``Eager PyTorch`` / ``torch.compile`` / ``Deplodock`` per iter so
    all three see the same warm GPU state. ``--profile`` (gated by
    ``DEPLODOCK_BENCH_NCU=1``) runs ncu once with a curated metric set
    and dumps the per-kernel CSV/JSON to the same dir.

    Does not assert on the ratio — the suite tracks performance, it
    doesn't gate on it.
    """

    def _run(case: Case, *, warmup: int = 10, iters: int | None = None) -> PerfRow:
        if iters is None:
            iters = case.iters

        row = _bench_via_subprocess(case, warmup=warmup, iters=iters, profile=_ncu_enabled())
        _collector(request.config).append(row)
        return row

    return _run


def _bench_via_subprocess(case: Case, *, warmup: int, iters: int, profile: bool) -> PerfRow:
    """Spawn ``deplodock run --bench`` for one case and harvest the dumps."""
    with tempfile.TemporaryDirectory(prefix=f"deplodock_bench_{case.name}_") as tmp:
        cmd = [
            sys.executable,
            "-m",
            "deplodock.deplodock",
            "run",
            "--code",
            case.code,
            "--bench",
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ]
        if profile:
            cmd.append("--profile")
        env = dict(os.environ)
        env["DEPLODOCK_DUMP_DIR"] = tmp
        env.pop("DEPLODOCK_NCU_CHILD", None)

        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        if res.returncode != 0:
            # Surface stderr so flaky-bench cases are diagnosable, but
            # synthesize a minimal row so the suite doesn't abort.
            sys.stderr.write(f"[bench {case.name}] exit={res.returncode}\n{res.stderr[-2000:]}\n")
            return PerfRow(
                name=case.name,
                op=case.op,
                shape=case.shape_str,
                dtype=case.dtype,
                torch_us=0.0,
                deplodock_us=0.0,
                ratio=0.0,
                launches=0,
                tags=case.tags,
                iters=iters,
            )

        bench = json.loads(Path(tmp, "60_bench_compare.json").read_text())
        backends = bench["backends"]
        torch_us = float(backends.get("Eager PyTorch", {}).get("latency_us", 0) or 0)
        depl_us = float(backends.get("Deplodock", {}).get("latency_us", 0) or 0)
        tcompile_us_raw = backends.get("torch.compile", {}).get("latency_us")
        tcompile_us = float(tcompile_us_raw) if tcompile_us_raw is not None else None

        launches = 0
        bench_path = Path(tmp, "60_benchmark.json")
        if bench_path.exists():
            launches = int(json.loads(bench_path.read_text()).get("num_launches", 0))

        ncu: dict[str, dict[str, float]] | None = None
        ncu_path = Path(tmp, "61_ncu_metrics.json")
        if ncu_path.exists():
            ncu = json.loads(ncu_path.read_text())

        ratio = (torch_us / depl_us) if depl_us > 0 else 0.0
        return PerfRow(
            name=case.name,
            op=case.op,
            shape=case.shape_str,
            dtype=case.dtype,
            torch_us=torch_us,
            deplodock_us=depl_us,
            ratio=ratio,
            launches=launches,
            tags=case.tags,
            iters=iters,
            torch_compile_us=tcompile_us,
            ncu=ncu,
        )


# ---------------------------------------------------------------------------
# Aggregate ncu metrics for the summary table
# ---------------------------------------------------------------------------


def _aggregate_ncu(ncu: dict[str, dict[str, float]] | None) -> dict[str, float]:
    """Per-row aggregate of the per-kernel ncu metrics for the summary
    table. Sum durations and conflicts; time-weight the percentage
    metrics so a fast minor kernel doesn't drag the average."""
    if not ncu:
        return {}
    total_ns = sum(m.get("gpu__time_duration.sum", 0.0) for m in ncu.values())
    total_conflicts_ld = sum(m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 0.0) for m in ncu.values())
    total_conflicts_st = sum(m.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 0.0) for m in ncu.values())
    total_lsu_inst = sum(m.get("smsp__inst_executed_pipe_lsu.sum", 0.0) for m in ncu.values())

    def _wavg(metric: str) -> float:
        if total_ns <= 0:
            return 0.0
        num = sum(m.get(metric, 0.0) * m.get("gpu__time_duration.sum", 0.0) for m in ncu.values())
        return num / total_ns

    return {
        "ncu_us": total_ns / 1000.0,
        "occ_pct": _wavg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "sm_pct": _wavg("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "fma_pct": _wavg("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
        "dram_pct": _wavg("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "conflicts": total_conflicts_ld + total_conflicts_st,
        "lsu_inst": total_lsu_inst,
        "regs": max((m.get("launch__registers_per_thread", 0.0) for m in ncu.values()), default=0.0),
    }


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def _format_table(rows: list[PerfRow]) -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r.ratio)  # losses first
    has_ncu = any(r.ncu for r in rows)
    has_compile = any(r.torch_compile_us is not None for r in rows)

    base_headers = ["case", "shape", "torch_us"]
    if has_compile:
        base_headers.append("tcomp_us")
    base_headers += ["depl_us", "ratio", "launches", "iters"]
    ncu_headers = ["occ%", "sm%", "fma%", "dram%", "conflicts", "regs"] if has_ncu else []
    headers = tuple(base_headers + ncu_headers)

    aggregates = [_aggregate_ncu(r.ncu) for r in rows]

    def _row_cells(r: PerfRow, agg: dict[str, float]) -> tuple[str, ...]:
        base = [r.name, r.shape, f"{r.torch_us:>8.1f}"]
        if has_compile:
            base.append(f"{r.torch_compile_us:>8.1f}" if r.torch_compile_us is not None else "—")
        base += [
            f"{r.deplodock_us:>7.1f}",
            f"{r.ratio:>5.2f}x",
            f"{r.launches:>8d}",
            f"{r.iters:>5d}",
        ]
        if not has_ncu:
            return tuple(base)
        if not agg:
            return tuple(base + ["—"] * len(ncu_headers))
        return tuple(
            base
            + [
                f"{agg['occ_pct']:>4.0f}",
                f"{agg['sm_pct']:>4.0f}",
                f"{agg['fma_pct']:>4.0f}",
                f"{agg['dram_pct']:>5.0f}",
                f"{int(agg['conflicts']):>9,d}",
                f"{int(agg['regs']):>4d}",
            ]
        )

    body = [_row_cells(r, a) for r, a in zip(rows, aggregates, strict=True)]
    cells = [headers] + body
    widths = [max(len(c) for c in col) for col in zip(*cells, strict=True)]

    def _fmt(row_cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row_cells, widths, strict=True))

    lines = [_fmt(headers), _fmt(tuple("-" * w for w in widths))]
    for r_cells in body:
        lines.append(_fmt(r_cells))
    return "\n".join(lines)


def pytest_sessionfinish(session, exitstatus):
    """Worker-side hand-off for ``pytest-xdist`` runs.

    Each xdist worker has its own ``config._perf_rows``; without this
    hook only the controller's (empty) list reaches the terminal
    summary. ``workeroutput`` is xdist's built-in dict for
    worker→controller payloads — the controller picks it back up in
    ``pytest_testnodedown``.
    """
    rows = getattr(session.config, "_perf_rows", [])
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None and rows:
        # Each row is a dataclass; serialize via asdict so the
        # controller can rehydrate without sharing the class import.
        workeroutput["perf_rows"] = [{**asdict(r), "tags": list(r.tags)} for r in rows]


def pytest_testnodedown(node, error):
    """Controller-side: drain a finished xdist worker's rows into the
    controller's collector so ``pytest_terminal_summary`` sees all
    cases (not just whichever ones happened to land on the controller).
    """
    payload = getattr(node, "workeroutput", {}).get("perf_rows", [])
    if not payload:
        return
    rows = _collector(node.config)
    for r in payload:
        rows.append(PerfRow(**{**r, "tags": tuple(r.get("tags", ()))}))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rows: list[PerfRow] = getattr(config, "_perf_rows", [])
    if not rows:
        return
    tw = terminalreporter

    tw.write_sep("=", "perf summary (sorted by ratio; >1.00x means deplodock wins)")
    tw.write_line(_format_table(rows))

    # Persist JSON for cross-run diffing. ncu metrics (when collected)
    # are nested under each row's ``ncu`` field; aggregated convenience
    # values are also written so downstream tooling doesn't need to
    # duplicate the time-weighted-average reduction.
    _RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    out = _RESULTS_DIR / f"{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "git_rev": os.environ.get("DEPLODOCK_GIT_REV", ""),
        "rows": [
            {
                **asdict(r),
                "tags": list(r.tags),
                "ncu_aggregate": _aggregate_ncu(r.ncu),
            }
            for r in rows
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    tw.write_line(f"perf results saved to {out}")
