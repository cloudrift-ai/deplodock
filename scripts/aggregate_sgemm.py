#!/usr/bin/env python3
"""Aggregate per-point SGEMM benchmark JSONs into comparison tables.

Reads all benchmark JSON traces from a run directory (one per variant),
groups them by GPU and batch count, and produces a combined Markdown report
with per-GPU tables and cross-GPU comparison pivots.

Usage:
    python scripts/aggregate_sgemm.py <run_dir>
    python scripts/aggregate_sgemm.py <run_dir> --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _fmt(v: float | None, spec: str) -> str:
    return format(v, spec) if v is not None else "—"


def _range(pre: int | None, post: int | None) -> str:
    if pre is None and post is None:
        return "—"
    if pre is None:
        return str(post)
    if post is None:
        return str(pre)
    return f"{pre}" if pre == post else f"{pre}→{post}"


def _config_for_size(config: dict, m: int) -> dict:
    """Pick the strategy_map entry whose threshold covers M (adaptive runs)."""
    smap = config.get("strategy_map")
    if not smap:
        return config if isinstance(config, dict) and "strategy" in config else {}
    for threshold, entry in smap:
        if m <= threshold:
            return entry
    return smap[-1][1]


def load_results(run_dir: Path) -> list[dict]:
    """Load all benchmark JSON files from run_dir, expanding multi-result traces."""
    entries = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in ("tasks.json", "instances.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        results = data.get("results", [])
        if not results:
            continue

        system_info = data.get("system_info", {})
        config = data.get("config", {})
        strategy = data.get("strategy", "")

        for r in results:
            dims = r.get("dimensions", {})
            entries.append(
                {
                    "gpu": system_info.get("gpu", "unknown"),
                    "system_info": system_info,
                    "config": config,
                    "strategy": strategy,
                    "m": dims.get("M", 0),
                    "n": dims.get("N", 0),
                    "k": dims.get("K", 0),
                    "batch": dims.get("batch", 1),
                    "kernel_time_ms": r.get("kernel_time_ms"),
                    "kernel_min_ms": r.get("kernel_min_ms"),
                    "kernel_max_ms": r.get("kernel_max_ms"),
                    "cublas_time_ms": r.get("cublas_time_ms"),
                    "cublas_min_ms": r.get("cublas_min_ms"),
                    "cublas_max_ms": r.get("cublas_max_ms"),
                    "gflops": r.get("gflops"),
                    "cublas_gflops": r.get("cublas_gflops"),
                    "efficiency_pct": r.get("efficiency_pct"),
                    "sm_clock_mhz_pre": r.get("sm_clock_mhz_pre"),
                    "sm_clock_mhz_post": r.get("sm_clock_mhz_post"),
                    "gpu_temp_c_pre": r.get("gpu_temp_c_pre"),
                    "gpu_temp_c_post": r.get("gpu_temp_c_post"),
                }
            )

    return entries


def render_detail_table(entries: list[dict], batch: int) -> list[str]:
    """Render a detailed results table for one (gpu, batch) group."""
    lines: list[str] = []
    title = f"Batch = {batch}" if batch > 1 else "Single GEMM (batch=1)"
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| Size | TM | BK | K-splits | Kernel ms | Kernel var %"
        " | cuBLAS ms | cuBLAS var % | Eff vs cuBLAS"
        " | TFLOPS | cuBLAS TFLOPS | Clock MHz | Temp °C |"
    )
    lines.append(
        "|------|----|----|----------|----------:|-------------:"
        "|----------:|-------------:|--------------:"
        "|-------:|--------------:|----------:|--------:|"
    )

    for e in sorted(entries, key=lambda x: x["m"]):
        m, n = e["m"], e["n"]
        size = f"{m}x{n}" + (f"x{batch}" if batch > 1 else "")

        cfg = _config_for_size(e["config"], m)
        tm = cfg.get("thread_m", "—")
        bk = cfg.get("block_k", "—")
        ks = cfg.get("k_splits", "—")

        eff = f"{e['efficiency_pct']:.1f}%" if e["efficiency_pct"] else "—"

        kt, kmin, kmax = e["kernel_time_ms"], e["kernel_min_ms"], e["kernel_max_ms"]
        kvar = f"{(kmax - kmin) / kt * 100:.1f}%" if kt and kmin is not None and kmax is not None else "—"

        ct, cmin, cmax = e["cublas_time_ms"], e["cublas_min_ms"], e["cublas_max_ms"]
        cvar = f"{(cmax - cmin) / ct * 100:.1f}%" if ct and cmin is not None and cmax is not None else "—"

        clk = _range(e["sm_clock_mhz_pre"], e["sm_clock_mhz_post"])
        tmp = _range(e["gpu_temp_c_pre"], e["gpu_temp_c_post"])

        lines.append(
            f"| {size} | {tm} | {bk} | {ks} | "
            f"{_fmt(kt, '.3f')} | {kvar} | "
            f"{_fmt(ct, '.3f')} | {cvar} | {eff} | "
            f"{_fmt(e['gflops'] and e['gflops'] / 1000, '.1f')} | "
            f"{_fmt(e['cublas_gflops'] and e['cublas_gflops'] / 1000, '.1f')} | "
            f"{clk} | {tmp} |"
        )

    lines.append("")
    return lines


def render_comparison_table(entries: list[dict], batch: int, gpus: list[str]) -> list[str]:
    """Render a cross-GPU comparison pivot table for one batch value."""
    lines: list[str] = []
    title = f"batch={batch}" if batch > 1 else "single GEMM"
    lines.append(f"### Cross-GPU Comparison ({title})")
    lines.append("")

    gpu_short = {g: g.split()[-1] if len(g.split()) > 2 else g for g in gpus}
    header = "| Size | " + " | ".join(f"{gpu_short[g]} eff%" for g in gpus) + " |"
    sep = "|------|" + "|".join("--------:" for _ in gpus) + "|"
    lines.append(header)
    lines.append(sep)

    by_gpu_size: dict[tuple[str, int], dict] = {}
    for e in entries:
        by_gpu_size[(e["gpu"], e["m"])] = e

    all_sizes = sorted({e["m"] for e in entries})
    for size in all_sizes:
        row = f"| {size} |"
        for g in gpus:
            e = by_gpu_size.get((g, size))
            if e and e["efficiency_pct"] is not None:
                row += f" {e['efficiency_pct']:.1f}% |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    return lines


def render_report(entries: list[dict]) -> str:
    """Build full Markdown report from all loaded entries."""
    lines: list[str] = []
    lines.append("# SGEMM Benchmark Report (Aggregated)")
    lines.append("")

    # System info from first entry per GPU
    gpus = sorted({e["gpu"] for e in entries})
    seen_gpu_info = set()
    for gpu in gpus:
        si = next((e["system_info"] for e in entries if e["gpu"] == gpu), {})
        if not si or gpu in seen_gpu_info:
            continue
        seen_gpu_info.add(gpu)
        lines.append(f"## System: {gpu}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for k in ("host", "os", "gpu", "compute_cap", "vram", "driver", "cuda", "cublas", "git"):
            if k in si:
                lines.append(f"| {k} | {si[k]} |")
        lines.append("")

    # Group by (gpu, batch)
    batches = sorted({e["batch"] for e in entries})
    by_gpu_batch: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in entries:
        by_gpu_batch[(e["gpu"], e["batch"])].append(e)

    # Cross-GPU comparison tables
    if len(gpus) > 1:
        lines.append("## Cross-GPU Comparison")
        lines.append("")
        for batch in batches:
            batch_entries = [e for e in entries if e["batch"] == batch]
            lines.extend(render_comparison_table(batch_entries, batch, gpus))

    # Per-GPU detailed tables
    for gpu in gpus:
        lines.append(f"## {gpu}")
        lines.append("")
        for batch in batches:
            group = by_gpu_batch.get((gpu, batch), [])
            if group:
                lines.extend(render_detail_table(group, batch))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate SGEMM benchmark results")
    parser.add_argument("run_dir", type=Path, help="Run directory containing benchmark JSONs")
    parser.add_argument("--output", type=Path, default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    entries = load_results(args.run_dir)
    if not entries:
        print(f"No benchmark results found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    report = render_report(entries)

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
