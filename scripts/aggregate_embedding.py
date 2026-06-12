#!/usr/bin/env python3
"""Aggregate embedding-recipe bench results into a stock-vs-deplodock report.

Reads every ``*.json`` result in the run dir (written by ``deplodock bench``),
groups variants by (GPU, random_input_len), pairs the stock-vLLM row with the
deplodock-plugin row by image name, and emits a markdown table per group:
request throughput, token throughput, mean/p99 E2E latency, the
deplodock/stock throughput ratio, and each variant's model_load_and_warmup
time (which contains the deplodock startup compile).

    python scripts/aggregate_embedding.py <run_dir> --output <run_dir>/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _engine_label(result: dict) -> str:
    image = result.get("recipe", {}).get("engine", {}).get("llm", {}).get("vllm", {}).get("image", "")
    return "deplodock" if "deplodock" in image else "stock vllm"


def _fmt(v, suffix="") -> str:
    return f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for path in sorted(args.run_dir.glob("*.json")):
        if path.name == "tasks.json":
            continue
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "metrics" not in result or "recipe" not in result:
            continue
        gpu = result.get("task", {}).get("gpu_short") or result.get("task", {}).get("gpu_name", "?")
        input_len = result["recipe"].get("benchmark", {}).get("random_input_len", 0)
        groups.setdefault((gpu, input_len), {})[_engine_label(result)] = result

    if not groups:
        print(f"no benchmark result JSONs found in {args.run_dir}", file=sys.stderr)
        return 1

    lines = ["# Embedding serving: stock vLLM vs deplodock plugin", ""]
    for (gpu, input_len), rows in sorted(groups.items()):
        lines.append(f"## {gpu} — {input_len} tokens/request")
        lines.append("")
        lines.append("| engine | req/s | tok/s | mean E2EL (ms) | p99 E2EL (ms) | load+warmup (s) |")
        lines.append("|---|---|---|---|---|---|")
        for label in ("stock vllm", "deplodock"):
            r = rows.get(label)
            if r is None:
                lines.append(f"| {label} | — | — | — | — | — |")
                continue
            m = r["metrics"]
            load = (r.get("timing") or {}).get("model_load_and_warmup")
            lines.append(
                f"| {label} | {_fmt(m.get('request_throughput'))} | {_fmt(m.get('total_token_throughput'))} "
                f"| {_fmt(m.get('mean_e2el_ms'))} | {_fmt(m.get('p99_e2el_ms'))} | {_fmt(load)} |"
            )
        a, b = rows.get("deplodock"), rows.get("stock vllm")
        if a and b:
            ra, rb = a["metrics"].get("request_throughput"), b["metrics"].get("request_throughput")
            if ra and rb:
                lines.append("")
                lines.append(f"deplodock/stock request-throughput ratio: **{ra / rb:.2f}x**")
        lines.append("")

    report = "\n".join(lines)
    if args.output:
        args.output.write_text(report)
        print(f"wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
