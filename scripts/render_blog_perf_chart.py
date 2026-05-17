"""Render the part-3 blog's per-kernel chart from a clean+tuned bench pair.

Reads two raw perf JSONs produced by ``make bench-kernels-clean`` and
``make bench-kernels-tuned``, writes ``per_kernel_data.{json,csv}`` and
``per_kernel_tuned.html`` to the blog's ``public/`` asset directory.

Usage::

    ./venv/bin/python scripts/render_blog_perf_chart.py \\
        tests/perf/.results/<clean>.json \\
        tests/perf/.results/<tuned>.json \\
        /path/to/cloudrift-landing/public/blog/building-gpu-compiler-from-scratch-3

Styling overrides the ``deplodock.visualize.bar_chart`` defaults so:

- title font is bumped from 14 → 20 (bold) and subtitle 12 → 14
- legend is pushed below the wrapped subtitle (top: 100)
- left margin widened from 200 → 280 so long case names aren't clipped
- top margin widened from 70 → 140 so subtitle + legend fit above the
  first row
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

# repo-root on sys.path so this script runs without ``pip install -e``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deplodock.visualize.bar_chart import Bar, BarChart, _option  # noqa: E402
from deplodock.visualize.page import render_html  # noqa: E402

CLEAN_COLOR = "#4dabf7"
TUNED_COLOR = "#ff6b6b"
TCOMP_COLOR = "#ffd166"


def _merge_rows(clean: dict, tuned: dict) -> list[dict]:
    by_clean = {r["name"]: r for r in clean["rows"]}
    by_tuned = {r["name"]: r for r in tuned["rows"]}
    merged = []
    for name in sorted(set(by_clean) | set(by_tuned)):
        c = by_clean.get(name)
        t = by_tuned.get(name)
        eager_us = c["torch_us"] if c else t["torch_us"]
        tcompile = (c or t).get("torch_compile_us")
        clean_us = c["deplodock_us"] if c else None
        tuned_us = t["deplodock_us"] if t else None
        merged.append({
            "name": name,
            "op": (c or t)["op"],
            "shape": (c or t)["shape"],
            "tags": (c or t).get("tags", []),
            "launches": (c or t).get("launches"),
            "eager_us": eager_us,
            "torch_compile_us": tcompile,
            "deplodock_clean_us": clean_us,
            "deplodock_tuned_us": tuned_us,
            "ratio_clean": (eager_us / clean_us) if clean_us else None,
            "ratio_tuned": (eager_us / tuned_us) if tuned_us else None,
        })
    return merged


def _write_data_files(rows: list[dict], clean: dict, tuned: dict, out_dir: Path) -> None:
    payload = {
        "timestamp_clean_utc": clean["timestamp_utc"],
        "timestamp_tuned_utc": tuned["timestamp_utc"],
        "git_rev_clean": clean.get("git_rev"),
        "git_rev_tuned": tuned.get("git_rev"),
        "gpu": "NVIDIA GeForce RTX 5090 (sm_120, Blackwell)",
        "cuda": "13.0",
        "torch": "2.11.0+cu130",
        "rows": rows,
    }
    (out_dir / "per_kernel_data.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / "per_kernel_data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "name", "op", "shape", "launches",
            "eager_us", "torch_compile_us", "deplodock_clean_us", "deplodock_tuned_us",
            "ratio_clean", "ratio_tuned",
        ])
        for r in rows:
            w.writerow([
                r["name"], r["op"], r["shape"], r["launches"],
                r["eager_us"], r["torch_compile_us"], r["deplodock_clean_us"], r["deplodock_tuned_us"],
                r["ratio_clean"], r["ratio_tuned"],
            ])


def _tcomp_ratio(r: dict) -> float | None:
    tc = r.get("torch_compile_us")
    if tc is None or tc <= 0:
        return None
    return round(r["eager_us"] / tc, 3)


def _tooltip(r: dict) -> str:
    tcr = _tcomp_ratio(r)
    tc_str = f"{r['torch_compile_us']:.1f} µs ({tcr:.2f}×)" if tcr else "—"
    clean_str = f"{r['deplodock_clean_us']:.1f} µs ({r['ratio_clean']:.2f}×)"
    tuned_str = f"{r['deplodock_tuned_us']:.1f} µs ({r['ratio_tuned']:.2f}×)"
    speedup = r["deplodock_clean_us"] / r["deplodock_tuned_us"] if r["deplodock_tuned_us"] else 0
    parts = [
        f"<b>{r['name']}</b>",
        f'<span style="color:#888">{r["shape"]}</span>',
        "",
        f'<span style="color:#999">■</span> eager: {r["eager_us"]:.1f} µs (1.00×)',
        f'<span style="color:{TCOMP_COLOR}">■</span> torch.compile: {tc_str}',
        f'<span style="color:{CLEAN_COLOR}">■</span> deplodock (clean): {clean_str}',
        f'<span style="color:{TUNED_COLOR}">■</span> deplodock (tuned): {tuned_str}',
    ]
    if speedup:
        parts += ["", f"tune speedup: {speedup:.2f}×"]
    return "<br>".join(parts)


def _render_chart(rows: list[dict], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["ratio_tuned"] or 0), reverse=True)
    tooltip_rows = [_tooltip(r) for r in rows]

    chart = BarChart(
        categories=[r["name"] for r in rows],
        bars=[
            Bar(name="deplodock tuned / eager",
                values=[round(r["ratio_tuned"], 3) if r["ratio_tuned"] else None for r in rows],
                color=TUNED_COLOR),
            Bar(name="deplodock clean / eager",
                values=[round(r["ratio_clean"], 3) if r["ratio_clean"] else None for r in rows],
                color=CLEAN_COLOR),
            Bar(name="torch.compile / eager",
                values=[_tcomp_ratio(r) for r in rows],
                color=TCOMP_COLOR),
        ],
        value_name="speedup vs eager (×)",
        title="Per-kernel speedup vs PyTorch eager — clean vs tuned",
        subtitle=(
            "FP32 on RTX 5090. Ratio = eager_us / backend_us (higher is faster). "
            "Eager is the baseline at 1.0 (dashed line). Sorted by tuned ratio: "
            "wins at top, losses at bottom."
        ),
        baseline=1.0,
        baseline_label="1.0× (eager)",
        tooltip_rows=tooltip_rows,
        orientation="horizontal",
        margin={"left": 280, "right": 50, "top": 140, "bottom": 60},
    )

    option = _option(chart, theme_name="dark")
    option["title"]["textStyle"]["fontSize"] = 20
    option["title"]["textStyle"]["fontWeight"] = "bold"
    option["title"]["subtextStyle"]["fontSize"] = 14
    option["title"]["subtextStyle"]["lineHeight"] = 20
    option["legend"]["top"] = 100
    option["legend"]["textStyle"]["fontSize"] = 13

    payload = {
        "option": option,
        "tooltipRows": tooltip_rows,
        "orientation": "horizontal",
        "rowHeight": chart.row_height,
        "n": len(chart.categories),
        "padTop": option["grid"]["top"],
        "padBot": option["grid"]["bottom"],
    }
    body_html = '<div id="chart" style="width:100%;"></div>\n'
    scripts_js = (
        "const PAYLOAD = " + json.dumps(payload) + ";\n"
        "const el = document.getElementById('chart');\n"
        "el.style.height = (PAYLOAD.n * PAYLOAD.rowHeight + PAYLOAD.padTop + PAYLOAD.padBot) + 'px';\n"
        "const chart = echarts.init(el, null, { renderer: 'canvas' });\n"
        "const opt = PAYLOAD.option;\n"
        "opt.tooltip.formatter = (params) => PAYLOAD.tooltipRows[params[0].dataIndex];\n"
        "chart.setOption(opt);\n"
        "window.addEventListener('resize', () => chart.resize());\n"
    )
    html = render_html(body_html=body_html, scripts_js=scripts_js, theme="dark",
                       title=chart.title, transparent=True)
    (out_dir / "per_kernel_tuned.html").write_text(html)


def _print_stats(rows: list[dict]) -> None:
    gm = statistics.geometric_mean
    rc = [r["ratio_clean"] for r in rows if r["ratio_clean"]]
    rt = [r["ratio_tuned"] for r in rows if r["ratio_tuned"]]
    sp = [r["deplodock_clean_us"] / r["deplodock_tuned_us"]
          for r in rows if r["deplodock_clean_us"] and r["deplodock_tuned_us"]]
    print(f"clean: n={len(rc)}  geomean={gm(rc):.3f}  ≥1.0×: {sum(1 for r in rc if r>=1)}/{len(rc)}  max={max(rc):.2f}")
    print(f"tuned: n={len(rt)}  geomean={gm(rt):.3f}  ≥1.0×: {sum(1 for r in rt if r>=1)}/{len(rt)}  max={max(rt):.2f}")
    print(f"clean→tuned: geomean={gm(sp):.3f}  best={max(sp):.2f}×  worst={min(sp):.2f}×")
    imp = sum(1 for s in sp if s > 1.02)
    reg = sum(1 for s in sp if s < 0.98)
    print(f"improved (>1.02×): {imp}  regressed (<0.98×): {reg}  flat: {len(sp) - imp - reg}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clean_json", type=Path)
    p.add_argument("tuned_json", type=Path)
    p.add_argument("out_dir", type=Path, help="blog public asset directory")
    p.add_argument("--copy-raw", action="store_true",
                   help="also copy the two input JSONs as raw_bench_untuned.json / raw_bench_tuned.json")
    args = p.parse_args()

    clean = json.loads(args.clean_json.read_text())
    tuned = json.loads(args.tuned_json.read_text())
    rows = _merge_rows(clean, tuned)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_data_files(rows, clean, tuned, args.out_dir)
    _render_chart(rows, args.out_dir)
    if args.copy_raw:
        shutil.copy(args.clean_json, args.out_dir / "raw_bench_untuned.json")
        shutil.copy(args.tuned_json, args.out_dir / "raw_bench_tuned.json")
    _print_stats(rows)
    print(f"wrote per_kernel_data.json/.csv + per_kernel_tuned.html → {args.out_dir}")


if __name__ == "__main__":
    main()
