"""Per-kernel vs-torch benchmark + chart for a whole model (or one layer).

End-to-end exercise of op provenance: compile the model with a dump (so every
kernel gets a prov name + a ``<kname>.torch.json`` reproducer of the original
Torch ops it implements), then for each kernel run ``deplodock run --ir
<repro> --bench`` to time it against eager PyTorch and ``torch.compile``, and
render a per-kernel bar chart with the shared ``deplodock.visualize`` plotting
(same path the perf tests use).

    # one-shot latency chart (uses whatever's in the autotune DB):
    python scripts/bench_model_kernels.py --model Qwen/Qwen3-Embedding-0.6B --layer 0

    # before/after-tuning comparison (clean isolated DB): bench → tune each
    # kernel → bench again → comparison table + plot:
    python scripts/bench_model_kernels.py --model Qwen/Qwen3-Embedding-0.6B --compare-tuning

``--compare-tuning`` points every step at a fresh per-run autotune DB
(``<dump-dir>/tune.db``) so the "before" numbers are untuned rule defaults and
the "after" numbers reflect this run's tuning only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_BIN = Path(sys.executable).parent
_ROW = re.compile(r"^(Eager PyTorch|torch\.compile|Deplodock)\s+([\d.]+)")


def _run(args: list[str], timeout: int | None = None, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    # Inherits os.environ (where main sets DEPLODOCK_TUNE_DB). DEPLODOCK_DUMP_DIR
    # is passed only to compile — never to bench/tune, whose run --ir would
    # otherwise create a CompilerDump that rmtree's the reproducer dir.
    env = {**os.environ, **extra_env} if extra_env else None
    return subprocess.run([str(_BIN / "deplodock"), *args], capture_output=True, text=True, timeout=timeout, env=env)


def _final_kernels(dump_dir: Path) -> list[Path]:
    """The ``.torch.json`` reproducers from the last (CUDA) stage dump."""
    kernel_dirs = sorted(dump_dir.glob("*.kernels"))
    return sorted(kernel_dirs[-1].glob("*.torch.json")) if kernel_dirs else []


def _bench(repro: Path, backends: str, timeout: int = 600) -> dict[str, float] | None:
    """Bench one reproducer; parse the latency table to ``{backend: us}``.

    Always has ``Deplodock``; ``Eager PyTorch`` / ``torch.compile`` only when the
    reproducer is torch-runnable. ``None`` if the kernel failed to run."""
    proc = _run(["run", "--ir", str(repro), "--bench", "--bench-backends", backends], timeout=timeout)
    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out if "Deplodock" in out else None


def _tune(repro: Path, patience: int, timeout: int = 1200) -> None:
    _run(["tune", str(repro), "--patience", str(patience)], timeout=timeout)


def _short(name: str) -> str:
    """Drop the ``.torch`` suffix + trailing structural hash for a readable label."""
    return re.sub(r"_[0-9a-f]{6}$", "", name.removesuffix(".torch"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--layer", type=int, default=None, help="trace one layer (omit for whole model)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--tune", action="store_true", help="autotune each kernel before benching (one-shot mode)")
    ap.add_argument("--compare-tuning", action="store_true", help="bench → tune each → bench again, with a comparison table")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--backends", default="eager,tcompile,deplodock")
    ap.add_argument("--dump-dir", default="/tmp/deplodock-model-kernels")
    ap.add_argument("--out", default=None, help="chart path (.html); a .png + .md table are written alongside")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    if args.compare_tuning:
        # Isolate the experiment in a fresh DB so "before" is truly untuned.
        # Sibling of the dump dir — not inside it (compile rmtree's the dump dir).
        tune_db = dump_dir.parent / f"{dump_dir.name}-tune.db"
        tune_db.unlink(missing_ok=True)
        os.environ["DEPLODOCK_TUNE_DB"] = str(tune_db)

    compile_args = ["compile", args.model, "--seq-len", str(args.seq_len)]
    if args.layer is not None:
        compile_args += ["--layer", str(args.layer)]
    print(f"[1/4] compiling {args.model} (dump → {dump_dir}) …", flush=True)
    # DEPLODOCK_DUMP_DIR only here — bench/tune must not dump (they'd rmtree it).
    proc = _run(compile_args, timeout=3600, extra_env={"DEPLODOCK_DUMP_DIR": str(dump_dir)})
    if proc.returncode != 0:
        print(f"compile failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)

    repros = _final_kernels(dump_dir)
    if not repros:
        print("no per-kernel reproducers found — did the compile produce CUDA kernels?", file=sys.stderr)
        sys.exit(1)

    if args.compare_tuning:
        _compare_tuning(repros, args)
    else:
        _one_shot(repros, args)


def _one_shot(repros: list[Path], args) -> None:
    print(f"[2/4] benching {len(repros)} kernels vs torch{' (with tuning)' if args.tune else ''} …", flush=True)
    rows: list[tuple[str, dict[str, float]]] = []
    for r in repros:
        if args.tune:
            _tune(r, args.patience)
        res = _bench(r, args.backends)
        label = _short(r.name)
        if res is None:
            print(f"  - {label}: skipped (kernel failed to run)")
            continue
        vs = f" eager={res['Eager PyTorch']:.0f}us" if "Eager PyTorch" in res else " (deplodock-only)"
        print(f"  - {label}: deplodock={res['Deplodock']:.0f}us{vs}")
        rows.append((label, res))
    if not rows:
        print("no kernels produced a benchmark", file=sys.stderr)
        sys.exit(1)
    _render_latency(rows, args)


def _compare_tuning(repros: list[Path], args) -> None:
    # before (untuned rule defaults)
    print(f"[2/4] benching {len(repros)} kernels (before tuning) …", flush=True)
    before: dict[str, dict[str, float]] = {}
    for r in repros:
        res = _bench(r, args.backends)
        if res is not None:
            before[_short(r.name)] = res
            print(f"  - {_short(r.name)}: deplodock={res['Deplodock']:.0f}us", flush=True)

    print(f"[3/4] tuning {len(repros)} kernels (patience={args.patience}) …", flush=True)
    for i, r in enumerate(repros, 1):
        print(f"  - [{i}/{len(repros)}] tuning {_short(r.name)} …", flush=True)
        _tune(r, args.patience)

    print(f"[4/4] benching {len(repros)} kernels (after tuning) …", flush=True)
    after: dict[str, dict[str, float]] = {}
    for r in repros:
        res = _bench(r, args.backends)
        if res is not None:
            after[_short(r.name)] = res
            print(f"  - {_short(r.name)}: deplodock={res['Deplodock']:.0f}us", flush=True)

    labels = [k for k in before if k in after]
    rows = [
        (k, before[k]["Deplodock"], after[k]["Deplodock"], before[k].get("Eager PyTorch"), before[k].get("torch.compile")) for k in labels
    ]
    rows.sort(key=lambda t: t[1], reverse=True)
    _render_comparison(rows, args)


def _fmt(v: float | None) -> str:
    return f"{v:.0f}" if v is not None else "—"


def _render_comparison(rows, args) -> None:
    from deplodock.visualize.bar_chart import Bar, BarChart, render_bar_chart

    out = Path(args.out) if args.out else Path(args.dump_dir) / "kernels.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- comparison table (markdown + stdout) ---
    head = f"# {args.model} — per-kernel tuning comparison\n\n"
    head += "| kernel | before µs | after µs | tune speedup | eager µs | torch.compile µs |\n"
    head += "|---|--:|--:|--:|--:|--:|\n"
    tb, ta = 0.0, 0.0
    lines = []
    for k, b, a, eager, tc in rows:
        sp = b / a if a else 0.0
        tb += b
        ta += a
        lines.append(f"| {k} | {b:.0f} | {a:.0f} | {sp:.2f}× | {_fmt(eager)} | {_fmt(tc)} |")
    total = f"| **TOTAL** | **{tb:.0f}** | **{ta:.0f}** | **{(tb / ta if ta else 0):.2f}×** | | |"
    table = head + "\n".join(lines) + "\n" + total + "\n"
    print("\n" + table)
    md = out.with_suffix(".md")
    md.write_text(table)
    print(f"comparison table → {md}")

    # --- plot: per-kernel before/after/eager latency ---
    chart = BarChart(
        categories=[k for k, *_ in rows],
        bars=[
            Bar("Deplodock (before)", [b for _, b, *_ in rows], color="#868e96"),
            Bar("Deplodock (tuned)", [a for _, _, a, *_ in rows], color="#4dabf7"),
            Bar("Eager PyTorch", [eager for *_, eager, _ in rows], color="#51cf66"),
        ],
        value_name="latency (µs) — lower is faster",
        title=f"{args.model} — per-kernel latency: before vs after tuning",
        subtitle=f"{len(rows)} kernels, benched from their dumped .torch.json reproducers. Tuned with patience={args.patience}.",
        orientation="horizontal",
    )
    html = render_bar_chart(chart, theme="dark", transparent=True)
    out.write_text(html)
    print(f"chart → {out}")
    _write_png(html, out, len(rows))


def _render_latency(rows: list[tuple[str, dict[str, float]]], args) -> None:
    from deplodock.visualize.bar_chart import Bar, BarChart, render_bar_chart

    rows.sort(key=lambda kv: kv[1]["Deplodock"], reverse=True)
    n_vs = sum("Eager PyTorch" in res for _, res in rows)

    def series(key):
        return [res.get(key) for _, res in rows]

    chart = BarChart(
        categories=[label for label, _ in rows],
        bars=[
            Bar("Deplodock", series("Deplodock"), color="#4dabf7"),
            Bar("Eager PyTorch", series("Eager PyTorch"), color="#999999"),
            Bar("torch.compile", series("torch.compile"), color="#ffd166"),
        ],
        value_name="latency (µs) — lower is faster",
        title=f"{args.model} — per-kernel latency",
        subtitle=(f"Each kernel benched from its dumped .torch.json ({len(rows)} kernels; {n_vs} torch-comparable, rest deplodock-only)."),
        orientation="horizontal",
    )
    html = render_bar_chart(chart, theme="dark", transparent=True)
    out = Path(args.out) if args.out else Path(args.dump_dir) / "kernels.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[3/4] chart → {out}")
    _write_png(html, out, len(rows))


def _write_png(html: str, out: Path, n: int) -> None:
    png = out.with_suffix(".png")
    try:
        from deplodock.visualize.image import render as render_image

        render_image(html, png, height=max(300, 40 * n))
        print(f"        png → {png}")
    except Exception as e:  # noqa: BLE001 — PNG is best-effort (needs the [visualize] extra)
        print(f"        png skipped: {e}")


if __name__ == "__main__":
    main()
