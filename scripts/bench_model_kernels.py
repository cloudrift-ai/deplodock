"""Per-kernel vs-torch benchmark + chart for a whole model (or one layer).

End-to-end exercise of op provenance: compile the model with a dump (so every
kernel gets a prov name + a ``<kname>.torch.json`` reproducer of the original
Torch ops it implements), then for each kernel run ``deplodock run --ir
<repro> --bench`` to time it against eager PyTorch and ``torch.compile``, and
render a per-kernel speedup bar chart with the shared ``deplodock.visualize``
plotting (same path the perf tests use).

    python scripts/bench_model_kernels.py --model Qwen/Qwen3-Embedding-0.6B --layer 0
    python scripts/bench_model_kernels.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layer 0 --tune

``--tune`` first autotunes each dumped kernel in isolation (the tuned variant
flows back to a whole-model compile via the shared autotune DB), then benches.
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


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(_BIN / "deplodock"), *args], capture_output=True, text=True, **kw)


def _final_kernels(dump_dir: Path) -> list[Path]:
    """The ``.torch.json`` reproducers from the last (CUDA) stage dump."""
    kernel_dirs = sorted(dump_dir.glob("*.kernels"))
    if not kernel_dirs:
        return []
    return sorted(kernel_dirs[-1].glob("*.torch.json"))


def _bench_one(repro: Path, backends: str, *, tune: bool, patience: int) -> dict[str, float] | None:
    """Tune (optional) then bench one reproducer; parse the latency table.

    Returns ``{backend: us}`` (always has ``Deplodock``; ``Eager PyTorch`` /
    ``torch.compile`` only when the reproducer is torch-runnable — pure-compute
    kernels), or ``None`` if the kernel failed to run at all."""
    if tune:
        _run(["tune", "--ir", str(repro), "--patience", str(patience)], timeout=900)
    proc = _run(["run", "--ir", str(repro), "--bench", "--bench-backends", backends], timeout=600)
    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out if "Deplodock" in out else None


def _short(name: str) -> str:
    """Drop the ``.torch`` + trailing structural hash for a readable label."""
    stem = name.removesuffix(".torch")
    return re.sub(r"_[0-9a-f]{6}$", "", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--layer", type=int, default=None, help="trace one layer (omit for whole model)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--tune", action="store_true", help="autotune each kernel before benching")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--backends", default="eager,tcompile,deplodock")
    ap.add_argument("--dump-dir", default="/tmp/deplodock-model-kernels")
    ap.add_argument("--out", default=None, help="chart path (.html); a .png is written alongside")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    compile_args = ["compile", args.model, "--seq-len", str(args.seq_len)]
    if args.layer is not None:
        compile_args += ["--layer", str(args.layer)]
    print(f"[1/3] compiling {args.model} (dump → {dump_dir}) …")
    env_note = f"DEPLODOCK_DUMP_DIR={dump_dir}"
    proc = _run(compile_args, env={**os.environ, "DEPLODOCK_DUMP_DIR": str(dump_dir)}, timeout=1800)
    if proc.returncode != 0:
        print(f"compile failed ({env_note}):\n{proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)

    repros = _final_kernels(dump_dir)
    if not repros:
        print("no per-kernel reproducers found — did the compile produce CUDA kernels?", file=sys.stderr)
        sys.exit(1)
    print(f"[2/3] benching {len(repros)} kernels vs torch{' (with tuning)' if args.tune else ''} …")

    rows: list[tuple[str, dict[str, float]]] = []
    for r in repros:
        res = _bench_one(r, args.backends, tune=args.tune, patience=args.patience)
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

    _render(rows, args)


def _render(rows: list[tuple[str, dict[str, float]]], args) -> None:
    from deplodock.visualize.bar_chart import Bar, BarChart, render_bar_chart

    # Per-kernel latency (µs), slowest first. Eager / torch.compile overlaid
    # where the reproducer is torch-runnable (pure-compute kernels); deplodock
    # is always present.
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
        subtitle=(
            f"Each kernel benched from its dumped .torch.json ({len(rows)} kernels; "
            f"{n_vs} torch-comparable, rest deplodock-only — const-transforming ops like linear "
            "have no standalone torch reference)."
        ),
        orientation="horizontal",
    )
    html = render_bar_chart(chart, theme="dark", transparent=True)
    out = Path(args.out) if args.out else Path(args.dump_dir) / "kernels.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[3/3] chart → {out}")
    png = out.with_suffix(".png")
    try:
        from deplodock.visualize.image import render as render_image

        render_image(html, png, height=max(300, 40 * len(rows)))
        print(f"        png → {png}")
    except Exception as e:  # noqa: BLE001 — PNG is best-effort (needs the [visualize] extra)
        print(f"        png skipped: {e}")


if __name__ == "__main__":
    main()
