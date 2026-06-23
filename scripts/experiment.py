#!/usr/bin/env python
"""ML-experiment tracker for the autotuner.

Run one isolated golden-set tune on the live GPU under a named knob configuration and
record a single self-describing JSON document capturing both axes we care about:

  * **search speed** — how much work the MCTS did (total GPU benches, per-op
    benches-to-best, patience stops, wall time);
  * **kernel quality** — how good the configs it landed on are (prior reachability over
    each op's measured leaves, deploy-perf vs the recorded golden, ranking calibration).

Each run uses an isolated prior + tune DB (``/tmp/deplodock-exp-<name>/``) so experiments
are independent and the user's real caches under ``~/.cache/deplodock`` are untouched.
Knobs are varied via the existing levers only (no new ML machinery): ``--patience`` /
``--ucb-c`` / ``--explore-eps`` plus the env-backed ``--o3-tol`` / ``--analytic-tilt`` /
``--catboost-iterations`` / ``--catboost-depth`` / ``--catboost-lr``.

  Run:   ./venv/bin/python scripts/experiment.py --name baseline
         ./venv/bin/python scripts/experiment.py --name lowpat --patience 10 --kernel square.512
         ./venv/bin/python scripts/experiment.py --name re-eval --skip-tune
  Diff:  ./venv/bin/python scripts/experiment.py --compare ml-experiments/A.json ml-experiments/B.json

Records land in ``ml-experiments/`` (``--out``) — distinct from the ``experiments/`` bench-results tree.

Drives the tune **in-process** (reusing ``deplodock.commands.tune``'s ``_tune_targets`` /
``_tune_backend`` / ``_tune_one``) so the search-speed metrics come straight off the result
objects — no stdout parsing, unlike the older ``scripts/tune_golden_set.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "exp"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def _gpu_name() -> str | None:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


# --- pure assembly (unit-testable, no GPU) -------------------------------------------


def build_record(*, name, config_dict, per_shape, quality, timestamp, git_commit, gpu, failures=None) -> dict:
    """Assemble the experiment record from already-collected pieces. Pure — no I/O, no
    GPU — so the schema is testable from synthetic inputs. ``failures`` lists shapes that
    couldn't be tuned/assembled (name + error), kept beside the metrics for provenance."""
    return {
        "name": name,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "gpu": gpu,
        "config": config_dict,
        "search_speed": {
            "total_benches": sum(s["total_benches"] for s in per_shape),
            "total_wall_s": round(sum(s["wall_s"] for s in per_shape), 2),
            "n_shapes": len(per_shape),
            "per_shape": per_shape,
        },
        "kernel_quality": quality,
        "failures": failures or [],
    }


def summarize_quality(prior) -> dict:
    """Structured kernel-quality metrics off the trained prior (no re-bench): reachability
    over each op's measured leaves, deploy-perf vs the recorded golden, ranking calibration,
    golden coverage. Reuses ``prior/diagnostics.py`` building blocks (structured returns)."""
    from deplodock.compiler.pipeline.search.data import Dataset
    from deplodock.compiler.pipeline.search.prior import diagnostics

    groups = Dataset.from_prior(prior).group_by_op()
    ratios = [row[3] for row in diagnostics.reachability(prior, groups)]  # row = (label, best, pick, ratio, n)
    deploy = diagnostics.golden_deploy_perf(prior)  # {golden_name: pick_us / golden_us}
    dvals = list(deploy.values())
    try:
        rho = diagnostics._calibration(prior, groups)
    except ImportError:  # scipy missing → calibration is optional
        rho = None
    covered, total = diagnostics._golden_coverage(groups)
    return {
        "reachability": {
            "mean_ratio": round(statistics.mean(ratios), 4) if ratios else None,
            "median_ratio": round(statistics.median(ratios), 4) if ratios else None,
            "worst_ratio": round(max(ratios), 4) if ratios else None,
            "n_ops": len(ratios),
        },
        "deploy_perf": {
            "median_vs_golden": round(statistics.median(dvals), 4) if dvals else None,
            "n_better_than_golden": sum(1 for v in dvals if v < 1.0),
            "n_shapes": len(dvals),
            "per_shape": {k: round(v, 4) for k, v in sorted(deploy.items())},
        },
        "calibration_rho": round(rho, 4) if rho is not None else None,
        "golden_coverage": [covered, total],
    }


# --- the run (GPU) -------------------------------------------------------------------


def _build_tune_args(args):
    """A fully-populated ``tune`` args namespace for ``--dataset golden`` + the CLI knob
    overrides, built off the real ``register_tune_command`` parser so every default exists."""
    from deplodock.commands.tune import register_tune_command

    p = argparse.ArgumentParser()
    register_tune_command(p.add_subparsers())
    argv = ["tune", "--dataset", "golden"]
    if args.patience is not None:
        argv += ["--patience", str(args.patience)]
    if args.ucb_c is not None:
        argv += ["--ucb-c", str(args.ucb_c)]
    if args.explore_eps is not None:
        argv += ["--explore-eps", str(args.explore_eps)]
    if args.kernel:
        argv += ["--kernel", args.kernel]
    return p.parse_args(argv)


def _run_tune(tune_args) -> tuple[list[dict], list[dict]]:
    """Mirror ``handle_tune``'s setup + per-target loop (minus its terminal ``os._exit``)
    so we can read each shape's ``TwoLevelResult`` and collect search-speed metrics.
    Returns ``(per_shape, failures)`` — a sweep is robust to a single shape that fails to
    tune/assemble (e.g. a cold greedy pick whose tile blows the smem budget on this card):
    it's recorded under ``failures`` and skipped, so the baseline still covers the rest."""
    from deplodock.commands.compile import apply_nvcc_flags, resolve_tune_db, setup_pipeline_runtime
    from deplodock.commands.tune import _tune_backend, _tune_one, _tune_targets
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.search import SearchDB

    setup_pipeline_runtime(tune_args)
    apply_nvcc_flags(tune_args, default="-Xcicc -O1")  # tune ranks at -O1; winners re-benched at -O3 into the prior
    db = SearchDB(path=resolve_tune_db())
    backends = [_tune_backend()]
    ctx = Context.probe()
    targets = _tune_targets(tune_args)
    sys.stderr.write(f"[exp] tuning {len(targets)} golden shape(s) into {resolve_tune_db()}\n")

    per_shape: list[dict] = []
    failures: list[dict] = []
    for i, (label, code, inp, dyn) in enumerate(targets, 1):
        tune_args.code, tune_args.input, tune_args.dynamic = code, inp, dyn
        sys.stderr.write(f"\n[exp] === {i}/{len(targets)}: {label} ===\n")
        t0 = time.monotonic()
        try:
            result, _ = _tune_one(tune_args, backends=backends, db=db, ctx=ctx, dump=None)
        except RuntimeError as exc:  # saturated bench queue → parent stream dirty; stop, keep partial
            sys.stderr.write(f"[exp] aborted at {label}: {exc} — recording partial results\n")
            failures.append({"name": label, "error": " ".join(str(exc).split())[:300]})
            break
        except Exception as exc:  # noqa: BLE001 — one shape failed to tune/assemble; skip it, keep sweeping
            sys.stderr.write(f"[exp] skipped {label}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}\n")
            failures.append({"name": label, "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"})
            continue
        wall = time.monotonic() - t0
        if result.best_reward is None:
            continue
        per_op = [
            {
                "name": r.name,
                "benches": r.benches,
                "benches_to_best": r.benches_to_best,
                "stop_reason": r.stop_reason,
                "best_us": r.best_us,
            }
            for r in result.best_reward.per_op
        ]
        per_shape.append({"name": label, "wall_s": round(wall, 2), "total_benches": result.best_reward.total_benches, "per_op": per_op})
    return per_shape, failures


def run_experiment(args) -> None:
    name = args.name
    exp_dir = Path(tempfile.gettempdir()) / f"deplodock-exp-{_slug(name)}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    # Isolate the prior + tune DB so each experiment is independent and the user's real
    # caches stay untouched. Deterministic per name so ``--skip-tune`` re-evaluates the
    # same isolated prior.
    os.environ["DEPLODOCK_PRIOR_FILE"] = str(exp_dir / "prior.json")
    os.environ["DEPLODOCK_TUNE_DB"] = str(exp_dir / "autotune.db")
    # Env-only knobs (no tune CLI flag). patience / ucb-c / explore-eps ride the tune args.
    if args.o3_tol is not None:
        os.environ["DEPLODOCK_O3_TOL"] = str(args.o3_tol)
    if args.analytic_tilt is not None:
        os.environ["DEPLODOCK_ANALYTIC_TILT"] = str(args.analytic_tilt)
    for env, val in (
        ("DEPLODOCK_CATBOOST_ITERATIONS", args.catboost_iterations),
        ("DEPLODOCK_CATBOOST_DEPTH", args.catboost_depth),
        ("DEPLODOCK_CATBOOST_LR", args.catboost_lr),
    ):
        if val is not None:
            os.environ[env] = str(val)

    from deplodock import config, storage
    from deplodock.compiler.pipeline.search.prior import CatBoostPrior

    tune_args = _build_tune_args(args)
    per_shape, failures = ([], []) if args.skip_tune else _run_tune(tune_args)

    config_dict = {
        "patience": tune_args.patience if tune_args.patience is not None else config.tune_patience(50),
        "ucb_c": tune_args.ucb_c,
        "explore_eps": tune_args.explore_eps if tune_args.explore_eps is not None else config.tune_eps(0.0),
        "o3_tol": config.o3_tol(),
        "analytic_tilt": config.analytic_tilt(),
        "catboost": config.catboost_params(),
        "kernel_filter": args.kernel,
    }
    prior = CatBoostPrior.load()  # the isolated, just-trained (and checkpointed) prior
    quality = summarize_quality(prior)
    record = build_record(
        name=name,
        config_dict=config_dict,
        per_shape=per_shape,
        quality=quality,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        gpu=_gpu_name(),
        failures=failures,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    rec_path = out_dir / f"{stamp}-{_slug(name)}.json"
    storage.write_json(rec_path, record)
    with open(out_dir / "index.jsonl", "a") as f:  # noqa: PTH123 — append-only run index
        ss, kq = record["search_speed"], record["kernel_quality"]
        f.write(
            json.dumps(
                {
                    "file": rec_path.name,
                    "name": name,
                    "timestamp": record["timestamp"],
                    "total_benches": ss["total_benches"],
                    "total_wall_s": ss["total_wall_s"],
                    "median_reachability": kq["reachability"]["median_ratio"],
                    "median_vs_golden": kq["deploy_perf"]["median_vs_golden"],
                }
            )
            + "\n"
        )
    sys.stderr.write(f"\n[exp] wrote {rec_path}\n")
    _print_run_summary(record)
    # Benching left daemon NVRTC workers holding the CUDA context; os._exit dodges the
    # cupy atexit pool-teardown deadlock (same reason ``tune`` does it).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _print_run_summary(rec: dict) -> None:
    ss, kq = rec["search_speed"], rec["kernel_quality"]
    print(f"\n== experiment '{rec['name']}' ({rec['git_commit']}) ==")
    print(f"  search:  {ss['total_benches']} benches over {ss['n_shapes']} shape(s), {ss['total_wall_s']:.1f}s wall")
    r = kq["reachability"]
    print(f"  quality: reachability median {r['median_ratio']} / worst {r['worst_ratio']} over {r['n_ops']} ops")
    d = kq["deploy_perf"]
    print(f"           deploy vs golden median {d['median_vs_golden']} ({d['n_better_than_golden']}/{d['n_shapes']} beat golden)")
    print(f"           calibration ρ {kq['calibration_rho']}, golden coverage {kq['golden_coverage'][0]}/{kq['golden_coverage'][1]}")
    if rec.get("failures"):
        print(f"  failures: {len(rec['failures'])} shape(s) skipped — {', '.join(f['name'] for f in rec['failures'])}")


# --- compare two records -------------------------------------------------------------


def _compare(path_a: str, path_b: str) -> None:
    from deplodock import storage

    a, b = storage.read_json(Path(path_a)), storage.read_json(Path(path_b))
    if a is None or b is None:
        sys.exit(f"could not read {path_a if a is None else path_b}")

    def _btb_median(rec):
        vals = [op["benches_to_best"] for s in rec["search_speed"]["per_shape"] for op in s["per_op"] if op["benches_to_best"] is not None]
        return statistics.median(vals) if vals else None

    sa, sb = a["search_speed"], b["search_speed"]
    ra, rb = a["kernel_quality"]["reachability"], b["kernel_quality"]["reachability"]
    da, db = a["kernel_quality"]["deploy_perf"], b["kernel_quality"]["deploy_perf"]
    print(f"{'metric':24s} {'A: ' + a['name']:>16s} {'B: ' + b['name']:>16s}  delta")
    print(f"  A = {path_a}\n  B = {path_b}\n")
    rows = [
        ("total benches", sa["total_benches"], sb["total_benches"]),
        ("total wall (s)", sa["total_wall_s"], sb["total_wall_s"]),
        ("median benches-to-best", _btb_median(a), _btb_median(b)),
        ("median reachability", ra["median_ratio"], rb["median_ratio"]),
        ("worst reachability", ra["worst_ratio"], rb["worst_ratio"]),
        ("median vs golden", da["median_vs_golden"], db["median_vs_golden"]),
        ("# beat golden", da["n_better_than_golden"], db["n_better_than_golden"]),
        ("calibration ρ", a["kernel_quality"]["calibration_rho"], b["kernel_quality"]["calibration_rho"]),
    ]
    for label, va, vb in rows:
        delta = f"{vb - va:+.4g}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else "—"
        print(f"{label:24s} {str(va):>16s} {str(vb):>16s}  {delta}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="experiment label (required unless --compare)")
    ap.add_argument("--out", default="ml-experiments", help="directory for record JSONs + index.jsonl (default: ml-experiments/)")
    ap.add_argument("--kernel", default=None, help="narrow the golden set to shapes whose name contains this substring")
    ap.add_argument("--skip-tune", action="store_true", help="reuse the existing isolated prior for this --name; re-evaluate quality only")
    # Knob overrides (the existing levers only). None → use the default / env.
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--ucb-c", type=float, default=None)
    ap.add_argument("--explore-eps", type=float, default=None)
    ap.add_argument("--o3-tol", type=float, default=None)
    ap.add_argument("--analytic-tilt", type=float, default=None)
    ap.add_argument("--catboost-iterations", type=int, default=None)
    ap.add_argument("--catboost-depth", type=int, default=None)
    ap.add_argument("--catboost-lr", type=float, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="diff two experiment records and exit")
    args = ap.parse_args()

    if args.compare:
        _compare(*args.compare)
        return
    if not args.name:
        ap.error("--name is required (unless --compare)")
    run_experiment(args)


if __name__ == "__main__":
    main()
