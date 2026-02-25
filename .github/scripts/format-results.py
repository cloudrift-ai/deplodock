#!/usr/bin/env python3
"""Format benchmark results as a markdown PR comment.

Reads manifest.json from experiment run directories and extracts metrics
from benchmark result files (throughput, TTFT, TPOT).
"""

import argparse
import json
import re
from pathlib import Path


def find_latest_run_dir(experiment_dir: str) -> Path | None:
    """Find the most recent timestamped run directory in an experiment."""
    exp_path = Path(experiment_dir)
    run_dirs = sorted(exp_path.glob("[0-9][0-9][0-9][0-9]-*/"), reverse=True)
    for run_dir in run_dirs:
        if (run_dir / "manifest.json").exists():
            return run_dir
    return None


def parse_result_file(filepath: Path) -> dict:
    """Extract metrics from a benchmark result .txt file."""
    metrics = {}
    try:
        content = filepath.read_text()
    except FileNotFoundError:
        return metrics

    patterns = {
        "total_throughput": r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def format_comment(experiments: list[str]) -> str:
    """Build a markdown comment summarizing benchmark results."""
    lines = ["### Experiment benchmark results\n"]

    all_succeeded = True
    any_results = False
    failed_tasks = []

    for experiment in experiments:
        exp_name = experiment.rstrip("/")
        run_dir = find_latest_run_dir(experiment)

        if not run_dir:
            lines.append(f"#### `{exp_name}`\n")
            lines.append("> No results found.\n")
            all_succeeded = False
            continue

        manifest = json.loads((run_dir / "manifest.json").read_text())
        tasks = manifest.get("tasks", [])

        if not tasks:
            lines.append(f"#### `{exp_name}`\n")
            lines.append("> No tasks in manifest.\n")
            continue

        completed = [t for t in tasks if t.get("status") == "completed"]
        failed = [t for t in tasks if t.get("status") != "completed"]

        if failed:
            all_succeeded = False
            for t in failed:
                failed_tasks.append(f"`{exp_name}` — {t.get('result_file', 'unknown')} ({t.get('status', 'unknown')})")

        if completed:
            any_results = True
            lines.append(f"#### `{exp_name}`\n")
            lines.append("| Result | GPU | Throughput (tok/s) | TTFT (ms) | TPOT (ms) |")
            lines.append("|--------|-----|--------------------|-----------|-----------|")

            for task in completed:
                result_file = task.get("result_file", "")
                gpu_short = task.get("gpu_short", "?")
                gpu_count = task.get("gpu_count", 1)
                gpu_label = f"{gpu_short} x{gpu_count}" if gpu_count > 1 else gpu_short

                # Result file is relative to the recipe subdir inside the run dir
                # Try both: directly in run_dir, and in recipe subdir
                filepath = run_dir / result_file
                if not filepath.exists():
                    recipe_name = task.get("recipe", "")
                    filepath = run_dir / recipe_name / result_file

                metrics = parse_result_file(filepath)
                throughput = f"{metrics['total_throughput']:.1f}" if "total_throughput" in metrics else "—"
                ttft = f"{metrics['mean_ttft_ms']:.1f}" if "mean_ttft_ms" in metrics else "—"
                tpot = f"{metrics['mean_tpot_ms']:.2f}" if "mean_tpot_ms" in metrics else "—"

                lines.append(f"| `{result_file}` | {gpu_label} | {throughput} | {ttft} | {tpot} |")

            lines.append("")

    # Status summary
    if all_succeeded and any_results:
        status = "All benchmarks completed successfully."
    elif any_results:
        status = "Some benchmarks failed (see below)."
    else:
        status = "No benchmark results were produced."

    lines.insert(1, f"\n{status}\n")

    # Failed tasks section
    if failed_tasks:
        lines.append("#### Failed tasks\n")
        for ft in failed_tasks:
            lines.append(f"- {ft}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Format benchmark results as PR comment")
    parser.add_argument("--experiments", required=True, help="JSON array of experiment directories")
    parser.add_argument("--output", required=True, help="Output file path for the markdown comment")
    args = parser.parse_args()

    experiments = json.loads(args.experiments)
    comment = format_comment(experiments)

    Path(args.output).write_text(comment)
    print(f"Wrote result comment to {args.output}")


if __name__ == "__main__":
    main()
