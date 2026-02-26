#!/usr/bin/env python3
"""Format benchmark results as a markdown PR comment.

Reads tasks.json from experiment run directories and reports
task_id + pass/fail status based on whether the result file exists.
"""

import argparse
import json
from pathlib import Path


def find_latest_run_dir(experiment_dir: str) -> Path | None:
    """Find the most recent timestamped run directory in an experiment."""
    exp_path = Path(experiment_dir)
    run_dirs = sorted(exp_path.glob("[0-9][0-9][0-9][0-9]-*/"), reverse=True)
    for run_dir in run_dirs:
        if (run_dir / "tasks.json").exists():
            return run_dir
    return None


def format_comment(experiments: list[str]) -> str:
    """Build a markdown comment summarizing benchmark results."""
    lines = ["### Experiment benchmark results\n"]

    all_succeeded = True
    any_results = False

    for experiment in experiments:
        exp_name = experiment.rstrip("/")
        run_dir = find_latest_run_dir(experiment)

        if not run_dir:
            lines.append(f"#### `{exp_name}`\n")
            lines.append("> No results found.\n")
            all_succeeded = False
            continue

        tasks = json.loads((run_dir / "tasks.json").read_text())

        if not tasks:
            lines.append(f"#### `{exp_name}`\n")
            lines.append("> No tasks found.\n")
            continue

        lines.append(f"#### `{exp_name}`\n")
        lines.append("| Task | Status |")
        lines.append("|------|--------|")

        for t in tasks:
            task_id = t.get("task_id", t.get("result_file", "unknown"))
            result_path = run_dir / t["result_file"]
            if result_path.exists():
                any_results = True
                lines.append(f"| `{task_id}` | pass |")
            else:
                all_succeeded = False
                lines.append(f"| `{task_id}` | fail |")

        lines.append("")

    # Status summary
    if all_succeeded and any_results:
        status = "All benchmarks completed successfully."
    elif any_results:
        status = "Some benchmarks failed (see below)."
    else:
        status = "No benchmark results were produced."

    lines.insert(1, f"\n{status}\n")

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
