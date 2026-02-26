#!/usr/bin/env python3
"""Detect experiment directories from a /run-experiment comment or git diff.

Two modes:
- Explicit: /run-experiment experiments/Foo/bar experiments/Baz/qux [--flags]
- Auto-detect: /run-experiment (no args) [--flags] — finds changed experiments via git diff

Any --flags in the comment are passed through to `deplodock bench` as-is.
Outputs a complete bench command that the workflow appends --dry-run or
--commit-results to.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_comment(comment):
    """Split a /run-experiment comment into experiment paths and extra tokens.

    Returns (experiments, extra_tokens) where experiments is a list of experiment
    directory strings and extra_tokens is a list of remaining tokens (flags and
    their values, in original order).
    """
    parts = comment.strip().split()
    experiments = []
    extra = []
    for token in parts[1:]:
        if token.startswith("experiments/"):
            experiments.append(token)
        else:
            extra.append(token)
    return experiments, extra


def detect_from_diff(base, head):
    """Find experiment directories with changes between base and head."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        check=True,
    )
    changed_files = result.stdout.strip().splitlines()

    experiment_dirs = set()
    for filepath in changed_files:
        if not filepath.startswith("experiments/"):
            continue
        path = Path(filepath)
        for parent in [path.parent, *path.parent.parents]:
            if str(parent) == "experiments" or str(parent) == ".":
                break
            if (parent / "recipe.yaml").exists():
                experiment_dirs.add(str(parent))
                break

    return sorted(experiment_dirs)


def main():
    parser = argparse.ArgumentParser(description="Detect experiments for CI benchmark")
    parser.add_argument("--comment", required=True, help="The /run-experiment comment text")
    parser.add_argument("--base", required=True, help="Base ref for git diff (e.g. origin/main)")
    parser.add_argument("--head", required=True, help="Head ref for git diff (e.g. HEAD)")
    args = parser.parse_args()

    experiments, extra_tokens = parse_comment(args.comment)

    if len(experiments) > 0:
        mode = "explicit"
    else:
        experiments = detect_from_diff(args.base, args.head)
        mode = "auto-detect"

    if not experiments:
        print(f"Error: No experiments found ({mode} mode)", file=sys.stderr)
        sys.exit(1)

    # Build the full bench command: deplodock bench <dirs> [flags]
    bench_command = " ".join(["deplodock", "bench", *experiments, *extra_tokens])

    print(f"Detected experiments ({mode}):")
    for exp in experiments:
        print(f"  - {exp}")
    print(f"Bench command: {bench_command}")

    # Write to GITHUB_OUTPUT
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"experiments={json.dumps(experiments)}\n")
            f.write(f"bench_command={bench_command}\n")
    else:
        print(json.dumps(experiments))
        print(f"bench_command={bench_command}")


if __name__ == "__main__":
    main()
