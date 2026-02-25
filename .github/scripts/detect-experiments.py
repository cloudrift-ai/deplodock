#!/usr/bin/env python3
"""Detect experiment directories from a /run-experiment comment or git diff.

Two modes:
- Explicit: /run-experiment experiments/Foo/bar experiments/Baz/qux
- Auto-detect: /run-experiment (no args) — finds changed experiments via git diff
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_explicit_paths(comment: str) -> list[str]:
    """Extract experiment paths from the comment text after /run-experiment."""
    parts = comment.strip().split()
    # First token is /run-experiment, rest are paths
    return [p for p in parts[1:] if p.startswith("experiments/")]


def detect_from_diff(base: str, head: str) -> list[str]:
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
        # Walk up from the changed file to find the directory containing recipe.yaml
        path = Path(filepath)
        for parent in [path.parent, *path.parent.parents]:
            if str(parent) == "experiments" or str(parent) == ".":
                break
            if (parent / "recipe.yaml").exists():
                experiment_dirs.add(str(parent))
                break

    return sorted(experiment_dirs)


def validate_experiments(dirs: list[str]) -> list[str]:
    """Filter to directories that contain a recipe.yaml."""
    valid = []
    for d in dirs:
        if Path(d, "recipe.yaml").exists():
            valid.append(d)
        else:
            print(f"Warning: {d} does not contain recipe.yaml, skipping", file=sys.stderr)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Detect experiments for CI benchmark")
    parser.add_argument("--comment", required=True, help="The /run-experiment comment text")
    parser.add_argument("--base", required=True, help="Base ref for git diff (e.g. origin/main)")
    parser.add_argument("--head", required=True, help="Head ref for git diff (e.g. HEAD)")
    args = parser.parse_args()

    # Try explicit paths first
    explicit = parse_explicit_paths(args.comment)
    if explicit:
        experiments = validate_experiments(explicit)
        mode = "explicit"
    else:
        experiments = detect_from_diff(args.base, args.head)
        experiments = validate_experiments(experiments)
        mode = "auto-detect"

    if not experiments:
        print(f"Error: No experiments found ({mode} mode)", file=sys.stderr)
        sys.exit(1)

    print(f"Detected experiments ({mode}):")
    for exp in experiments:
        print(f"  - {exp}")

    # Write to GITHUB_OUTPUT
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"experiments={json.dumps(experiments)}\n")
    else:
        # Running locally — just print the JSON
        print(json.dumps(experiments))


if __name__ == "__main__":
    main()
