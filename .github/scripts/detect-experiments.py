#!/usr/bin/env python3
"""Detect experiment directories from a /run-experiment comment or git diff.

Two modes:
- Explicit: /run-experiment experiments/Foo/bar experiments/Baz/qux
- Auto-detect: /run-experiment (no args) — finds changed experiments via git diff
"""

import argparse
import difflib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Known CLI flags for /run-experiment comments
# ---------------------------------------------------------------------------
# Each entry: canonical flag name -> {"type": "int"|"str", "aliases": [...]}
KNOWN_FLAGS = {
    "--gpu-concurrency": {"type": "int", "default": 1, "aliases": ["--gpu-parallelism"]},
}

# Build reverse lookup: alias -> canonical
_ALIAS_TO_CANONICAL = {}
_ALL_KNOWN = set()
for _canon, _meta in KNOWN_FLAGS.items():
    _ALL_KNOWN.add(_canon)
    _ALIAS_TO_CANONICAL[_canon] = _canon
    for _alias in _meta.get("aliases", []):
        _ALL_KNOWN.add(_alias)
        _ALIAS_TO_CANONICAL[_alias] = _canon


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def parse_flags(comment):
    """Parse flags from the comment, resolving aliases and warning on unknowns.

    Returns (parsed_flags, warnings) where parsed_flags maps canonical flag
    names to their values and warnings is a list of warning strings.
    """
    parts = comment.strip().split()
    parsed = {}
    warnings = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if token.startswith("--"):
            if token in _ALIAS_TO_CANONICAL:
                canonical = _ALIAS_TO_CANONICAL[token]
                meta = KNOWN_FLAGS[canonical]
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    raw_value = parts[i + 1]
                    i += 1
                    if meta["type"] == "int":
                        try:
                            parsed[canonical] = max(1, int(raw_value))
                        except ValueError:
                            warnings.append(f"Invalid value `{raw_value}` for `{token}`, expected integer. Using default.")
                    else:
                        parsed[canonical] = raw_value
                else:
                    warnings.append(f"Flag `{token}` requires a value but none was provided.")
            else:
                # Unknown flag — suggest close matches
                all_flags = sorted(_ALL_KNOWN)
                close = difflib.get_close_matches(token, all_flags, n=1, cutoff=0.5)
                if close:
                    warnings.append(f"Unknown flag `{token}` was ignored. Did you mean `{close[0]}`?")
                else:
                    warnings.append(f"Unknown flag `{token}` was ignored.")
                # Skip the value if present
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    i += 1
        i += 1

    # Apply defaults for missing flags
    for canonical, meta in KNOWN_FLAGS.items():
        if canonical not in parsed:
            parsed[canonical] = meta["default"]

    return parsed, warnings


def parse_explicit_paths(comment):
    """Extract experiment paths from the comment text after /run-experiment."""
    parts = comment.strip().split()
    return [p for p in parts[1:] if p.startswith("experiments/")]


# ---------------------------------------------------------------------------
# Experiment detection
# ---------------------------------------------------------------------------


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


def validate_experiments(dirs):
    """Filter to directories that contain a recipe.yaml."""
    valid = []
    for d in dirs:
        if Path(d, "recipe.yaml").exists():
            valid.append(d)
        else:
            print(f"Warning: {d} does not contain recipe.yaml, skipping", file=sys.stderr)
    return valid


def _write_multiline_output(f, name, value):
    """Write a multiline value to GITHUB_OUTPUT using heredoc delimiter."""
    delimiter = f"ghadelim_{uuid.uuid4().hex[:8]}"
    f.write(f"{name}<<{delimiter}\n")
    f.write(value)
    if not value.endswith("\n"):
        f.write("\n")
    f.write(f"{delimiter}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Detect experiments for CI benchmark")
    parser.add_argument("--comment", required=True, help="The /run-experiment comment text")
    parser.add_argument("--base", required=True, help="Base ref for git diff (e.g. origin/main)")
    parser.add_argument("--head", required=True, help="Head ref for git diff (e.g. HEAD)")
    args = parser.parse_args()

    # Parse flags (with alias resolution and unknown-flag warnings)
    parsed_flags, warnings = parse_flags(args.comment)
    gpu_concurrency = parsed_flags["--gpu-concurrency"]

    # Detect experiments
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

    # Console output
    print(f"Detected experiments ({mode}):")
    for exp in experiments:
        print(f"  - {exp}")
    if gpu_concurrency > 1:
        print(f"GPU concurrency: {gpu_concurrency}")
    if warnings:
        for w in warnings:
            print(f"Warning: {w}", file=sys.stderr)

    # Write to GITHUB_OUTPUT
    warnings_text = "\n".join(warnings)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"experiments={json.dumps(experiments)}\n")
            f.write(f"gpu_concurrency={gpu_concurrency}\n")
            _write_multiline_output(f, "warnings", warnings_text)
    else:
        print(json.dumps(experiments))
        print(f"gpu_concurrency={gpu_concurrency}")
        if warnings_text:
            print(f"warnings={warnings_text}")


if __name__ == "__main__":
    main()
