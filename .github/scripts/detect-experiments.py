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
import re
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

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
# Duplicated pure functions from deplodock (CI script cannot import deplodock)
# ---------------------------------------------------------------------------

# From deplodock/recipe/matrix.py
PARAM_ABBREVIATIONS = {
    "max_concurrency": "c",
    "num_prompts": "n",
    "random_input_len": "in",
    "random_output_len": "out",
    "max_concurrent_requests": "mcr",
    "context_length": "ctx",
}

# From deplodock/hardware.py
GPU_SHORT_NAMES = {
    "NVIDIA GeForce RTX 4090": "rtx4090",
    "NVIDIA GeForce RTX 5090": "rtx5090",
    "NVIDIA RTX PRO 6000 Workstation Edition": "pro6000",
    "NVIDIA RTX PRO 6000 Server Edition": "pro6000",
    "NVIDIA L40S": "l40s",
    "NVIDIA H100 80GB": "h100",
    "NVIDIA H200 141GB": "h200",
    "NVIDIA B200": "b200",
    "NVIDIA A100 40GB": "a100",
    "NVIDIA A100 80GB": "a100",
    "AMD Instinct MI350X": "mi350x",
}


def gpu_short_name(full_name):
    """Map a full GPU name to a short name for filenames.

    Duplicated from deplodock/hardware.py.
    """
    if full_name in GPU_SHORT_NAMES:
        return GPU_SHORT_NAMES[full_name]
    return re.sub(r"[^a-z0-9]", "", full_name.lower())


def dot_to_nested(key, value):
    """Convert a dot-notation key + value into a nested dict.

    Duplicated from deplodock/recipe/matrix.py.
    """
    parts = key.split(".")
    result = current = {}
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars.

    Duplicated from deplodock/recipe/recipe.py.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_matrix_entry(entry):
    """Expand one matrix entry using broadcast + zip semantics.

    Duplicated from deplodock/recipe/matrix.py.
    """
    scalar_keys = {}
    list_keys = {}

    for key, value in entry.items():
        if isinstance(value, list):
            list_keys[key] = value
        else:
            scalar_keys[key] = value

    if not list_keys:
        return [dict(scalar_keys)]

    lengths = {k: len(v) for k, v in list_keys.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        detail = ", ".join(f"{k}={n}" for k, n in lengths.items())
        raise ValueError(f"All lists in a matrix entry must have the same length, got: {detail}")

    n = unique_lengths.pop()
    combinations = []
    for i in range(n):
        combo = dict(scalar_keys)
        for key, values in list_keys.items():
            combo[key] = values[i]
        combinations.append(combo)

    return combinations


def matrix_label(combination, variable_keys):
    """Generate a filename-safe label from variable params of a combination.

    Duplicated from deplodock/recipe/matrix.py.
    """
    if not variable_keys:
        return ""

    parts = []
    for key in sorted(variable_keys):
        value = combination[key]
        last_segment = key.rsplit(".", 1)[-1]
        abbrev = PARAM_ABBREVIATIONS.get(last_segment, last_segment)
        parts.append(f"{abbrev}{value}")

    return "_".join(parts)


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
# Task enumeration (lightweight — no deplodock import)
# ---------------------------------------------------------------------------


def _detect_engine(merged_config):
    """Detect engine name from a merged config dict.

    Returns 'sglang' if engine.llm.sglang is present, otherwise 'vllm'.
    """
    engine = merged_config.get("engine", {})
    llm = engine.get("llm", {})
    if llm.get("sglang") is not None:
        return "sglang"
    return "vllm"


def _build_variant_name(gpu_name, combination, variable_keys):
    """Build auto-generated run identifier from GPU name and matrix combination."""
    short = gpu_short_name(gpu_name)
    label = matrix_label(combination, variable_keys)
    if label:
        return f"{short}_{label}"
    return short


def _build_override(combination):
    """Convert a flat dot-notation combination into a nested dict."""
    result = {}
    for key, value in combination.items():
        nested = dot_to_nested(key, value)
        result = deep_merge(result, nested)
    return result


def enumerate_task_summaries(experiment_dirs):
    """Parse recipe.yaml files and expand matrices into task summaries.

    Returns a list of dicts with keys: experiment, variant, result_file, gpu_short.
    """
    tasks = []
    for exp_dir in experiment_dirs:
        recipe_path = os.path.join(exp_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        matrices = raw.get("matrices", [])
        if not matrices:
            continue

        base_config = {k: v for k, v in raw.items() if k != "matrices"}

        for entry in matrices:
            combinations = expand_matrix_entry(entry)
            variable_keys = {k for k, v in entry.items() if isinstance(v, list)}

            for combo in combinations:
                override = _build_override(combo)
                merged = deep_merge(base_config, override)

                gpu_name = merged.get("deploy", {}).get("gpu")
                if not gpu_name:
                    continue

                variant = _build_variant_name(gpu_name, combo, variable_keys)
                engine = _detect_engine(merged)
                result_file = f"{variant}_{engine}_benchmark.txt"
                short = gpu_short_name(gpu_name)

                # Use the experiment leaf directory name for grouping
                exp_name = os.path.basename(exp_dir)

                tasks.append(
                    {
                        "experiment": exp_name,
                        "variant": variant,
                        "result_file": result_file,
                        "gpu_short": short,
                    }
                )

    return tasks


def format_task_summary(tasks):
    """Format task summaries into a markdown list grouped by GPU short name.

    Returns a markdown string.
    """
    if not tasks:
        return ""

    # Group by gpu_short
    groups = {}
    for task in tasks:
        groups.setdefault(task["gpu_short"], []).append(task)

    lines = []
    for gpu_short in sorted(groups):
        lines.append(f"**{gpu_short}:**")
        for task in groups[gpu_short]:
            lines.append(f"- `{task['experiment']}` — `{task['result_file']}`")
        lines.append("")

    return "\n".join(lines).rstrip()


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

    # Enumerate tasks from recipe matrices
    tasks = enumerate_task_summaries(experiments)
    total_tasks = len(tasks)
    task_summary = format_task_summary(tasks)

    # Console output
    print(f"Detected experiments ({mode}):")
    for exp in experiments:
        print(f"  - {exp}")
    if gpu_concurrency > 1:
        print(f"GPU concurrency: {gpu_concurrency}")
    if warnings:
        for w in warnings:
            print(f"Warning: {w}", file=sys.stderr)
    if total_tasks:
        print(f"Tasks ({total_tasks}):")
        print(task_summary)

    # Write to GITHUB_OUTPUT
    warnings_text = "\n".join(warnings)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"experiments={json.dumps(experiments)}\n")
            f.write(f"gpu_concurrency={gpu_concurrency}\n")
            f.write(f"total_tasks={total_tasks}\n")
            _write_multiline_output(f, "task_summary", task_summary)
            _write_multiline_output(f, "warnings", warnings_text)
    else:
        print(json.dumps(experiments))
        print(f"gpu_concurrency={gpu_concurrency}")
        print(f"total_tasks={total_tasks}")
        if warnings_text:
            print(f"warnings={warnings_text}")


if __name__ == "__main__":
    main()
