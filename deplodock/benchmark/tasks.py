"""Benchmark task enumeration."""

import os
import sys

import yaml

from deplodock.hardware import gpu_short_name
from deplodock.planner import BenchmarkTask
from deplodock.recipe.matrix import build_override, expand_matrix_entry, matrix_label
from deplodock.recipe.recipe import _validate_and_build, deep_merge


def _build_variant_name(gpu_name, combination, variable_keys):
    """Build auto-generated run identifier from GPU name and matrix combination."""
    short = gpu_short_name(gpu_name)
    label = matrix_label(combination, variable_keys)
    if label:
        return f"{short}_{label}"
    return short


def enumerate_tasks(recipe_dirs):
    """Build BenchmarkTask list from recipe dirs using matrices.

    For each recipe dir, reads the raw YAML, extracts matrices entries,
    expands each entry via broadcast + zip, and builds a BenchmarkTask
    per combination.
    """
    tasks = []
    for recipe_dir in recipe_dirs:
        recipe_path = os.path.join(recipe_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            print(f"Warning: No recipe.yaml in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        matrices = raw.get("matrices", [])
        if not matrices:
            print(f"Warning: No matrices in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        base_config = {k: v for k, v in raw.items() if k != "matrices"}

        for entry in matrices:
            combinations = expand_matrix_entry(entry)

            # Determine which keys were lists (variable keys) for labeling
            variable_keys = {k for k, v in entry.items() if isinstance(v, list)}

            for combo in combinations:
                override = build_override(combo)
                merged = deep_merge(base_config, override)
                recipe = _validate_and_build(merged)

                if recipe.deploy.gpu is None:
                    print(
                        f"Warning: matrix entry in {recipe_dir} missing 'deploy.gpu', skipping.",
                        file=sys.stderr,
                    )
                    continue

                variant_name = _build_variant_name(recipe.deploy.gpu, combo, variable_keys)
                tasks.append(
                    BenchmarkTask(
                        recipe_dir=recipe_dir,
                        variant=variant_name,
                        recipe=recipe,
                        gpu_name=recipe.deploy.gpu,
                        gpu_count=recipe.deploy.gpu_count,
                    )
                )

    return tasks


def _task_meta(task: BenchmarkTask, run_dir, status: str) -> dict:
    """Build a task metadata dict for the manifest."""
    return {
        "recipe": task.recipe_name,
        "variant": task.variant,
        "gpu_name": task.gpu_name,
        "gpu_short": gpu_short_name(task.gpu_name),
        "gpu_count": task.gpu_count,
        "model_name": task.model_name,
        "result_file": str(task.result_path(run_dir).relative_to(run_dir)),
        "status": status,
    }
