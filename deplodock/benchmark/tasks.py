"""Benchmark task enumeration."""

import os
import sys

import yaml

from deplodock.hardware import gpu_short_name
from deplodock.planner import BenchmarkTask
from deplodock.recipe import load_recipe


def enumerate_tasks(recipe_dirs, variants_filter=None):
    """Build BenchmarkTask list from recipe dirs and optional variant filter.

    For each recipe dir, reads the raw variants from recipe.yaml. If
    variants_filter is given, only matching variants are kept (with a
    warning for missing ones). Each selected variant is loaded and
    turned into a BenchmarkTask.
    """
    tasks = []
    for recipe_dir in recipe_dirs:
        recipe_path = os.path.join(recipe_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            print(f"Warning: No recipe.yaml in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        available_variants = list((raw.get("variants") or {}).keys())
        if not available_variants:
            print(f"Warning: No variants in {recipe_dir}, skipping.", file=sys.stderr)
            continue

        if variants_filter is not None:
            selected = []
            for v in variants_filter:
                if v in available_variants:
                    selected.append(v)
                else:
                    print(
                        f"Warning: variant '{v}' not in {recipe_dir} (available: {', '.join(available_variants)}), skipping.",
                        file=sys.stderr,
                    )
            variants_to_run = selected
        else:
            variants_to_run = available_variants

        for variant in variants_to_run:
            recipe = load_recipe(recipe_dir, variant=variant)
            if recipe.gpu is None:
                print(
                    f"Warning: variant '{variant}' in {recipe_dir} missing 'gpu', skipping.",
                    file=sys.stderr,
                )
                continue

            tasks.append(
                BenchmarkTask(
                    recipe_dir=recipe_dir,
                    variant=variant,
                    recipe=recipe,
                    gpu_name=recipe.gpu,
                    gpu_count=recipe.gpu_count,
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
