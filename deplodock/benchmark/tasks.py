"""Benchmark task enumeration."""

import logging
import os

import yaml

from deplodock.hardware import gpu_short_name
from deplodock.planner import BenchmarkTask
from deplodock.recipe.matrix import build_override, expand_matrix_entry, matrix_label
from deplodock.recipe.recipe import _validate_and_build, deep_merge

logger = logging.getLogger(__name__)


def _build_variant_name(gpu_name, combination, variable_keys):
    """Build auto-generated run identifier from GPU name and matrix combination."""
    short = gpu_short_name(gpu_name)
    label = matrix_label(combination, variable_keys)
    if label:
        return f"{short}_{label}"
    return short


def enumerate_tasks(recipe_dirs) -> list[BenchmarkTask]:
    """Build BenchmarkTask list from recipe dirs using matrices.

    For each recipe dir, reads the raw YAML, extracts matrices entries,
    expands each entry via broadcast + zip, and builds a BenchmarkTask
    per combination.
    """
    tasks = []
    for recipe_dir in recipe_dirs:
        recipe_path = os.path.join(recipe_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            logger.warning(f"Warning: No recipe.yaml in {recipe_dir}, skipping.")
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        matrices = raw.get("matrices", [])
        if not matrices:
            logger.warning(f"Warning: No matrices in {recipe_dir}, skipping.")
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
                    logger.warning(
                        f"Warning: matrix entry in {recipe_dir} missing 'deploy.gpu', skipping.",
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
