"""Benchmark task enumeration."""

import logging
import os

import yaml

from deplodock.planner import BenchmarkTask
from deplodock.planner.variant import Variant
from deplodock.recipe.matrix import build_override, expand_matrix, filter_combinations
from deplodock.recipe.recipe import _validate_and_build, deep_merge

logger = logging.getLogger(__name__)


def enumerate_tasks(recipe_dirs, filters=None) -> list[BenchmarkTask]:
    """Build BenchmarkTask list from recipe dirs using matrices.

    For each recipe dir, reads the raw YAML, extracts the matrices spec,
    expands it via cross/zip combinators, and builds a BenchmarkTask
    per combination.

    Args:
        recipe_dirs: List of recipe directory paths.
        filters: Optional list of (key, glob_pattern) tuples for variant
            filtering (AND logic).
    """
    tasks = []
    for recipe_dir in recipe_dirs:
        recipe_path = os.path.join(recipe_dir, "recipe.yaml")
        if not os.path.isfile(recipe_path):
            logger.warning(f"Warning: No recipe.yaml in {recipe_dir}, skipping.")
            continue

        with open(recipe_path) as f:
            raw = yaml.safe_load(f)

        matrices = raw.get("matrices")
        if not matrices:
            logger.warning(f"Warning: No matrices in {recipe_dir}, skipping.")
            continue

        base_config = {k: v for k, v in raw.items() if k != "matrices"}

        combinations = expand_matrix(matrices)
        if filters:
            combinations = filter_combinations(combinations, filters)

        for combo in combinations:
            override = build_override(combo)
            merged = deep_merge(base_config, override)
            recipe = _validate_and_build(merged)

            if recipe.deploy.gpu is None:
                logger.warning(
                    f"Warning: matrix entry in {recipe_dir} missing 'deploy.gpu', skipping.",
                )
                continue

            variant = Variant(params=combo)
            tasks.append(
                BenchmarkTask(
                    recipe_dir=recipe_dir,
                    variant=variant,
                    recipe=recipe,
                )
            )

    return tasks
