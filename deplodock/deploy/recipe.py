"""Recipe loading and deep merge."""

import os

import yaml


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_recipe(recipe_dir, variant=None):
    """Load recipe.yaml from recipe_dir, optionally deep-merging a variant."""
    recipe_path = os.path.join(recipe_dir, "recipe.yaml")
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(recipe_path) as f:
        config = yaml.safe_load(f)

    variants = config.pop("variants", {})

    if variant is not None:
        if variant not in variants:
            available = ", ".join(sorted(variants.keys())) if variants else "none"
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {available}"
            )
        config = deep_merge(config, variants[variant])

    return config
