"""Recipe loading and deep merge."""

import os

import yaml

# CLI flags already emitted by generate_compose() from named fields or hardcoded values.
# These must not appear in extra_args to avoid duplication.
BANNED_EXTRA_ARG_FLAGS = {
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--max-num-seqs",
    "--trust-remote-code",
    "--host",
    "--port",
    "--model",
    "--served-model-name",
}


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_extra_args(extra_args):
    """Raise ValueError if extra_args contains flags managed by named recipe fields."""
    tokens = extra_args.split()
    found = []
    for token in tokens:
        flag = token.split("=")[0]
        if flag in BANNED_EXTRA_ARG_FLAGS:
            found.append(flag)
    if found:
        raise ValueError(
            f"extra_args contains flags managed by named fields: {', '.join(sorted(found))}. "
            f"Use the corresponding recipe YAML keys instead."
        )


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
            raise ValueError(f"Unknown variant '{variant}'. Available variants: {available}")
        config = deep_merge(config, variants[variant])

    extra_args = config.get("backend", {}).get("vllm", {}).get("extra_args", "")
    validate_extra_args(extra_args)

    return config
