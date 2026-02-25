"""Recipe loading and deep merge."""

import os

import yaml

from deplodock.recipe.engines import banned_extra_arg_flags
from deplodock.recipe.types import Recipe


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_extra_args(extra_args, engine="vllm"):
    """Raise ValueError if extra_args contains flags managed by named recipe fields."""
    banned = banned_extra_arg_flags(engine)
    tokens = extra_args.split()
    found = []
    for token in tokens:
        flag = token.split("=")[0]
        if flag in banned:
            found.append(flag)
    if found:
        raise ValueError(
            f"extra_args contains flags managed by named fields: {', '.join(sorted(found))}. "
            f"Use the corresponding recipe YAML keys instead."
        )


def _load_raw_config(recipe_dir) -> dict:
    """Load recipe.yaml and return raw dict (with matrices still present)."""
    recipe_path = os.path.join(recipe_dir, "recipe.yaml")
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(recipe_path) as f:
        return yaml.safe_load(f)


def _validate_and_build(config: dict) -> Recipe:
    """Validate extra_args and build Recipe from config dict."""
    llm_dict = config.get("engine", {}).get("llm", {})
    if "sglang" in llm_dict:
        engine = "sglang"
        extra_args = llm_dict.get("sglang", {}).get("extra_args", "")
    else:
        engine = "vllm"
        extra_args = llm_dict.get("vllm", {}).get("extra_args", "")
    validate_extra_args(extra_args, engine=engine)

    return Recipe.from_dict(config)


def load_recipe(recipe_dir):
    """Load recipe.yaml and return base Recipe (no matrix expansion).

    Strips the 'matrices' section before building the Recipe.
    """
    config = _load_raw_config(recipe_dir)
    config.pop("matrices", None)
    return _validate_and_build(config)
