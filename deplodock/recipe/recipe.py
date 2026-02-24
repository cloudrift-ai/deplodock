"""Recipe loading, deep merge, and legacy migration."""

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


def _migrate_legacy_format(d):
    """Convert old backend.vllm.* format to new engine.llm.* + engine.llm.vllm.* format.

    Also converts model.name → model.huggingface.
    """
    # Migrate model.name → model.huggingface
    model = d.get("model", {})
    if "name" in model and "huggingface" not in model:
        model["huggingface"] = model.pop("name")
        d["model"] = model

    # Migrate backend.vllm.* → engine.llm.* + engine.llm.vllm.*
    backend = d.get("backend")
    if backend is not None:
        vllm_old = backend.get("vllm", {})

        # Fields that belong at engine.llm level (engine-agnostic)
        LLM_FIELDS = {
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "gpu_memory_utilization",
            "context_length",
            "max_concurrent_requests",
        }
        # Fields that belong at engine.llm.vllm level (engine-specific)
        VLLM_FIELDS = {"image", "extra_args"}

        llm = d.get("engine", {}).get("llm", {})
        vllm_new = llm.get("vllm", {})

        for key, value in vllm_old.items():
            if key in LLM_FIELDS:
                llm[key] = value
            elif key in VLLM_FIELDS:
                vllm_new[key] = value

        if vllm_new:
            llm["vllm"] = vllm_new
        if llm:
            d.setdefault("engine", {})["llm"] = llm

        del d["backend"]

    return d


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


def load_recipe(recipe_dir, variant=None):
    """Load recipe.yaml from recipe_dir, optionally deep-merging a variant.

    Returns a Recipe dataclass.
    """
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

    # Migrate legacy format if needed
    config = _migrate_legacy_format(config)

    # Determine engine and validate extra_args
    llm_dict = config.get("engine", {}).get("llm", {})
    if "sglang" in llm_dict:
        engine = "sglang"
        extra_args = llm_dict.get("sglang", {}).get("extra_args", "")
    else:
        engine = "vllm"
        extra_args = llm_dict.get("vllm", {}).get("extra_args", "")
    validate_extra_args(extra_args, engine=engine)

    return Recipe.from_dict(config)
