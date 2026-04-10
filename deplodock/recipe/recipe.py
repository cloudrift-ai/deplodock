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


def validate_docker_options(docker_options: dict) -> None:
    """Raise ValueError if docker_options contains keys managed by the compose template."""
    conflicts = set(docker_options) & _MANAGED_COMPOSE_KEYS
    if conflicts:
        raise ValueError(
            f"docker_options contains keys managed by the compose template: "
            f"{', '.join(sorted(conflicts))}. Remove them from docker_options."
        )


# Keys already rendered by generate_compose() — must not appear in docker_options.
_MANAGED_COMPOSE_KEYS = frozenset(
    {
        "image",
        "container_name",
        "entrypoint",
        "deploy",
        "devices",
        "group_add",
        "volumes",
        "environment",
        "ports",
        "shm_size",
        "ipc",
        "command",
        "healthcheck",
    }
)


def _validate_and_build(config: dict) -> Recipe:
    """Validate extra_args and docker_options, then build Recipe from config dict."""
    has_command = "command" in config and config["command"] is not None
    has_engine_llm = bool(config.get("engine", {}).get("llm"))

    if has_command and has_engine_llm:
        raise ValueError("Recipe must specify exactly one of 'engine.llm' or 'command', not both.")

    if has_command:
        # Command recipes don't go through engine extra_args validation.
        return Recipe.from_dict(config)
    llm_dict = config.get("engine", {}).get("llm", {})
    if "sglang" in llm_dict:
        engine = "sglang"
        extra_args = llm_dict.get("sglang", {}).get("extra_args", "")
    else:
        engine = "vllm"
        extra_args = llm_dict.get("vllm", {}).get("extra_args", "")
    validate_extra_args(extra_args, engine=engine)
    validate_docker_options(llm_dict.get("docker_options", {}))

    return Recipe.from_dict(config)


def load_recipe(recipe_dir):
    """Load recipe.yaml and return base Recipe (no matrix expansion).

    Strips the 'matrices' section before building the Recipe.
    """
    config = _load_raw_config(recipe_dir)
    config.pop("matrices", None)
    return _validate_and_build(config)


def resolve_for_hardware(recipe_dir: str, gpu_name: str, gpu_count: int | None = None) -> "Recipe":
    """Load recipe and resolve the best matrix entry for the given hardware.

    Expands the full matrix (cross/zip), then picks the best match:
    1. Exact match: deploy.gpu == gpu_name AND deploy.gpu_count == gpu_count
    2. Divisible match: deploy.gpu == gpu_name AND gpu_count is a multiple of
       the entry's deploy.gpu_count (for scale-out). Picks the largest entry
       gpu_count that divides evenly.
    3. Name-only match: deploy.gpu == gpu_name (when gpu_count is None)

    If no matrices section exists, returns the base recipe.
    Raises ValueError if no match is found.
    """
    from deplodock.recipe.matrix import build_override, expand_matrix

    config = _load_raw_config(recipe_dir)
    matrices = config.pop("matrices", None)

    if not matrices:
        return _validate_and_build(config)

    combinations = expand_matrix(matrices)

    # Collect combinations matching gpu_name
    available_gpus = set()
    candidates = []
    for combo in combinations:
        gpu = combo.get("deploy.gpu")
        if gpu is not None:
            available_gpus.add(gpu)
        if gpu == gpu_name:
            candidates.append(combo)

    if not candidates:
        raise ValueError(f"No matrix entry matches GPU '{gpu_name}'. Available GPUs: {', '.join(sorted(available_gpus))}")

    # If no gpu_count specified, return first name match
    if gpu_count is None:
        override = build_override(candidates[0])
        merged = deep_merge(config, override)
        return _validate_and_build(merged)

    # Try exact count match first
    for combo in candidates:
        entry_count = combo.get("deploy.gpu_count", 1)
        if entry_count == gpu_count:
            override = build_override(combo)
            merged = deep_merge(config, override)
            return _validate_and_build(merged)

    # Try divisible match: gpu_count is a multiple of entry's count.
    # Pick largest entry count that divides evenly.
    best = None
    best_count = 0
    for combo in candidates:
        entry_count = combo.get("deploy.gpu_count", 1)
        if gpu_count % entry_count == 0 and entry_count > best_count:
            best = combo
            best_count = entry_count

    if best is not None:
        override = build_override(best)
        merged = deep_merge(config, override)
        return _validate_and_build(merged)

    entry_counts = sorted({c.get("deploy.gpu_count", 1) for c in candidates})
    raise ValueError(f"No matrix entry for GPU '{gpu_name}' matches gpu_count={gpu_count}. Available counts: {entry_counts}")
