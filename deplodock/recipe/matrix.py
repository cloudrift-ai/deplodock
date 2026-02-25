"""Matrix expansion: broadcast + zip semantics for benchmark parameter sweeps."""

# Abbreviations for common parameter names in auto-generated run identifiers.
PARAM_ABBREVIATIONS = {
    "max_concurrency": "c",
    "num_prompts": "n",
    "random_input_len": "in",
    "random_output_len": "out",
    "max_concurrent_requests": "mcr",
    "context_length": "ctx",
}


def dot_to_nested(key: str, value) -> dict:
    """Convert a dot-notation key + value into a nested dict.

    Example: dot_to_nested("engine.llm.max_concurrent_requests", 256)
    returns {"engine": {"llm": {"max_concurrent_requests": 256}}}
    """
    parts = key.split(".")
    result = current = {}
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def expand_matrix_entry(entry: dict) -> list[dict]:
    """Expand one matrix entry using broadcast + zip semantics.

    Scalars are broadcast to all runs. Lists are zipped together (all lists
    in one entry must have the same length). Returns a list of flat dicts
    mapping dot-notation keys to scalar values (one dict per run).
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

    # Validate all lists have the same length
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


def matrix_label(combination: dict, variable_keys: set) -> str:
    """Generate a filename-safe label from the variable (list) params of a combination.

    Only variable keys (those that came from lists) contribute to the label.
    Returns empty string for single-point entries (all scalars).
    """
    if not variable_keys:
        return ""

    parts = []
    for key in sorted(variable_keys):
        value = combination[key]
        # Use last path segment for abbreviation lookup
        last_segment = key.rsplit(".", 1)[-1]
        abbrev = PARAM_ABBREVIATIONS.get(last_segment, last_segment)
        parts.append(f"{abbrev}{value}")

    return "_".join(parts)


def build_override(combination: dict) -> dict:
    """Convert a flat dot-notation combination into a nested dict for deep_merge."""
    from deplodock.recipe.recipe import deep_merge

    result = {}
    for key, value in combination.items():
        nested = dot_to_nested(key, value)
        result = deep_merge(result, nested)
    return result
