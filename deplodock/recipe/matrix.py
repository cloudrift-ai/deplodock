"""Matrix expansion: cross-product and zip combinators for benchmark parameter sweeps.

A matrix spec is a dict that can contain:
- Scalar values: broadcast to all combinations
- List values: in cross mode, each is an independent axis (cartesian product);
  in zip mode, all must be same length and are paired element-wise
- ``cross`` key (dict value): nested cross-product combinator
- ``zip`` key (dict value): nested zip combinator

The ``cross``/``zip`` keys are only treated as combinators when their value is
a dict.  A scalar or list value for those keys is treated as a regular parameter.
"""

import itertools
from fnmatch import fnmatch


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


def _expand_cross(node: dict) -> list[dict]:
    """Expand a cross-product node.

    Scalars are broadcast.  Lists become independent axes whose cartesian
    product is computed.  Nested ``zip``/``cross`` sub-dicts are recursed and
    their result becomes one axis.
    """
    broadcast: dict = {}
    axes: list[list[dict]] = []

    for key, value in node.items():
        if key == "cross" and isinstance(value, dict):
            axes.append(_expand_cross(value))
        elif key == "zip" and isinstance(value, dict):
            axes.append(_expand_zip(value))
        elif isinstance(value, list):
            axes.append([{key: v} for v in value])
        else:
            broadcast[key] = value

    if not axes:
        return [dict(broadcast)]

    combinations = []
    for product_tuple in itertools.product(*axes):
        combo = dict(broadcast)
        for d in product_tuple:
            combo.update(d)
        combinations.append(combo)
    return combinations


def _expand_zip(node: dict) -> list[dict]:
    """Expand a zip node.

    Scalars are broadcast.  Lists are zipped element-wise (must all have the
    same length).  Nested ``cross``/``zip`` sub-dicts are recursed and their
    result becomes one zip axis.
    """
    broadcast: dict = {}
    axes: list[list[dict]] = []

    for key, value in node.items():
        if key == "cross" and isinstance(value, dict):
            axes.append(_expand_cross(value))
        elif key == "zip" and isinstance(value, dict):
            axes.append(_expand_zip(value))
        elif isinstance(value, list):
            axes.append([{key: v} for v in value])
        else:
            broadcast[key] = value

    if not axes:
        return [dict(broadcast)]

    lengths = [len(a) for a in axes]
    if len(set(lengths)) != 1:
        detail = ", ".join(f"axis {i} len={n}" for i, n in enumerate(lengths))
        raise ValueError(f"All axes in a zip node must have the same length, got: {detail}")

    n = lengths[0]
    combinations = []
    for i in range(n):
        combo = dict(broadcast)
        for axis in axes:
            combo.update(axis[i])
        combinations.append(combo)
    return combinations


def expand_matrix(matrices) -> list[dict]:
    """Expand a matrices spec into a flat list of parameter combinations.

    Accepts:
    - A dict with a top-level ``cross`` key (dict value) → cross-product
    - A dict with a top-level ``zip`` key (dict value) → zip
    - A plain dict (no cross/zip) → implicit zip
    - A list of dicts → legacy experiment format (each entry is an implicit
      zip, results are concatenated)
    """
    if isinstance(matrices, list):
        # Legacy experiment snapshot format
        result = []
        for entry in matrices:
            result.extend(_expand_zip(entry))
        return result

    if not isinstance(matrices, dict):
        raise TypeError(f"matrices must be a dict or list, got {type(matrices).__name__}")

    if "cross" in matrices and isinstance(matrices["cross"], dict):
        return _expand_cross(matrices["cross"])
    if "zip" in matrices and isinstance(matrices["zip"], dict):
        return _expand_zip(matrices["zip"])
    return _expand_zip(matrices)


def filter_combinations(
    combinations: list[dict],
    filters: list[tuple[str, str]],
) -> list[dict]:
    """Filter combinations by key=glob_pattern pairs (AND logic)."""
    result = []
    for combo in combinations:
        if all(fnmatch(str(combo.get(key, "")), pattern) for key, pattern in filters):
            result.append(combo)
    return result


def build_override(combination: dict) -> dict:
    """Convert a flat dot-notation combination into a nested dict for deep_merge."""
    from deplodock.recipe.recipe import deep_merge

    result = {}
    for key, value in combination.items():
        nested = dot_to_nested(key, value)
        result = deep_merge(result, nested)
    return result
