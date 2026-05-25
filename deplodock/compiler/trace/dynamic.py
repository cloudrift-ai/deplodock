"""Position-based dynamic-dim spec parsing for ``--dynamic NAME@INPUT:AXIS``.

The trace flow runs ``torch.export.export`` with a
``dynamic_shapes={input: {axis: torch.export.Dim(name)}}`` mapping;
torch's SymInt propagation produces graph nodes whose shapes carry
``Dim(name)`` exactly where the dynamic axis flows. The two helpers
here turn the CLI spec strings into that ``dynamic_shapes`` dict.

Position-based identification (axis ``N`` of input ``X``) eliminates
the value-collision class that bites a naive ``Dim(VALUE) → Dim(NAME)``
rewrite (e.g. ``--seq-len 32`` on a model whose ``num_heads == 32``).
"""

from __future__ import annotations


def parse_position_specs(specs: list[str] | None) -> list[tuple[str, str, int]]:
    """Parse ``--dynamic NAME@INPUT:AXIS`` CLI strings to ``(name, input,
    axis)`` triples.

    These feed straight into ``torch.export.export``'s ``dynamic_shapes``
    argument via :func:`build_torch_dynamic_shapes`. Raises ``ValueError``
    with a CLI-friendly message on a bad spec; the caller is expected to
    ``sys.exit(2)`` so the failure surfaces as a usage error.
    """
    out: list[tuple[str, str, int]] = []
    if not specs:
        return out
    seen: set[str] = set()
    for raw in specs:
        if "@" not in raw or ":" not in raw:
            raise ValueError(f"--dynamic {raw!r}: expected NAME@INPUT:AXIS form (e.g. ``seq_len@x:1``)")
        name, _, locator = raw.partition("@")
        name = name.strip()
        if not name:
            raise ValueError(f"--dynamic {raw!r}: NAME is empty")
        input_name, _, axis_str = locator.rpartition(":")
        input_name = input_name.strip()
        if not input_name:
            raise ValueError(f"--dynamic {raw!r}: INPUT name is empty")
        try:
            axis = int(axis_str)
        except ValueError as e:
            raise ValueError(f"--dynamic {raw!r}: AXIS must be an int, got {axis_str!r}") from e
        if axis < 0:
            raise ValueError(f"--dynamic {raw!r}: AXIS must be ≥ 0, got {axis}")
        if name in seen:
            raise ValueError(f"--dynamic {raw!r}: NAME {name!r} appears more than once")
        seen.add(name)
        out.append((name, input_name, axis))
    return out


def build_torch_dynamic_shapes(specs: list[tuple[str, str, int]]) -> dict | None:
    """Convert position specs to a ``torch.export`` ``dynamic_shapes`` dict.

    Returns ``None`` when there are no position specs — callers pass the
    result straight to ``torch.export.export(..., dynamic_shapes=...)``
    without a guard.
    """
    if not specs:
        return None
    import torch

    out: dict[str, dict[int, object]] = {}
    for name, input_name, axis in specs:
        out.setdefault(input_name, {})[axis] = torch.export.Dim(name, min=1, max=4096)
    return out


__all__ = ["parse_position_specs", "build_torch_dynamic_shapes"]
