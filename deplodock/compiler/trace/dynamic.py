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
    argument via :func:`build_torch_dynamic_shapes`. Multiple specs may
    share the same ``NAME`` — that's the canonical way to mark several
    inputs as carrying the same symbolic dim (e.g.
    ``input_ids:1`` AND ``attention_mask:2`` AND ``attention_mask:3``
    are all ``seq_len``). The ``(INPUT, AXIS)`` pair must be unique
    across specs to keep each location addressed unambiguously.

    Raises ``ValueError`` with a CLI-friendly message on a bad spec; the
    caller is expected to ``sys.exit(2)`` so the failure surfaces as a
    usage error.
    """
    out: list[tuple[str, str, int]] = []
    if not specs:
        return out
    seen_positions: set[tuple[str, int]] = set()
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
        if (input_name, axis) in seen_positions:
            raise ValueError(f"--dynamic {raw!r}: ({input_name}, axis {axis}) appears more than once")
        seen_positions.add((input_name, axis))
        out.append((name, input_name, axis))
    return out


def build_torch_dynamic_shapes(specs: list[tuple[str, str, int]]) -> dict | None:
    """Convert position specs to a ``torch.export`` ``dynamic_shapes`` dict.

    Two specs that share the same ``NAME`` get the SAME
    ``torch.export.Dim`` instance — torch needs Dim identity (not just
    name) to recognise that ``input_ids.shape[1]`` and
    ``attention_mask.shape[-1]`` are the same symbolic value. Returns
    ``None`` when there are no specs.
    """
    if not specs:
        return None
    import torch

    dims_by_name: dict[str, object] = {}
    out: dict[str, dict[int, object]] = {}
    for name, input_name, axis in specs:
        if name not in dims_by_name:
            dims_by_name[name] = torch.export.Dim(name, min=1, max=4096)
        out.setdefault(input_name, {})[axis] = dims_by_name[name]
    return out


__all__ = ["parse_position_specs", "build_torch_dynamic_shapes"]
