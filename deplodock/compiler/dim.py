"""``Dim`` — one tensor / axis extent.

Static (``Dim(32)``) or symbolic (``Dim("seq_len")``). Symbolic values
resolve to an ``int`` only at runtime, from the actual input shapes.

Reads:

- ``d.value`` always works — returns the underlying ``int | str``.
- ``d.as_static()`` returns the int, or raises if the dim is symbolic.
- ``d == 32`` / ``d == "seq_len"`` works for migration ergonomics.

No ``__int__`` / ``__index__``: ``int(d)`` and ``range(d)`` deliberately
fail. Sites that need a static int must say so via ``.as_static()`` —
that way introducing a symbolic dim later breaks loudly at the sites
that can't yet handle it, rather than silently casting through.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class Dim:
    value: int | str

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, str)):
            raise TypeError(f"Dim.value must be int or str, got {type(self.value).__name__}: {self.value!r}")

    @property
    def is_static(self) -> bool:
        return isinstance(self.value, int)

    def as_static(self) -> int:
        if not isinstance(self.value, int):
            raise TypeError(f"Dim({self.value!r}) is symbolic — cannot resolve to int statically")
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dim):
            return self.value == other.value
        if isinstance(other, (int, str)):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"Dim({self.value!r})"

    def __str__(self) -> str:
        # Used by f-strings / ``str()`` — render transparently so pretty IR
        # output (``for i in 0..32``) and ``Body.structural_key`` digests
        # don't change shape just because the type wrapper exists. Use
        # ``repr(d)`` (→ ``Dim(32)``) when you need the type to be visible.
        return str(self.value)


def to_dim(value: int | str | Dim) -> Dim:
    """Coerce ``int`` / ``str`` to ``Dim``; pass ``Dim`` through unchanged."""
    return value if isinstance(value, Dim) else Dim(value)


def as_static(d: object) -> int:
    """Coerce a shape element (``Dim``, ``int``, or ``str``) to a static
    int. Use at boundaries where the upstream type isn't yet uniformly
    ``Dim`` — e.g. ``infer_output_shape`` returns that mix ints/strings
    from constructor args with ``Dim`` from another Tensor's shape.

    Raises if ``d`` is symbolic (``Dim('seq_len')`` or ``str``)."""
    if isinstance(d, Dim):
        return d.as_static()
    if isinstance(d, int):
        return d
    raise TypeError(f"as_static: cannot convert {d!r} to int (symbolic shape element?)")


__all__ = ["Dim", "to_dim", "as_static"]
