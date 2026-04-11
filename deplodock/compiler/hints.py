"""Advisory hints for influencing compiler decisions.

Hints are metadata attached to Graph nodes or the Graph itself.  They do
not affect computation semantics — backends may ignore unknown hints.
Keys use dotted namespaces (e.g. ``cuda.matmul.strategy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deplodock.compiler.ir import Graph


@dataclass
class Hints:
    """Advisory metadata bag keyed by dotted namespace strings."""

    _data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a hint value."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Return whether *key* is present."""
        return key in self._data

    def remove(self, key: str) -> None:
        """Remove *key* if present."""
        self._data.pop(key, None)

    def merge(self, other: Hints) -> None:
        """Merge *other* into self.  Other's values win on conflict."""
        self._data.update(other._data)

    def prefix(self, ns: str) -> dict[str, Any]:
        """Return all hints under *ns* as a flat dict with the prefix stripped.

        >>> h = Hints(); h.set("cuda.matmul.strategy", "naive")
        >>> h.prefix("cuda.matmul")
        {'strategy': 'naive'}
        """
        dot = ns if ns.endswith(".") else ns + "."
        return {k[len(dot) :]: v for k, v in self._data.items() if k.startswith(dot)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dict(self._data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Hints:
        """Deserialize from a dict."""
        return Hints(_data=dict(data))

    def __bool__(self) -> bool:
        return bool(self._data)

    def __repr__(self) -> str:
        return f"Hints({self._data!r})"


def resolve_hints(graph: Graph, node_id: str) -> Hints:
    """Merge graph-level hints with node-level hints (node wins)."""
    merged = Hints()
    merged.merge(graph.hints)
    merged.merge(graph.nodes[node_id].hints)
    return merged
