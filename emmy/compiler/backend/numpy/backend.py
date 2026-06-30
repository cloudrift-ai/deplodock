"""Numpy backend: interpret a Graph IR using numpy arrays.

No GPU required — useful for correctness testing and debugging.
``compile`` is a no-op (the graph is its own compiled artifact);
``run`` is the default ``Backend.run`` topo-walk interpreter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emmy.compiler.backend import Backend

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph


class NumpyBackend(Backend):
    """Graph → in-memory interpreter → numpy arrays.

    Inherits the default ``run`` and wall-time ``benchmark`` from ``Backend``.
    """

    def compile(self, graph: Graph) -> Graph:
        return graph
