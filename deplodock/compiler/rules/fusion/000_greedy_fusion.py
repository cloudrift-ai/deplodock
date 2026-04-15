"""Seed the fusion pass by running the greedy edge-merger on the whole graph.

Wraps the existing ``auto_fuse`` algorithm as a Rewriter rule so fusion
composes with decomposition and optimization passes. Fires once (idempotently
— once the graph has KernelOps, a second call is a no-op) and returns the
same graph object when nothing changed to terminate the fixed-point loop.

Subsequent rules in this directory (``001_structure_contraction``,
``002_structure_reduce``, ...) restructure the resulting flat-prologue
KernelOps into ``ContractionCore`` / ``ReduceCore`` form.

Pattern: ``_`` matches every node; rewrite fires at most once per pass
because subsequent invocations produce graphs with identical node counts.
"""

from __future__ import annotations

from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match

PATTERN = "_"


def rewrite(graph: Graph, match: Match) -> Graph:
    new_graph = auto_fuse(graph)
    if len(new_graph.nodes) == len(graph.nodes):
        return graph
    return new_graph
