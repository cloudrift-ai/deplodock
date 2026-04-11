"""Compiler pipeline entry points."""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.rewriter import Rewriter
from deplodock.compiler.trace import PassTrace


def compile_graph(graph: Graph, rewriter: Rewriter) -> tuple[Graph, list[PassTrace]]:
    """Run rewrite passes only, return optimized graph + traces.

    This is the primary entry point for graph-level compilation.
    The result can be passed to plan_graph() → Backend for execution.
    """
    pass_traces: list[PassTrace] = []
    graph = rewriter.apply(graph, pass_traces=pass_traces)
    return graph, pass_traces
