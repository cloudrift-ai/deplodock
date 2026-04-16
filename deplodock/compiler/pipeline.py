"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of ``KernelOp`` ready for backend codegen.
The implementation lives in ``deplodock.compiler.lower`` (added in a
follow-up commit); this module is the stable public entry point.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph


def compile_graph(graph: Graph) -> list:
    """Lower a traced ``Graph`` to a list of ``KernelOp``.

    Returns the per-kernel structural IR consumed by backends. Each
    ``KernelOp`` carries its own input-assembly tree, optional contraction,
    optional reduce-stage chain, and optional epilogue.
    """
    raise NotImplementedError("compile_graph: structural lowering lands in feature/structural-compiler c3")
