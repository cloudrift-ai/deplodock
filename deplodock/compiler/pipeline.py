"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of structural ``KernelOp``s ready for
backend codegen. The lowering itself lives in ``deplodock.compiler.lower``;
this module is the stable public entry point.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.lower import lower
from deplodock.compiler.ops import KernelOp


def compile_graph(graph: Graph) -> list[KernelOp]:
    """Lower a traced ``Graph`` to a list of ``KernelOp``.

    Each ``KernelOp`` describes one GPU kernel's worth of work: its input
    assembly tree(s), optional contraction, optional reduce-stage chain,
    and optional epilogue. Adjacent kernels communicate through external
    buffers named by ``Port.buffer_id`` (which correspond to graph node
    ids).
    """
    return lower(graph)
