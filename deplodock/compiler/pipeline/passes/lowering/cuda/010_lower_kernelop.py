"""Lower each ``KernelOp`` node to a ``CudaOp``.

Tile-flavored KernelOp→CudaOp lowering (launch-geometry derivation from the
body's tiles, TMA descriptor collection, render) was demolished alongside the
tile IR; this pass is a stub pending the tile-lowering rebuild.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.pipeline import Match, Pattern

PATTERN = [Pattern("root", KernelOp)]


def rewrite(match: Match, root: Node) -> Graph | None:  # noqa: ARG001 — signature required by rule dispatch
    raise NotImplementedError("tile lowering demolished — pending rebuild")
