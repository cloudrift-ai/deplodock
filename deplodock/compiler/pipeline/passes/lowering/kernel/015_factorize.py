"""Factorize — expand the high-level ``MmaContraction`` into the warp-tier fragment soup.

``010_materialize`` emits a warp/mma contraction as a single :class:`MmaContraction` node (the
op tree + schedule are gone by then). This pass does the **exact atom factorization**: it
replaces that node with the ``Tile`` of ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` /
``RegStore`` — the four-way GRID/WARP/REGISTER/ATOM split, operand staging, and the per-cell
projection epilogue (all in :func:`_warp_factor.factorize_mma`). It runs at ``015`` — right
after materialize, before ``030_stamp_types`` — so every later kernel pass + the cuda render
see ordinary expanded kernel-IR; the ``MmaContraction`` never survives this pass.

The materializer always emits the contraction as the lone stmt of the ``KernelOp`` body
(``Body((mma,))``), so the match is a single top-level node — no recursion."""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import MmaContraction
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._warp_factor import factorize_mma

PATTERN = [Pattern("root", KernelOp)]


def rewrite(root: Node) -> KernelOp | None:
    op: KernelOp = root.op
    body = op.body
    if not (len(body) == 1 and isinstance(body[0], MmaContraction)):
        raise RuleSkipped("no MmaContraction to factorize")  # every other kernel passes through
    return KernelOp(body=Body((factorize_mma(body[0]),)), name=op.name)
