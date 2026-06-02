"""Lower the tensor-core matmul cell to the kernel-IR MMA fragment chain.

The matmul cell arrives in tensor-core form: ``tile/011_lower_atom_cell``
tagged the operand ``Load``s with ``atom`` / ``role`` and fused the compute
into an :class:`~deplodock.compiler.ir.stmt.Mma`, and the staging passes
carried both through (the loads staged like any other, the ``Mma`` keeping its
reduce loop ``is_reduce``). The partition planner emits one of four cell
shapes:

A. Direct K-loop, no staging (pruned for mma.sync — ldmatrix is smem-only):
   ``AtomTile > SerialTile(K_o) > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
B. Single-bundle staged (SYNC or single-buffer ASYNC):
   ``AtomTile > SerialTile(K_o) > StageBundle > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
C. Filtered K (single-iter, no SerialTile): ``AtomTile > [Load a*, Load b*, Mma]`` (inline)
D. Pipelined + buffered (cp.async double-buffered): prologue StageBundle, K_o-1
   loop with issue-next StageBundle + AsyncWait + reduce, AsyncWait, epilogue reduce, Write.

This pass walks the ``TileOp`` body, finds each ``AtomTile``, and rewrites its
cell into ``RegFragment`` decls + per-reduce ``LdmatrixLoad a + LdmatrixLoad b
+ MmaSyncPtx`` + final ``RegStore``, then strips the ``AtomTile`` wrapper. The
fragment SSA names are seeded once from the FIRST reduce site (stable across
prologue/inner/epilogue, which is what the per-cell replicator in
``010_split_register_axes`` expects). Operands are matched per reduce site by
the ``Load.role`` tag; the slab ``src_index`` / ``ldm`` come from re-harvesting
the live ``Source`` (``_mma_src_index``), so the phase-prefix prepend for
double-buffered slabs stays correct.

Each cell's :class:`~deplodock.compiler.ir.tile.ir.Atom` spec (shape + operand
dtypes) is read straight off its ``Mma`` — no ``ATOM_KIND`` knob lookup. The
lowering helpers live in the sibling ``_``-prefixed ``_mma`` module, which the
pass loader skips; this file is the pattern + ``rewrite`` entry point.
Eligibility: an ``AtomTile`` in the body (scalar TileOps have none → skip).
Idempotence: after this pass the ``AtomTile`` is gone, so a second visit finds
nothing and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._mma import lower_atom_cells

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    # No knob lookup: each cell's atom spec is read off its ``Mma`` (the
    # ``AtomTile`` presence is the signal). Scalar / already-lowered TileOps
    # have no ``AtomTile`` → found is False → skip.
    lowered, found = lower_atom_cells(op.body, smem_sources={})
    if not found:
        raise RuleSkipped("no AtomTile in body (scalar, or already lowered)")
    return TileOp(body=lowered, name=op.name, knobs=op.knobs)
