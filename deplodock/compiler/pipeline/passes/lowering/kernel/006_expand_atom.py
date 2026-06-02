"""Expand Tile-IR ``Atom`` nodes into the kernel-IR MMA fragment chain.

The **expansion** half of the Atom lowering split (the recognition half is
``005_lower_atom_tile``, which emits the ``Atom``). Runs right after 005 and
before ``010_split_register_axes`` in the kernel chain, so ``Atom`` lives in
the IR only across this single gap — every downstream pass keeps consuming the
same ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore`` chain
as before the split.

For each ``Atom`` in the TileOp body :func:`expand_atom` prepends the
``RegFragment`` decls, rewrites the (staged) K-reduce skeleton — replacing each
reduce ``SerialTile`` body with the ``ldmatrix a + ldmatrix b + mma.sync`` MAC
chain (shape C inlines a single chain), and appends the ``RegStore`` epilogue
from the ``Atom``'s hoisted store target. The per-operand slab addressing comes
from the ``Source`` snapshots carried on the node, so no enclosing-scope source
threading is needed here.

Eligibility: ``op.knobs["ATOM_KIND"]`` set (only warp-tier matmul rows carry an
``Atom``). Idempotence: after this pass the ``Atom`` is gone, so on a second
visit the walk finds nothing and the pass skips. See the ``_mma`` module and
the ``005_lower_atom_tile`` / ``Atom`` docstrings.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._mma import expand_atoms

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    if "ATOM_KIND" not in op.knobs:
        raise RuleSkipped("not an MMA TileOp (no ATOM_KIND knob)")
    expanded, found = expand_atoms(op.body, smem_sources={})
    if not found:
        # Second visit (Atom already expanded) or a non-MMA tower.
        raise RuleSkipped("no Atom in body — already expanded")
    return TileOp(body=expanded, name=op.name, knobs=op.knobs)
