"""Compose adjacent single-source IndexMapOps into one.

When an IndexMapOp ``X`` reads a parent IndexMapOp ``Y`` (both single-source),
the two can be collapsed into one IndexMapOp ``Z`` that reads ``Y``'s source
directly. ``Z``'s ``out_shape`` equals ``X.out_shape``; its ``coord_map[i]``
is ``Y.coord_map[i]`` with ``Y``'s output placeholders substituted by the
matching entry in ``X.coord_map``.

Applies even when the parent has multiple consumers: each consumer gets
its own composed IndexMapOp. The parent stays in the graph while other
consumers still read it; if this composition was the parent's sole
consumer, ``_remove_orphans`` cleans it up after the rewrite. ``select``
predicates, if any, combine as a logical AND so multi-source cat chains
remain correct.

This optimization runs before fusion so rule ``001_assemble_kernels`` sees
a simpler IR: each external input has at most one IndexMapOp between it
and the compute region.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, BinOp, substitute
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("outer", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    op = node.op
    if not isinstance(op, IndexMapOp) or len(op.sources) != 1:
        return None

    outer_src = op.sources[0]
    parent_id = node.inputs[outer_src.input_idx]
    parent = graph.nodes.get(parent_id)
    if parent is None or not isinstance(parent.op, IndexMapOp) or len(parent.op.sources) != 1:
        return None

    parent_src = parent.op.sources[0]
    grandparent_id = parent.inputs[parent_src.input_idx]
    grandparent = graph.nodes.get(grandparent_id)
    if grandparent is None:
        return None

    # Substitute parent's output placeholders with the outer's coord_map entries.
    # The parent's output has len(outer.coord_map) dims; placeholder k ranges over those dims.
    subst = {f"{PLACEHOLDER_PREFIX}{k}": outer_src.coord_map[k] for k in range(len(outer_src.coord_map))}

    composed_cm = tuple(substitute(c, subst) for c in parent_src.coord_map)

    # If either link has a select predicate, propagate both (combined via &&).
    composed_select = None
    if parent_src.select is not None:
        composed_select = substitute(parent_src.select, subst)
    if outer_src.select is not None:
        composed_select = outer_src.select if composed_select is None else BinOp("&&", composed_select, outer_src.select)

    new_op = IndexMapOp(
        out_shape=op.out_shape,
        sources=(IndexSource(input_idx=0, coord_map=composed_cm, select=composed_select),),
    )

    frag = Graph()
    frag.add_node(
        InputOp(),
        [],
        Tensor(grandparent_id, grandparent.output.shape, grandparent.output.dtype),
        node_id=grandparent_id,
    )
    out_id = frag.add_node(new_op, [grandparent_id], Tensor(node.output.name, node.output.shape, node.output.dtype), node_id=nid)
    frag.outputs = [out_id]

    # Replace only the outer (``nid``). When this was the parent's sole
    # consumer, ``_remove_orphans`` cleans the parent up after the rewrite;
    # if the parent has other consumers, it stays so they still have an
    # IndexMapOp to read.
    match.consumed = {nid}
    return frag
