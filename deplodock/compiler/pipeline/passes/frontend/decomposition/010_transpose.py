"""Lower TransposeOp(x, axes) → IndexMapOp.

``axes`` is either a length-2 swap (``aten.transpose(dim0, dim1)``) or a
full permutation of length ``ndim`` (``aten.permute``).
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend.ir import TransposeOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", TransposeOp)]


def rewrite(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    in_shape = tuple(inp_x.output.shape)
    out_shape = tuple(out.shape)
    ndim = len(in_shape)
    axes = root.op.axes

    if len(axes) == 2:
        a = axes[0] if axes[0] >= 0 else ndim + axes[0]
        b = axes[1] if axes[1] >= 0 else ndim + axes[1]
        if not (0 <= a < ndim and 0 <= b < ndim):
            return None
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
    elif len(axes) == ndim:
        perm = [(a if a >= 0 else ndim + a) for a in axes]
        if sorted(perm) != list(range(ndim)):
            return None
    else:
        return None

    inv = [0] * ndim
    for i, p in enumerate(perm):
        inv[p] = i
    coord_map = tuple(placeholder(inv[j]) for j in range(ndim))

    frag = open_fragment(graph, [inp_x])
    new_node = single_indexmap(frag, inp_x, out_shape=out_shape, coord_map=coord_map, name=out.name)
    frag.outputs = [new_node.id]
    return frag
