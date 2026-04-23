"""Lower TransposeOp(x, axes) → IndexMapOp.

``TransposeOp.axes`` is either a length-2 swap (matching
``aten.transpose(dim0, dim1)``) or a full permutation of length ``ndim``
(matching ``aten.permute``).
"""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend_ir import TransposeOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor_ir import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", TransposeOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = tuple(root.output.shape)
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

    # ``perm[i]`` is the input dim that becomes output dim i. Invert so we can
    # express each input dim as a function of output-coordinate placeholders.
    inv = [0] * ndim
    for i, p in enumerate(perm):
        inv[p] = i
    coord_map = tuple(placeholder(inv[j]) for j in range(ndim))

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=coord_map),)),
        inputs=[x_id],
        output=Tensor(root.output.name, out_shape, root.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
