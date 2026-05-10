"""Lower CatOp([a, b], dim) → IndexMapOp.

Tracer convention: CatOp.inputs = [tensor_a, tensor_b, dim_const]. Only the
2-tensor variant is supported here (covers Qwen rotary's
``cat(neg, slice_1, dim=-1)``); add a 3-tensor rule if a model needs it.

After decomposition: IndexMapOp.inputs = [tensor_a, tensor_b]; the dim is baked
into the source selects and the second source's coord_map offset.

**In-bounds clamping**: The cat-source Selects gate which value is *used*
at each output coordinate, but downstream lifting / fusion turns each
source into an unconditional ``Load``. With the naive coord_map A/B
would read out-of-range indices on the half where the other source is
selected (e.g. rotary's ``cat([-x[..., half:], x[..., :half]], -1)``
issues ``Load(x, dim - half)`` for ``dim < half`` → negative offset).
Most allocator layouts mask the OOB Load behind a same-page read, but
swap-bencher allocations land tight enough to segfault. Each source's
cat-dim coord is wrapped in a ``TernaryExpr`` that clamps to its valid
range when out of domain — the loaded value is wrong but the Select
chain never picks it.
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import Literal, TernaryExpr, placeholder
from deplodock.compiler.ir.frontend.ir import CatOp
from deplodock.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from deplodock.compiler.pipeline.engine import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", CatOp)]


def rewrite(match: Match, inp_a: Node, inp_b: Node, inp_dim: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    a_shape = tuple(inp_a.output.shape)
    out_shape = tuple(out.shape)
    ndim = len(out_shape)

    if not (isinstance(inp_dim.op, ConstantOp) and inp_dim.op.value is not None):
        raise RuleSkipped("cat dim must be a ConstantOp with a value")
    dim = int(inp_dim.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

    if not isinstance(a_shape[norm_dim], int):
        raise RuleSkipped(f"cat split point a_shape[{norm_dim}]={a_shape[norm_dim]!r} must be a static int")
    split = a_shape[norm_dim]

    frag = open_fragment(graph, [inp_a, inp_b])

    # In-domain predicate / clamps: source A is valid for dim < split, source B
    # for dim ≥ split. When the post-fusion Load fires on the off-domain side,
    # the ternary collapses the cat-dim coord into the source's valid range so
    # the read stays in-bounds (the Select downstream discards the value).
    cat_var = placeholder(norm_dim)
    in_a = cat_var.lt(Literal(split, "int"))
    a_clamped = TernaryExpr(cond=in_a, if_true=cat_var, if_false=Literal(0, "int"))
    b_clamped = TernaryExpr(cond=in_a, if_true=Literal(0, "int"), if_false=cat_var - Literal(split, "int"))

    # Source A: clamped coord_map for cat dim, identity elsewhere; select gates
    # *use* of the value (TernaryExpr in the consumer's index expression).
    coord_map_a = tuple(a_clamped if i == norm_dim else placeholder(i) for i in range(ndim))
    src_a = IndexSource(
        input_idx=0,
        coord_map=coord_map_a,
        select=in_a,
    )
    # Source B: clamped coord_map; default (else) branch — no select.
    coord_map_b = tuple(b_clamped if i == norm_dim else placeholder(i) for i in range(ndim))
    src_b = IndexSource(input_idx=1, coord_map=coord_map_b)

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(src_a, src_b)),
        inputs=[inp_a, inp_b],
        output=Tensor(out.name, out_shape, out.dtype),
    )

    frag.outputs = [new_id]
    return frag
