"""Shared helper for the ``004a`` / ``004b`` constant-fold rules.

Both rules absorb a single layout op (``TransposeOp`` / ``ReshapeOp``)
whose only input is a parameter ``ConstantOp`` into the constant's
``load_ops`` chain. The rewrite is uniform; the rule files differ only
in their ``PATTERN`` op type, so the body lives here.

Gated on sm_90+ (the TMA path). Folding under the default cp.async path
changes shapes in ways other passes (RoPE, SDPA, register-tile) don't
expect — the consumer Load index already matches the smem layout via
``010_transpose`` / ``011_reshape``'s IndexMap fusion. Folding under
the TMA path is what enables LDS.128 vectorization on the asymmetric
``(BN, BM)`` tile shape (see ``tuning.py``). The gate is a behavior
preservation choice; the ``load_ops`` field itself is general-purpose
and is honored by the loader regardless of how a chain got recorded.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment
from deplodock.compiler.target import compute_capability


def fold_into_constant(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    """Append ``root.op`` to ``inp_x.op.load_ops`` and rebuild the constant.

    Skips scalar constants (``value is not None``) — the loader never
    visits them. Skips activations — only parameter/buffer constants
    get a load_ops chain.
    """
    if compute_capability() < (9, 0):
        raise RuleSkipped("TMA path inactive (compute capability < sm_90) — fold not needed")
    if not isinstance(inp_x.op, ConstantOp):
        raise RuleSkipped("input is not a ConstantOp")
    if inp_x.op.value is not None:
        raise RuleSkipped("scalar constants are not folded into")

    new_load_ops = inp_x.op.load_ops + (root.op,)
    new_op = ConstantOp(
        name=inp_x.op.name,
        load_ops=new_load_ops,
        source_path=inp_x.op.source_path,
        source_shape=inp_x.op.source_shape,
        source_dtype=inp_x.op.source_dtype,
    )
    frag = open_fragment(graph, [])
    new_id = frag.add_node(op=new_op, inputs=[], output=Tensor(out.name, out.shape, out.dtype))
    frag.outputs = [new_id]
    return frag
