"""Fold ``TransposeOp(ConstantOp)`` into a single ``ConstantOp`` with a
runtime transpose marker. The original parameter still backs the constant
at bind time; the binder applies the recorded permutation before
flattening and downstream Loads see the post-transpose layout.

Why: a ``TransposeOp`` lowered later via ``010_transpose`` becomes an
``IndexMapOp``, which gets fused into consumer Loads' index expressions.
The runtime tensor stays in its original layout and the access pattern
just reads the transposed-coords element of the original storage. That's
correct but defeats the smem layout cuBLAS-style SGEMM kernels rely on:
``007_stage_inputs`` orients smem to match the *source* dim order, so
when source is ``[N, K]`` the smem is ``[N, K]`` with K innermost. For
the cuBLAS-beating SIMT FP32 path we want ``[K, N]`` with N innermost so
per-thread N-tile reads vectorize to LDS.128.

Pre-transposing the constant solves this: after fold, the runtime tensor
is physically ``[K, N]``, the consumer index reads ``[K, N]``, and 007's
source-preserving staging puts N innermost in smem.

Pattern: ``TransposeOp`` whose only input is a parameter ``ConstantOp``
(no scalar value, just a name pointing to a module attribute). Skips
scalar constants (where folding adds no value) and chained transposes
(handled by composing the existing ``transpose`` field, but currently
left to a second pass run since the engine reapplies rules to fixed
point).
"""

import os

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.frontend.ir import TransposeOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", TransposeOp)]


def rewrite(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    # Gated on the TMA path. The default cp.async ``+1`` padding path
    # doesn't need this transform — the consumer Load index already
    # matches the smem layout via ``010_transpose``'s IndexMap fusion.
    # Folding under the default path changes shapes in ways other
    # passes (RoPE, SDPA, register-tile) don't expect; folding under
    # the TMA path is exactly what enables LDS.128 vectorization on
    # the asymmetric ``(BN, BM)`` tile shape (see ``tuning.py``).
    if os.environ.get("DEPLODOCK_TMA") != "1":
        raise RuleSkipped("DEPLODOCK_TMA != 1 — fold disabled")
    if not isinstance(inp_x.op, ConstantOp):
        raise RuleSkipped("transpose input is not a ConstantOp")
    if inp_x.op.value is not None:
        raise RuleSkipped("scalar constants don't benefit from layout transpose")

    in_shape = tuple(inp_x.output.shape)
    ndim = len(in_shape)
    axes = root.op.axes

    # Resolve the transpose op's axes into a full permutation.
    if len(axes) == 2:
        a = axes[0] if axes[0] >= 0 else ndim + axes[0]
        b = axes[1] if axes[1] >= 0 else ndim + axes[1]
        if not (0 <= a < ndim and 0 <= b < ndim):
            raise RuleSkipped(f"transpose axes {axes} out of range for ndim={ndim}")
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
    elif len(axes) == ndim:
        perm = [(a if a >= 0 else ndim + a) for a in axes]
        if sorted(perm) != list(range(ndim)):
            raise RuleSkipped(f"transpose perm {perm} is not a valid permutation of [0,{ndim})")
    else:
        raise RuleSkipped(f"transpose axes length {len(axes)} is neither 2 nor ndim={ndim}")

    # Compose with any existing transpose on the source constant. The
    # post-fold permutation maps post-source coords to pre-source coords:
    # combined[i] = existing[perm[i]] when an inner transpose already
    # exists; perm itself otherwise. The source ConstantOp's stored
    # ``transpose`` is what the binder applies to the parameter, so the
    # composition is just standard permutation chaining.
    existing = inp_x.op.transpose
    combined: tuple[int, ...]
    if existing is None:
        combined = tuple(perm)
    else:
        combined = tuple(existing[i] for i in perm)

    # Replace the TransposeOp with a fresh ConstantOp carrying the
    # composed permutation. Reuses the source name so ``const_targets``
    # lookup at bind time still resolves to the original parameter.
    frag = open_fragment(graph, [])
    new_op = ConstantOp(name=inp_x.op.name, transpose=combined)
    new_id = frag.add_node(op=new_op, inputs=[], output=Tensor(out.name, out.shape, out.dtype))
    frag.outputs = [new_id]
    return frag
