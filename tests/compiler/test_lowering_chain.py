"""Verify the lowering-chain backref ``CudaOp -> KernelOp -> TileOp -> LoopOp``.

Each lowering rule (``tileify``, ``materialize_tile``, ``lower_kernelop``)
stamps its predecessor op on the produced op's ``source`` field. From a
fully lowered CudaOp the originating LoopOp is reachable via
``cuda.source.source.source``. The chain is attribution metadata —
``Graph.structural_key`` and ``op_cache_key`` must ignore it so equivalent
kernels still dedup.
"""

from __future__ import annotations

from deplodock.compiler.cache import op_cache_key
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import CUDA_PASSES, run_pipeline


def _elementwise_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), inputs=[], output=Tensor(name="x", shape=(8,), dtype="float32"), node_id="x")
    g.add_node(InputOp(), inputs=[], output=Tensor(name="z", shape=(8,), dtype="float32"), node_id="z")
    g.add_node(ElementwiseOp(op="add"), inputs=["x", "z"], output=Tensor(name="y", shape=(8,), dtype="float32"), node_id="y")
    g.inputs = ["x", "z"]
    g.outputs = ["y"]
    return g


def test_lowering_chain_reaches_loop_op() -> None:
    g = run_pipeline(_elementwise_graph(), CUDA_PASSES)
    cuda_nodes = [n for n in g.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_nodes, "pipeline should produce at least one CudaOp"
    for n in cuda_nodes:
        cuda = n.op
        assert isinstance(cuda.source, KernelOp), f"CudaOp.source must be KernelOp, got {type(cuda.source).__name__}"
        assert isinstance(cuda.source.source, TileOp), "KernelOp.source must be TileOp"
        assert isinstance(cuda.source.source.source, LoopOp), "TileOp.source must be LoopOp"


def test_source_excluded_from_structural_key() -> None:
    """Two CudaOps that differ only in the chain they came from (i.e. only
    in ``.source``) should hash equal — the cache must dedup them."""
    g = run_pipeline(_elementwise_graph(), CUDA_PASSES)
    cuda = next(n.op for n in g.nodes.values() if isinstance(n.op, CudaOp))
    other = CudaOp(
        kernel_source=cuda.kernel_source,
        kernel_name=cuda.kernel_name,
        arg_order=cuda.arg_order,
        grid=cuda.grid,
        block=cuda.block,
        smem_bytes=cuda.smem_bytes,
        tma_descriptors=cuda.tma_descriptors,
        zero_outputs=cuda.zero_outputs,
        source=None,  # different lowering chain
    )
    assert op_cache_key(cuda) == op_cache_key(other)
