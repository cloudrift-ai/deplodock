"""End-to-end fp16 elementwise chain on the CUDA backend.

Verifies that an fp16-typed graph compiles through NVRTC with the
``__half`` parameter type + ``cuda_fp16.h`` include + load/store
conversion intrinsics, runs on the GPU, and returns fp16 numerics that
match eager numpy within fp16 tolerance.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler import dtype as dt
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline

from .conftest import requires_cuda


def _fp16_chain_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1024,), dt.F16), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (1024,), dt.F16), node_id="e")
    g.add_node(op=ElementwiseOp("negative"), inputs=["e"], output=Tensor("y", (1024,), dt.F16), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


@requires_cuda
def test_fp16_elementwise_chain_cuda():
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    graph = _fp16_chain_graph()
    compiled = CudaBackend().compile(Pipeline.build(LOOP_PASSES).run(graph))

    # Verify the rendered CUDA source picked up fp16 signature + include
    # for at least one kernel (there will be one after fusion).
    from deplodock.compiler.ir.cuda import CudaOp

    cuda_nodes = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_nodes, "expected at least one CudaOp after lowering"
    sources = "\n".join(n.op.kernel_source for n in cuda_nodes)
    assert "__half" in sources, f"expected __half in kernel sources, got:\n{sources}"
    assert "cuda_fp16.h" in sources, f"expected cuda_fp16.h include, got:\n{sources}"
    assert "__half2float" in sources or "__float2half" in sources, sources

    rng = np.random.default_rng(0)
    x_data = (rng.standard_normal(1024) * 0.5).astype(np.float16)

    result = CudaBackend().run(compiled, input_data={"x": x_data})
    out = next(iter(result.outputs.values()))

    assert out.dtype == np.float16, f"expected float16 output, got {out.dtype}"
    expected = (-np.exp(x_data.astype(np.float32))).astype(np.float16)
    np.testing.assert_allclose(out.reshape(-1), expected, rtol=5e-3, atol=5e-3)
