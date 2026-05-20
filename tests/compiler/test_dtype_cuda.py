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
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
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
    # Native fp16 chain — no boundary conversions on the data path.
    # ``hexp`` (fp16 exp) + ``__float2half(0.0f)`` for the negation literal,
    # native ``operator-`` on __half. No ``__half2float`` anywhere.
    assert "hexp" in sources, f"expected native hexp, got:\n{sources}"
    assert "__half2float" not in sources, f"native fp16 chain should not promote to float, got:\n{sources}"

    rng = np.random.default_rng(0)
    x_data = (rng.standard_normal(1024) * 0.5).astype(np.float16)

    result = CudaBackend().run(compiled, input_data={"x": x_data})
    out = next(iter(result.outputs.values()))

    assert out.dtype == np.float16, f"expected float16 output, got {out.dtype}"
    expected = (-np.exp(x_data.astype(np.float32))).astype(np.float16)
    np.testing.assert_allclose(out.reshape(-1), expected, rtol=5e-3, atol=5e-3)


def _fp16_erf_graph() -> Graph:
    """fp16 graph that exercises the f32 fallback path: ``erf`` has no native
    fp16 form, so the kernel must promote inputs to float, run ``erff``, and
    demote back to ``__half`` for the store."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1024,), dt.F16), node_id="x")
    g.add_node(op=ElementwiseOp("erf"), inputs=["x"], output=Tensor("y", (1024,), dt.F16), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


@requires_cuda
def test_fp16_fallback_to_float_for_non_native_op():
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp

    compiled = CudaBackend().compile(Pipeline.build(LOOP_PASSES).run(_fp16_erf_graph()))
    sources = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    # Signature stays __half; the fallback inserts __half2float on the
    # input and __float2half on the store, with f32 erff in between.
    assert "__half* y" in sources or "__half* x" in sources, sources
    assert "erff" in sources, f"expected fallback to f32 erff, got:\n{sources}"
    assert "__half2float" in sources, f"expected promote-to-float at use, got:\n{sources}"
    assert "__float2half" in sources, f"expected demote-to-half at store, got:\n{sources}"

    rng = np.random.default_rng(1)
    x_data = (rng.standard_normal(1024) * 0.5).astype(np.float16)
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    out = next(iter(result.outputs.values()))
    assert out.dtype == np.float16
    import math  # noqa: PLC0415

    expected = np.array([math.erf(float(v)) for v in x_data.astype(np.float32)], dtype=np.float16)
    np.testing.assert_allclose(out.reshape(-1), expected, rtol=5e-3, atol=5e-3)


def _fp16_sum_graph() -> Graph:
    """fp16 input + sum reduction. The Init-placement pass picks F32 for
    the accumulator (any fp16 input promotes), so the rendered kernel
    declares ``float acc`` and combines via ``acc += __half2float(value)``."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1024,), dt.F16), node_id="x")
    # Output dtype is fp16 to match HF reduction conventions (sum
    # stays in input dtype downstream; the accumulator is the only
    # f32 step).
    g.add_node(op=ReduceOp(op="sum", axis=0), inputs=["x"], output=Tensor("s", (1,), dt.F16), node_id="s")
    g.inputs = ["x"]
    g.outputs = ["s"]
    return g


@requires_cuda
def test_fp16_reduction_uses_fp32_accumulator_on_cuda():
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp

    compiled = CudaBackend().compile(Pipeline.build(LOOP_PASSES).run(_fp16_sum_graph()))
    sources = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    # Accumulator declared as float (f32 promotion); value converted at
    # the combine; final store back to __half.
    assert "float acc" in sources or "float v" in sources or "float r" in sources, f"expected f32 accumulator local, got:\n{sources}"
    assert "__half2float" in sources, f"expected __half2float at combine, got:\n{sources}"
    assert "__float2half" in sources, f"expected __float2half at store, got:\n{sources}"

    rng = np.random.default_rng(2)
    x_data = (rng.standard_normal(1024) * 0.1).astype(np.float16)
    result = CudaBackend().run(compiled, input_data={"x": x_data})
    out = next(iter(result.outputs.values()))
    assert out.dtype == np.float16
    expected = np.array([x_data.astype(np.float32).sum()], dtype=np.float16)
    np.testing.assert_allclose(out.reshape(-1), expected, rtol=1e-2, atol=1e-2)
