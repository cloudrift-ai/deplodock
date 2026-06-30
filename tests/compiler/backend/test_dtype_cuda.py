"""End-to-end fp16 elementwise chain on the CUDA backend.

Verifies that an fp16-typed graph compiles through NVRTC with the
``__half`` parameter type + ``cuda_fp16.h`` include + load/store
conversion intrinsics, runs on the GPU, and returns fp16 numerics that
match eager numpy within fp16 tolerance.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler import dtype as dt
from emmy.compiler.backend.cuda.dtype import canonical_from_cuda_name, cuda_name, nbytes_of
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SoftmaxOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline

from ..conftest import requires_cuda


def test_cuda_name_int_dtypes():
    # I32/I64 must map to the C type names the kernel renderer emits in
    # parameter signatures for graphs whose placeholder inputs carry an
    # integer dtype (input_ids / position_ids from HF whole-model traces).
    assert cuda_name(dt.I32) == "int"
    assert cuda_name(dt.I64) == "long long"
    # Aliases route through the same mapping.
    assert cuda_name("int32") == "int"
    assert cuda_name("int64") == "long long"
    assert cuda_name("long") == "long long"


def test_canonical_from_cuda_name_int_dtypes():
    # The inverse mapping is what ``Smem.render`` consults to recover the
    # canonical DataType from a kernel-internal C name.
    assert canonical_from_cuda_name("int") == "i32"
    assert canonical_from_cuda_name("long long") == "i64"


def test_nbytes_of_int_dtypes():
    # ``nbytes_of`` is the per-element size table used by the slab budget
    # / smem accounting. Must agree with the canonical DataType sizes.
    assert nbytes_of(dt.I32) == 4
    assert nbytes_of(dt.I64) == 8
    assert nbytes_of("i32") == 4
    assert nbytes_of("i64") == 8


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
    from emmy.compiler.backend.cuda.backend import CudaBackend

    graph = _fp16_chain_graph()
    compiled = CudaBackend().compile(Pipeline.build(LOOP_PASSES).run(graph))
    # Verify the rendered CUDA source picked up fp16 signature + include
    # for at least one kernel (there will be one after fusion).
    from emmy.compiler.ir.cuda import CudaOp

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

    result, _ = CudaBackend().run(compiled, input_data={"x": x_data})
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
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.cuda import CudaOp

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
    result, _ = CudaBackend().run(compiled, input_data={"x": x_data})
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
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.cuda import CudaOp

    compiled = CudaBackend().compile(Pipeline.build(LOOP_PASSES).run(_fp16_sum_graph()))
    sources = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    # Accumulator declared as float (f32 promotion); value converted at
    # the combine; final store back to __half.
    assert "float acc" in sources or "float v" in sources or "float r" in sources, f"expected f32 accumulator local, got:\n{sources}"
    assert "__half2float" in sources, f"expected __half2float at combine, got:\n{sources}"
    assert "__float2half" in sources, f"expected __float2half at store, got:\n{sources}"

    rng = np.random.default_rng(2)
    x_data = (rng.standard_normal(1024) * 0.1).astype(np.float16)
    result, _ = CudaBackend().run(compiled, input_data={"x": x_data})
    out = next(iter(result.outputs.values()))
    assert out.dtype == np.float16
    expected = np.array([x_data.astype(np.float32).sum()], dtype=np.float16)
    np.testing.assert_allclose(out.reshape(-1), expected, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Real-world reduction shapes — matmul / softmax / RMSNorm
# ---------------------------------------------------------------------------


@requires_cuda
def test_fp16_matmul_cuda():
    """fp16 matmul: k-reduction with __half loads and f32 accumulator.
    The decomposition expands to a Loop(k) > Loop(...) > Accum chain;
    the Init-placement pass picks F32 because both inputs are fp16."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    m, k, n = 8, 16, 8
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k), dt.F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n), dt.F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (m, n), dt.F16), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    rng = np.random.default_rng(3)
    a_data = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b_data = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)

    be = CudaBackend()
    result, _ = be.run(be.compile(g), input_data={"a": a_data, "b": b_data})
    out = next(iter(result.outputs.values())).reshape(m, n)
    assert out.dtype == np.float16

    expected = (a_data.astype(np.float32) @ b_data.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(out, expected, rtol=5e-3, atol=5e-3)


@requires_cuda
def test_fp16_softmax_cuda():
    """fp16 softmax along last dim: two reductions (max + sum) on f16
    values with f32 accumulators, then a per-element divide."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    rows, cols = 4, 64
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (rows, cols), dt.F16), node_id="x")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["x"], output=Tensor("y", (rows, cols), dt.F16), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    rng = np.random.default_rng(4)
    x_data = rng.standard_normal((rows, cols)).astype(np.float16)

    be = CudaBackend()
    result, _ = be.run(be.compile(g), input_data={"x": x_data})
    out = next(iter(result.outputs.values())).reshape(rows, cols)
    assert out.dtype == np.float16

    xf = x_data.astype(np.float32)
    m = xf.max(axis=-1, keepdims=True)
    e = np.exp(xf - m)
    expected = (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)
    np.testing.assert_allclose(out, expected, rtol=5e-3, atol=5e-3)


@requires_cuda
def test_fp16_rmsnorm_cuda():
    """fp16 RMSNorm: ``x * rsqrt(mean(x*x) + eps) * weight``. The
    sum-of-squares reduction needs the f32 accumulator (x^2 in fp16
    can overflow / underflow for typical activation magnitudes)."""
    from emmy.compiler.backend.cuda.backend import CudaBackend

    rows, cols = 4, 64
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (rows, cols), dt.F16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (cols,), dt.F16), node_id="w")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "w"], output=Tensor("y", (rows, cols), dt.F16), node_id="y")
    g.inputs = ["x", "w"]
    g.outputs = ["y"]

    rng = np.random.default_rng(5)
    x_data = rng.standard_normal((rows, cols)).astype(np.float16)
    w_data = rng.standard_normal((cols,)).astype(np.float16)

    be = CudaBackend()
    result, _ = be.run(be.compile(g), input_data={"x": x_data, "w": w_data})
    out = next(iter(result.outputs.values())).reshape(rows, cols)
    assert out.dtype == np.float16

    xf = x_data.astype(np.float32)
    rms = np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + 1e-6)
    expected = ((xf / rms) * w_data.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)


@requires_cuda
def test_fp16_max_reduction_stays_in_fp16():
    """``max`` is a selection (no magnitude accumulation), so it stays in
    the input dtype. The kernel should declare ``__half`` for the
    accumulator and use ``__hmax`` — not promote to float."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.cuda import CudaOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1024,), dt.F16), node_id="x")
    g.add_node(op=ReduceOp(op="maximum", axis=0), inputs=["x"], output=Tensor("m", (1,), dt.F16), node_id="m")
    g.inputs = ["x"]
    g.outputs = ["m"]

    be = CudaBackend()
    compiled = be.compile(g)
    sources = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    # Accumulator declared as __half — selection ops don't need precision boost.
    assert "__half acc" in sources, f"expected fp16 accumulator for max, got:\n{sources}"
    # The combine uses __hmax (native fp16 max), not fmaxf.
    assert "__hmax" in sources, f"expected __hmax intrinsic for fp16 max, got:\n{sources}"
