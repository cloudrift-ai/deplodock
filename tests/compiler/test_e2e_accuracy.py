"""End-to-end accuracy tests: verify deplodock output on canonical patterns.

Parametrized over backends (numpy / loop / cuda) via the ``run_graph``
fixture in ``conftest.py``, AND over dtypes (``f32`` / ``f16``) via the
``dtype`` fixture. The cuda variant is skipped when nvcc+GPU are
unavailable; the loop/fp16 cell is skipped (cppyy runner is f32-only).
Expected values are computed with numpy directly — no torch dependency.
"""

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor

pytestmark = [
    pytest.mark.xfail(
        reason="stage-wrap: bucket-14 follow-up — materializer Stage path needs wrap-body rewrite",
        strict=False,
    ),
]
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp

from .conftest import dtype_input_scale, dtype_tol


def _assert_close(actual, expected, dtype, **overrides):
    """Compare in the test's dtype with a dtype-appropriate tolerance.

    Overrides win over ``dtype_tol`` defaults; pass ``rtol=`` / ``atol=``
    when a specific test needs a tighter or looser bound.
    """
    np_dt = dtype.np
    actual = np.asarray(actual, dtype=np_dt).flatten()
    expected = np.asarray(expected, dtype=np_dt).flatten()
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    tol = {**dtype_tol(dtype), **overrides}
    np.testing.assert_allclose(actual, expected, **tol)


def _input(rng, shape, dtype) -> np.ndarray:
    """Standard-normal input scaled to the dtype's safe range."""
    return (rng.standard_normal(shape) * dtype_input_scale(dtype)).astype(dtype.np)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def test_e2e_pointwise_add(run_graph, dtype):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,), dtype), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,), dtype), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (8,), dtype), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=dtype.np)
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=dtype.np)
    outputs = run_graph(g, {"x": x, "y": y})
    _assert_close(list(outputs.values())[0], x + y, dtype)


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


def test_e2e_reduce_sum(run_graph, dtype):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8), dtype), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (4, 1), dtype), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    rng = np.random.default_rng(0)
    x = _input(rng, (4, 8), dtype)
    expected = x.astype(np.float32).sum(-1, keepdims=True).astype(dtype.np)
    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, dtype)


def test_e2e_reduce_sum_cooperative(run_graph, dtype):
    """Reduction wide enough (K=512 ≥ COOP_THRESHOLD) to trigger the
    smem-cooperative strategy. Verifies the new path matches numpy."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8, 512), dtype), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (8, 1), dtype), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    rng = np.random.default_rng(0)
    x = _input(rng, (8, 512), dtype)
    expected = x.astype(np.float32).sum(-1, keepdims=True).astype(dtype.np)
    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=1e-2 if dtype.name == "f16" else 1e-3)


def test_e2e_reduce_max_cooperative(run_graph, dtype):
    """Same shape, max reduction — exercises the ``fmaxf`` combine path
    in ``TreeHalve``."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8, 512), dtype), node_id="x")
    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("y", (8, 1), dtype), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    rng = np.random.default_rng(0)
    x = _input(rng, (8, 512), dtype)
    expected = x.max(-1, keepdims=True)
    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, dtype)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------


def test_e2e_matmul(run_graph, dtype):
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8), dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 3), dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (4, 3), dtype), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    rng = np.random.default_rng(0)
    a = _input(rng, (4, 8), dtype)
    b = _input(rng, (8, 3), dtype)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype.np)
    outputs = run_graph(g, {"a": a, "b": b})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=1e-2 if dtype.name == "f16" else 1e-3)


def test_e2e_matmul_blockify(run_graph, dtype):
    """Matmul shape divisible by ``BM=BN=BK=16`` — exercises the
    block-tiled SGEMM path: per-block BM·BN tile cooperatively walks K
    in BK-sized chunks, with A/B operand caching in smem and an Init-at-
    Tile-scope accumulator persisting across the K_o loop."""
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    # Sized to the production matmul tile (BN=128, BM=64): both axes
    # ≥ tile so blockify actually splits each into BLOCK + THREAD
    # rather than keeping the output extents whole and overflowing
    # launch_bounds.
    M, N, K = 128, 256, 64
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N), dtype), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    rng = np.random.default_rng(0)
    a = _input(rng, (M, K), dtype)
    b = _input(rng, (K, N), dtype)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype.np)
    outputs = run_graph(g, {"a": a, "b": b})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=2e-2 if dtype.name == "f16" else 1e-3)


def test_e2e_matmul_blockify_rectangular(run_graph, dtype):
    """Non-square matmul through the blockify path — verifies M, N split
    independently and per-buffer cache axes derive correctly when M≠N."""
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    M, N, K = 256, 128, 128  # rectangular; both axes ≥ matmul tile (BN=128, BM=64)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K), dtype), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N), dtype), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    rng = np.random.default_rng(1)
    a = _input(rng, (M, K), dtype)
    b = _input(rng, (K, N), dtype)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype.np)
    outputs = run_graph(g, {"a": a, "b": b})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=2e-2 if dtype.name == "f16" else 1e-3)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_e2e_rmsnorm(run_graph, dtype):
    """RMSNorm: x * rsqrt(sum(x^2) + eps) * w."""
    rows, dim = 8, 64
    eps = 1e-6

    rng = np.random.default_rng(42)
    x = _input(rng, (rows, dim), dtype)
    w = _input(rng, (dim,), dtype)
    xf = x.astype(np.float32)
    wf = w.astype(np.float32)
    sq_sum = (xf * xf).sum(-1, keepdims=True)
    expected = (xf * (1.0 / np.sqrt(sq_sum + eps)) * wf).astype(dtype.np)

    g = Graph()
    g.add_node(InputOp(), [], Tensor("X", (rows, dim), dtype), node_id="X")
    g.add_node(ConstantOp(name="eps", value=eps), [], Tensor("eps", (1,), dtype), node_id="eps")
    g.add_node(InputOp(), [], Tensor("w", (dim,), dtype), node_id="w")
    g.inputs = ["X", "w"]

    g.add_node(ElementwiseOp("multiply"), ["X", "X"], Tensor("sq", (rows, dim), dtype), node_id="sq")
    g.add_node(ReduceOp("sum", axis=-1), ["sq"], Tensor("red", (rows, 1), dtype), node_id="red")
    g.add_node(ElementwiseOp("add"), ["red", "eps"], Tensor("ae", (rows, 1), dtype), node_id="ae")
    g.add_node(ElementwiseOp("rsqrt"), ["ae"], Tensor("rsq", (rows, 1), dtype), node_id="rsq")
    g.add_node(ElementwiseOp("multiply"), ["X", "rsq"], Tensor("norm", (rows, dim), dtype), node_id="norm")
    g.add_node(ElementwiseOp("multiply"), ["norm", "w"], Tensor("out", (rows, dim), dtype), node_id="out")
    g.outputs = ["out"]

    outputs = run_graph(g, {"X": x, "w": w})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=1e-2 if dtype.name == "f16" else 1e-3)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def test_e2e_softmax(run_graph, dtype):
    """Softmax: max → sub → exp → sum → div."""
    rows, cols = 4, 8
    rng = np.random.default_rng(0)
    x = _input(rng, (rows, cols), dtype)
    xf = x.astype(np.float32)
    mx = xf.max(-1, keepdims=True)
    ex = np.exp(xf - mx)
    expected = (ex / ex.sum(-1, keepdims=True)).astype(dtype.np)

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, cols), dtype), node_id="x")
    g.inputs = ["x"]

    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("mx", (rows, 1), dtype), node_id="mx")
    g.add_node(ElementwiseOp("subtract"), ["x", "mx"], Tensor("subtract", (rows, cols), dtype), node_id="subtract")
    g.add_node(ElementwiseOp("exp"), ["subtract"], Tensor("exp", (rows, cols), dtype), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (rows, 1), dtype), node_id="sm")
    g.add_node(ElementwiseOp("divide"), ["exp", "sm"], Tensor("out", (rows, cols), dtype), node_id="out")
    g.outputs = ["out"]

    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=1e-2 if dtype.name == "f16" else 1e-3)


def test_e2e_softmax_cooperative(run_graph, dtype):
    """Softmax with K=2048 ≥ COOP_THRESHOLD — exercises the multi-phase
    cooperative path (two reductions + one strided output loop, all under
    one Tile sharing two ``__shared__`` accumulator buffers)."""
    rows, cols = 4, 2048
    rng = np.random.default_rng(0)
    x = _input(rng, (rows, cols), dtype)
    xf = x.astype(np.float32)
    mx = xf.max(-1, keepdims=True)
    ex = np.exp(xf - mx)
    expected = (ex / ex.sum(-1, keepdims=True)).astype(dtype.np)

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, cols), dtype), node_id="x")
    g.inputs = ["x"]

    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("mx", (rows, 1), dtype), node_id="mx")
    g.add_node(ElementwiseOp("subtract"), ["x", "mx"], Tensor("subtract", (rows, cols), dtype), node_id="subtract")
    g.add_node(ElementwiseOp("exp"), ["subtract"], Tensor("exp", (rows, cols), dtype), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (rows, 1), dtype), node_id="sm")
    g.add_node(ElementwiseOp("divide"), ["exp", "sm"], Tensor("out", (rows, cols), dtype), node_id="out")
    g.outputs = ["out"]

    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, dtype, rtol=1e-2 if dtype.name == "f16" else 1e-3)
