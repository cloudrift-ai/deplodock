"""End-to-end accuracy tests: verify deplodock output on canonical patterns.

Parametrized over backends (numpy / loop / cuda) via the ``run_graph``
fixture in ``conftest.py``; the cuda variant is skipped when nvcc+GPU
are unavailable. Expected values are computed with numpy directly — no
torch dependency.
"""

import numpy as np

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def _assert_close(actual, expected, *, rtol: float = 1e-4, atol: float = 1e-5):
    actual = np.asarray(actual, dtype=np.float32).flatten()
    expected = np.asarray(expected, dtype=np.float32).flatten()
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def test_e2e_pointwise_add(run_graph):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (8,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
    outputs = run_graph(g, {"x": x, "y": y})
    _assert_close(list(outputs.values())[0], x + y)


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


def test_e2e_reduce_sum(run_graph):
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (4, 1)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]

    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    expected = x.sum(-1, keepdims=True)
    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------


def test_e2e_matmul(run_graph):
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 3)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (4, 3)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 3)).astype(np.float32)
    expected = a @ b
    outputs = run_graph(g, {"a": a, "b": b})
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_e2e_rmsnorm(run_graph):
    """RMSNorm: x * rsqrt(sum(x^2) + eps) * w."""
    rows, dim = 8, 64
    eps = 1e-6

    rng = np.random.default_rng(42)
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    w = rng.standard_normal((dim,)).astype(np.float32)
    sq_sum = (x * x).sum(-1, keepdims=True)
    expected = x * (1.0 / np.sqrt(sq_sum + eps)) * w

    g = Graph()
    g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    g.add_node(ConstantOp(name="eps", value=eps), [], Tensor("eps", (1,)), node_id="eps")
    g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.inputs = ["X", "w"]

    g.add_node(ElementwiseOp("multiply"), ["X", "X"], Tensor("sq", (rows, dim)), node_id="sq")
    g.add_node(ReduceOp("sum", axis=-1), ["sq"], Tensor("red", (rows, 1)), node_id="red")
    g.add_node(ElementwiseOp("add"), ["red", "eps"], Tensor("ae", (rows, 1)), node_id="ae")
    g.add_node(ElementwiseOp("rsqrt"), ["ae"], Tensor("rsq", (rows, 1)), node_id="rsq")
    g.add_node(ElementwiseOp("multiply"), ["X", "rsq"], Tensor("norm", (rows, dim)), node_id="norm")
    g.add_node(ElementwiseOp("multiply"), ["norm", "w"], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = ["out"]

    outputs = run_graph(g, {"X": x, "w": w})
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


def test_e2e_softmax(run_graph):
    """Softmax: max → sub → exp → sum → div."""
    rows, cols = 4, 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    mx = x.max(-1, keepdims=True)
    ex = np.exp(x - mx)
    expected = ex / ex.sum(-1, keepdims=True)

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.inputs = ["x"]

    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("mx", (rows, 1)), node_id="mx")
    g.add_node(ElementwiseOp("subtract"), ["x", "mx"], Tensor("subtract", (rows, cols)), node_id="subtract")
    g.add_node(ElementwiseOp("exp"), ["subtract"], Tensor("exp", (rows, cols)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (rows, 1)), node_id="sm")
    g.add_node(ElementwiseOp("divide"), ["exp", "sm"], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = ["out"]

    outputs = run_graph(g, {"x": x})
    _assert_close(list(outputs.values())[0], expected, rtol=1e-3)
