"""Tests for the compile-and-run pipeline."""

import json
from pathlib import Path

import pytest

from deplodock.compiler.cuda.runner import has_cuda_gpu, has_nvcc
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.pipeline import compile_and_run
from deplodock.compiler.rewriter import Pass, Rewriter, Rule

# ---- helpers ----


def _make_matmul_graph(m, k, n):
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (m, k)), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(op=ElementwiseOp(fn="mul"), inputs=[a, b], output=Tensor("AB", (m, k, n)), node_id="ew")
    red = g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=[ew], output=Tensor("C", (m, n)), node_id="red")
    g.outputs = [red]
    return g


def _python_matmul(a, b, m, k, n):
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i * k + kk] * b[kk * n + j]
            c[i * n + j] = s
    return c


def _load_rewriter():
    rule_path = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion" / "001_fuse_reduce_elementwise.py"
    rule = Rule.from_file(rule_path)
    return Rewriter(passes=[Pass(name="fusion", rules=[rule])])


# ---- tests ----

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


@requires_cuda
def test_pipeline_end_to_end():
    """Full pipeline produces a valid trace with execution results."""
    M, K, N = 4, 3, 2
    g = _make_matmul_graph(M, K, N)
    rewriter = _load_rewriter()

    a_data = [float(i + 1) for i in range(M * K)]
    b_data = [float(i + 1) for i in range(K * N)]
    expected = _python_matmul(a_data, b_data, M, K, N)

    trace = compile_and_run(
        graph=g,
        rewriter=rewriter,
        inputs={"A": a_data, "B": b_data},
        output_name="C",
        output_size=M * N,
        dim_args={"M": M, "N": N, "K": K},
        expected=expected,
    )

    # Trace should have no error.
    assert trace.error is None

    # Check the trace structure.
    assert trace.input_graph is not None
    assert len(trace.passes) == 1
    assert trace.passes[0].name == "fusion"
    assert len(trace.passes[0].rules_applied) == 1
    assert trace.cuda_kernel is not None
    assert "__global__" in trace.cuda_kernel

    # Check execution results.
    assert trace.execution is not None
    assert trace.execution.correct is True
    assert trace.execution.max_error < 1e-4
    assert trace.execution.kernel_time_ms is not None
    assert trace.execution.kernel_time_ms >= 0
    assert trace.execution.dimensions == {"M": M, "N": N, "K": K}

    # Ensure the full trace is valid JSON.
    j = trace.to_json()
    parsed = json.loads(j)
    assert parsed["execution"]["correct"] is True
    assert "cuda_kernel" in parsed
