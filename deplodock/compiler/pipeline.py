"""End-to-end compile-and-run pipeline with structured trace output."""

from __future__ import annotations

import logging

from deplodock.compiler.cuda.codegen import emit_kernel
from deplodock.compiler.cuda.lower import MatmulConfig, lower_graph
from deplodock.compiler.cuda.runner import run_kernel
from deplodock.compiler.ir import Graph
from deplodock.compiler.rewriter import Rewriter
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace

logger = logging.getLogger(__name__)


def compile_graph(graph: Graph, rewriter: Rewriter) -> tuple[Graph, list[PassTrace]]:
    """Run rewrite passes only, return optimized graph + traces.

    This is the primary entry point for graph-level compilation without
    CUDA lowering — useful for inspecting fusion results.
    """
    pass_traces: list[PassTrace] = []
    graph = rewriter.apply(graph, pass_traces=pass_traces)
    return graph, pass_traces


def compile_and_run(
    graph: Graph,
    rewriter: Rewriter,
    inputs: dict[str, list[float]],
    output_name: str,
    output_size: int,
    dim_args: dict[str, int],
    expected: list[float] | None = None,
    tolerance: float = 1e-4,
    matmul_config: MatmulConfig | None = None,
) -> CompilerTrace:
    """Run the full pipeline: rewrite → lower → codegen → execute → trace.

    Args:
        graph: Input compute graph.
        rewriter: Configured rewriter with passes.
        inputs: Mapping of param name → flat float data.
        output_name: Name of the output kernel parameter.
        output_size: Number of output elements.
        dim_args: Mapping of dimension param name → int value.
        expected: Optional expected output for correctness checking.
        tolerance: Max absolute error for correctness.

    Returns:
        CompilerTrace with full structured log of every stage.
    """
    trace = CompilerTrace()
    trace.input_graph = graph.to_dict()

    # --- Rewrite passes ---
    try:
        graph = rewriter.apply(graph, pass_traces=trace.passes)
    except Exception as e:
        trace.error = f"Rewrite failed: {e}"
        return trace

    # --- Lower to CUDA IR ---
    try:
        kernel = lower_graph(graph, config=matmul_config)
    except Exception as e:
        trace.error = f"Lowering failed: {e}"
        return trace

    # --- Codegen ---
    try:
        source = emit_kernel(kernel)
        trace.cuda_kernel = source
    except Exception as e:
        trace.error = f"Codegen failed: {e}"
        return trace

    # --- Execute ---
    try:
        result = run_kernel(
            kernel=kernel,
            kernel_source=source,
            inputs=inputs,
            output_name=output_name,
            output_size=output_size,
            dim_args=dim_args,
        )
    except Exception as e:
        trace.error = f"Execution failed: {e}"
        return trace

    # --- Build execution result ---
    exec_result = ExecutionResult(
        output=result.output,
        kernel_time_ms=result.kernel_time_ms,
        dimensions=dim_args,
    )

    if expected is not None:
        exec_result.expected = expected
        max_err = max(abs(a - b) for a, b in zip(result.output, expected, strict=True))
        exec_result.max_error = max_err
        exec_result.correct = max_err < tolerance

    trace.execution = exec_result
    return trace
