"""Conftest for ``tests/compiler/``.

Defines the ``run_graph`` parametrized fixture that runs an accuracy test
through each backend (numpy / loop / cuda). A test that takes ``run_graph``
automatically executes three times under different param IDs — any
disagreement between backends makes bug attribution mechanical.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


def _skip_if_no_cuda() -> None:
    from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

    if not (has_nvcc() and has_cuda_gpu()):
        pytest.skip("CUDA not available (need nvcc + GPU)")


@pytest.fixture(params=["numpy", "loop", "cuda"])
def run_graph(request) -> Callable:
    """Return a callable ``run(graph, input_data) -> dict[name, ndarray]``.

    Each parametrized variant routes through a different backend; the
    callable hides the compile/run split. ``input_data`` values are
    numpy arrays with declared shapes; outputs are ndarrays reshaped to
    match the graph's declared output shapes.
    """
    kind = request.param

    if kind == "cuda":
        _skip_if_no_cuda()

    def _run(graph, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if kind == "numpy":
            from deplodock.compiler.backend.numpy import NumpyBackend

            be = NumpyBackend()
            compiled = be.compile(graph)
            return be.run_arrays(compiled, input_data=input_data)
        if kind == "loop":
            from deplodock.compiler.backend.loop import LoopBackend

            be = LoopBackend()
            compiled = be.compile(graph)
            # Auto-supply ConstantOp scalar values (matches NumpyBackend convention).
            augmented = _inject_constants(dict(input_data), compiled.constant_values)
            return be.run_arrays(compiled, input_data=augmented)
        # cuda
        from deplodock.compiler.backend.cuda.backend import CudaBackend

        be = CudaBackend()
        compiled = be.compile(graph)
        augmented = _inject_constants(dict(input_data), compiled.loop.constant_values)
        return be.run_arrays(compiled, input_data=augmented)

    return _run


def _inject_constants(input_data: dict[str, np.ndarray], constant_values: dict[str, float]) -> dict[str, np.ndarray]:
    """Auto-supply scalar ConstantOp values that weren't passed in input_data."""
    for nid, val in constant_values.items():
        if nid not in input_data:
            input_data[nid] = np.array([val], dtype=np.float32)
    return input_data
