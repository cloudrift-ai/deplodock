"""Conftest for ``tests/compiler/``.

Defines the ``run_graph`` parametrized fixture that runs an accuracy test
through each backend (numpy / loop / cuda). A test that takes ``run_graph``
automatically executes three times under different param IDs — any
disagreement between backends makes bug attribution mechanical.

Also exposes the ``has_cuda_gpu()`` predicate and the ``requires_cuda``
skip marker shared by all CUDA-gated tests in this package.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


def has_cuda_gpu() -> bool:
    """Check if cupy is importable and sees at least one CUDA device."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not has_cuda_gpu(),
    reason="CUDA not available (need cupy + GPU)",
)


@pytest.fixture(params=["f32", "f16"], ids=["f32", "f16"])
def dtype(request):
    """Parametrize a test over the float dtypes deplodock supports.

    Yields a :class:`deplodock.compiler.dtype.DataType`. Tests that take
    this fixture run once per dtype; combined with ``run_graph``, the
    full matrix is (backend × dtype) minus the loop/fp16 cell (skipped).
    """
    from deplodock.compiler import dtype as _dt  # noqa: PLC0415

    return _dt.get(request.param)


def dtype_tol(dtype) -> dict[str, float]:
    """Default ``np.testing.assert_allclose`` kwargs for a given dtype.

    fp16 needs ~3 orders of magnitude looser tolerance than fp32 to
    survive elementwise / reduction chains. Tests that need tighter
    bounds (or different ones per op) override locally.
    """
    return {"rtol": 5e-3, "atol": 5e-3} if dtype.name == "f16" else {"rtol": 1e-4, "atol": 1e-5}


def dtype_input_scale(dtype) -> float:
    """Magnitude scale for random inputs so fp16 reductions don't overflow.

    Standard-normal inputs feeding a sum / matmul over K elements have
    output magnitude ~ sqrt(K). fp16's representable range tops out at
    65504 — scaling by 0.05 keeps K up to ~1M safe.
    """
    return 0.05 if dtype.name == "f16" else 1.0


def _skip_if_no_cuda() -> None:
    if not has_cuda_gpu():
        pytest.skip("CUDA not available (need cupy + GPU)")


@pytest.fixture(params=["numpy", "loop", "cuda"])
def run_graph(request) -> Callable:
    """Return a callable ``run(graph, input_data) -> dict[name, ndarray]``.

    Each parametrized variant routes through a different backend; the
    callable hides the compile/run split. ``input_data`` values are
    numpy arrays with declared shapes; outputs are ndarrays reshaped to
    match the graph's declared output shapes.

    When tests also take the ``dtype`` fixture (parametrized over
    ``[F32, F16]``), this fixture will skip the ``loop`` backend for the
    fp16 row — the cppyy-driven loop runner is hardcoded to ``float`` in
    its generated kernels and has no fp16 path today.
    """
    kind = request.param

    if kind == "cuda":
        _skip_if_no_cuda()
    if kind == "loop":
        # If the test also takes a ``dtype`` fixture and it's fp16, skip
        # the loop backend (cppyy runner is f32-only — see
        # ``ir/loop/runner.py``). The fixture is opt-in, so tests that
        # don't request ``dtype`` aren't affected.
        dtype_node = request.node.callspec.params.get("dtype") if hasattr(request.node, "callspec") else None
        if dtype_node is not None and getattr(dtype_node, "name", None) == "f16":
            pytest.skip("loop backend (cppyy) has no fp16 path; covered by numpy + cuda")

    def _run(graph, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if kind == "numpy":
            from deplodock.compiler.backend.numpy import NumpyBackend

            be = NumpyBackend()
            return be.run(be.compile(graph), input_data=input_data)[0].outputs
        if kind == "loop":
            from deplodock.compiler.backend.loop import LoopBackend

            be = LoopBackend()
            compiled = be.compile(graph)
            augmented = _inject_constants(dict(input_data), compiled)
            return be.run(compiled, input_data=augmented)[0].outputs
        # cuda
        from deplodock.compiler.backend.cuda.backend import CudaBackend

        be = CudaBackend()
        compiled = be.compile(graph)
        augmented = _inject_constants(dict(input_data), compiled)
        return be.run(compiled, input_data=augmented)[0].outputs

    return _run


def _inject_constants(input_data: dict[str, np.ndarray], graph) -> dict[str, np.ndarray]:
    """Auto-supply scalar ConstantOp values that weren't passed in input_data.

    Also re-binds tensor constants whose ``ConstantOp.name`` matches an
    entry in the input data but whose graph node id differs (e.g. after
    ``050_fold_into_constant`` replaces ``TransposeOp(c)`` with a
    new ConstantOp carrying a runtime transpose marker — same source
    name, fresh node id, transposed shape). Applies the recorded
    permutation here so the loop / numpy backends see the post-transpose
    data.
    """
    from deplodock.compiler.ir.base import ConstantOp

    for nid, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp):
            continue
        if nid in input_data:
            continue
        if node.op.value is not None:
            input_data[nid] = np.array([node.op.value], dtype=np.float32)
            continue
        src = input_data.get(node.op.name)
        if src is not None:
            from deplodock.compiler.loader.binder import apply_load_ops

            arr = np.asarray(src, dtype=np.float32)
            if node.op.load_ops and node.op.source_shape is not None:
                arr = arr.reshape(node.op.source_shape)
            input_data[nid] = apply_load_ops(arr, node.op.load_ops)
    return input_data
