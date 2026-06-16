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


def matmul_graph(m: int, k: int, n: int) -> Graph:  # noqa: F821
    """Plain `(m,k) @ (k,n) -> (m,n)` matmul graph for lowering / backend tests.

    Returned shape: two ``InputOp`` named ``a`` / ``b``, one ``MatmulOp``
    named ``o`` — the same trivial GEMM graph that the lowering-accuracy
    suites and several backend tests would otherwise rebuild inline.
    Callers needing the input shapes can derive them as
    ``{"a": (m, k), "b": (k, n)}``.
    """
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


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


def drain_tune(pipeline, graph, *, on=None, **kwargs):
    """Synchronously collect :meth:`Pipeline.tune_async` terminals for tests.

    ``Pipeline.tune`` (the sync generator) is gone — the autotuner is async-only —
    so tests drive ``tune_async`` through one ``asyncio.run``. Returns the list of
    terminal candidates. ``on(candidate)`` is invoked per terminal as it arrives;
    return truthy from it to stop early (the async mirror of a ``for ... break``).
    A real ``CudaBackend``'s device-pinned worker is bound to this loop, so it is
    awaited-closed before the loop ends (no orphaned subprocess transport)."""
    import asyncio

    async def _collect():
        out = []
        try:
            async for cand in pipeline.tune_async(graph, **kwargs):
                out.append(cand)
                if on is not None and on(cand):
                    break
        finally:
            backend = kwargs.get("backend")
            if backend is not None and hasattr(backend, "aclose_async_worker"):
                await backend.aclose_async_worker()
        return out

    return asyncio.run(_collect())


def run_inner_reward(
    fused_graph, *, ctx, db, backend=None, backends=None, patience, ucb_c=None, explore_eps=0.0, seed=0, progress=None, prior=None
):
    """Synchronously run the async per-op inner reward for tests.

    ``two_level.inner_reward`` (the sync ``asyncio.run`` bridge) is gone — the tuner
    is async-only — so tests drive ``_inner_reward_async`` through one ``asyncio.run``,
    building the device-pinned ``pool`` from ``backend`` / ``backends`` exactly as the
    production outer loop does. Mirrors the old signature so call sites only rename."""
    import asyncio

    from deplodock.compiler.pipeline import TuningSearch
    from deplodock.compiler.pipeline.search.two_level import _inner_reward_async

    if ucb_c is None:
        ucb_c = TuningSearch.DEFAULT_UCB_C
    pool = list(backends) if backends else [backend]
    return asyncio.run(
        _inner_reward_async(
            fused_graph,
            ctx=ctx,
            db=db,
            pool=pool,
            patience=patience,
            ucb_c=ucb_c,
            explore_eps=explore_eps,
            seed=seed,
            progress=progress,
            prior=prior,
        )
    )


def run_two_level(graph, **kwargs):
    """Synchronously run the async :func:`two_level.run_two_level_tune` for tests."""
    import asyncio

    from deplodock.compiler.pipeline.search.two_level import run_two_level_tune

    return asyncio.run(run_two_level_tune(graph, **kwargs))


@pytest.fixture(params=["static", "dynamic"])
def shape_mode(request) -> str:
    """Parametrize a test over a static vs dynamic (symbolic-M) shape. Any test
    that names this fixture runs once per mode — the static/dynamic parity
    automation. Pair with :func:`dyn_M` to flip a graph's leading/M axis:

        def test_x(shape_mode):
            g = build_graph(M=dyn_M(shape_mode, 256), ...)   # int 256 or Dim('seq_len')

    The dynamic mode compiles ONE symbolic kernel (``Dim('seq_len')``, runtime
    ``int seq_len`` arg) tiled at the 512 hint and run at the concrete M fed in
    the input arrays; the static mode bakes M. Use tile-divisor M for strict
    static-vs-dynamic parity (off-hint masked sizes live in
    ``test_matmul_mma_masked.py``)."""
    return request.param


def dyn_M(mode: str, M: int):
    """``Dim('seq_len')`` for ``mode == 'dynamic'``, else the int ``M``. The
    one-liner that turns a static graph builder into a static/dynamic one."""
    if mode == "dynamic":
        from deplodock.compiler.dim import Dim

        return Dim("seq_len")
    return M


def from_pretrained_or_skip(loader, *args, **kwargs):
    """Run a HuggingFace ``from_pretrained`` download, skipping the test on a Hub
    connection / rate-limit failure.

    CI runners regularly hit HTTP 429 (Too Many Requests) from huggingface.co —
    transformers retries then wraps it as ``OSError("We couldn't connect …")``.
    That's an environment issue, not a code regression, so the network-dependent
    e2e / trace tests skip rather than hard-fail. ``loader`` is e.g.
    ``AutoConfig.from_pretrained``; ``args[0]`` is the model id (for the message)."""
    try:
        return loader(*args, **kwargs)
    except OSError as exc:  # transformers wraps 429 / connection errors as OSError here
        model = args[0] if args else kwargs.get("pretrained_model_name_or_path", "?")
        pytest.skip(f"HuggingFace Hub unavailable for {model} (likely rate-limited): {exc}")
