"""CUDA graph capture of the per-kernel bench (``capture_graphs``).

The per-kernel reproducer bench wraps the measured region in CUDA graphs — cupy stream capture for
deplodock's per-launch batch loop, ``torch.cuda.CUDAGraph`` for the torch closures — so the CUDA
event windows measure dense GPU work instead of per-launch dispatch gaps. These tests cover:

- the captured ``benchmark_program`` path returns the same result shape with sane timings,
- captured replays compute the same outputs as the plain launch loop,
- the all-or-nothing fallback: a capture failure on either side falls the *whole*
  ``bench_lowered_vs_torch`` invocation back to uncaptured timing (``captured=False``), never mixing
  semantics within one table,
- the whole-program (e2e) timing windows: automatic for multi-launch programs under capture, ``None``
  for single-launch programs (the solo window is the program time) and when capture is off or the
  program-graph capture fails — which is never fatal.
"""

from deplodock.compiler.backend.cuda.program import GraphCaptureError, benchmark_program
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp

from ..conftest import requires_cuda
from .test_program import EW_ADD_SOURCE, _make_add_graph


def _make_two_launch_graph(n: int = 8) -> Graph:
    """Two chained elementwise adds (C = A + B; D = C + B) — the smallest
    program where the whole-program window covers more than one launch."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (n,)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (n,)), node_id="B")
    op = CudaOp(
        kernel_source=EW_ADD_SOURCE,
        kernel_name="ew_add",
        arg_order=("A", "B", "C"),
        grid=((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
    )
    g.add_node(op=op, inputs=["A", "B"], output=Tensor("C", (n,)), node_id="C")
    op2 = CudaOp(
        kernel_source=EW_ADD_SOURCE,
        kernel_name="ew_add",
        arg_order=("C", "B", "D"),
        grid=((n + 255) // 256, 1, 1),
        block=(256, 1, 1),
    )
    g.add_node(op=op2, inputs=["C", "B"], output=Tensor("D", (n,)), node_id="D")
    g.inputs = ["A", "B"]
    g.outputs = ["D"]
    return g


@requires_cuda
def test_benchmark_program_e2e_automatic_for_multi_launch():
    result = benchmark_program(_make_two_launch_graph(), warmup=2, num_iters=5, capture_graphs=True)
    assert result.captured is True
    assert result.e2e_ms is not None and result.e2e_ms > 0
    assert result.e2e_min_ms is not None and 0 < result.e2e_min_ms <= result.e2e_ms
    # Per-launch fields are unaffected by the extra windows.
    assert result.num_launches == 2
    assert result.per_launch is not None and len(result.per_launch) == 2


@requires_cuda
def test_benchmark_program_e2e_skipped_for_single_launch():
    # One launch: the solo per-launch window already IS the program time — no second measurement.
    result = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5, capture_graphs=True)
    assert result.e2e_ms is None and result.e2e_min_ms is None


@requires_cuda
def test_benchmark_program_e2e_none_without_capture():
    result = benchmark_program(_make_two_launch_graph(), warmup=2, num_iters=5, capture_graphs=False)
    assert result.e2e_ms is None and result.e2e_min_ms is None


@requires_cuda
def test_benchmark_program_e2e_capture_failure_is_nonfatal(monkeypatch):
    from deplodock.compiler.backend.cuda.program import CompiledProgram

    def _boom(self):
        raise GraphCaptureError("forced whole-program capture failure")

    monkeypatch.setattr(CompiledProgram, "capture_program_graph", _boom)
    result = benchmark_program(_make_two_launch_graph(), warmup=2, num_iters=5, capture_graphs=True)
    assert result.captured is True  # per-launch capture is independent of the e2e graph
    assert result.time_ms > 0
    assert result.e2e_ms is None and result.e2e_min_ms is None


@requires_cuda
def test_benchmark_program_capture_graphs_returns_timing():
    captured = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5, capture_graphs=True)
    assert captured.time_ms > 0
    assert captured.num_launches == 1
    assert captured.per_launch is not None and len(captured.per_launch) == 1
    assert captured.captured is True


@requires_cuda
def test_benchmark_program_capture_failure_falls_back_in_place(monkeypatch):
    """A capture failure inside ``benchmark_program`` must not raise (the autotune
    sweep would mark the variant bench_fail) — it continues uncaptured and reports
    it via ``result.captured``."""
    from deplodock.compiler.backend.cuda.program import CompiledProgram

    def _boom(self, batch_sizes):
        raise GraphCaptureError("forced capture failure")

    monkeypatch.setattr(CompiledProgram, "capture_launch_graphs", _boom)
    result = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5, capture_graphs=True)
    assert result.time_ms > 0
    assert result.captured is False


@requires_cuda
def test_benchmark_program_capture_graphs_warmup_zero():
    # warmup=0 skips calibration entirely; capture must still happen (all-1 batches).
    result = benchmark_program(_make_add_graph(1024), warmup=0, num_iters=3, capture_graphs=True)
    assert result.time_ms > 0


@requires_cuda
def test_captured_replay_matches_plain_launch_outputs():
    """A captured graph replay must compute exactly what the Python launch loop computes.
    ``_allocate`` fills inputs deterministically, so two fresh programs see identical data."""
    import numpy as np

    from deplodock.compiler.backend.cuda.program import CompiledProgram, run_program
    from deplodock.compiler.backend.gpu_lock import gpu_lock

    reference, _ = run_program(_make_add_graph(8))

    with gpu_lock():
        prog = CompiledProgram.build(_make_add_graph(8))
        prog.capture_launch_graphs([2])
        prog.iter_once(batch_sizes=[2])
        out = prog.outputs()["C"]
    np.testing.assert_allclose(out, reference.outputs["C"])


def _lowered_rmsnorm():
    """Frontend snapshot + lowered graph for a small torch_ref-runnable op (the pattern the
    per-kernel sweep uses; see ``test_bench_worker_compare``)."""
    from deplodock.commands.run import _detect_stage, _passes_after_stage
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.pipeline import Pipeline

    g, _, _ = graph_from_code("torch.nn.RMSNorm(512)(torch.randn(8, 512))")
    fe = g.copy()
    tail = _passes_after_stage(_detect_stage(g))
    lowered = Pipeline.build(tail).run(g) if tail else g
    return fe, lowered


@requires_cuda
def test_bench_lowered_vs_torch_captures():
    from deplodock.commands.run import bench_lowered_vs_torch
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    fe, lowered = _lowered_rmsnorm()
    results, bench, torch_available, captured = bench_lowered_vs_torch(
        fe, lowered, CudaBackend(), seed=0, do_bench=True, warmup=2, iters=5, bench_backends="eager,deplodock"
    )
    assert torch_available
    assert captured is True
    assert results.get("Eager PyTorch", 0) > 0
    assert results.get("Deplodock", 0) > 0
    assert bench is not None


@requires_cuda
def test_deplodock_capture_failure_falls_back_uncaptured(monkeypatch):
    from deplodock.commands.run import bench_lowered_vs_torch
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram

    def _boom(self, batch_sizes):
        raise GraphCaptureError("forced capture failure")

    monkeypatch.setattr(CompiledProgram, "capture_launch_graphs", _boom)
    fe, lowered = _lowered_rmsnorm()
    results, _, torch_available, captured = bench_lowered_vs_torch(
        fe, lowered, CudaBackend(), seed=0, do_bench=True, warmup=2, iters=5, bench_backends="eager,deplodock"
    )
    assert torch_available
    assert captured is False, "deplodock-side capture failure must fall the whole bench back to uncaptured"
    assert results.get("Eager PyTorch", 0) > 0
    assert results.get("Deplodock", 0) > 0


@requires_cuda
def test_torch_capture_failure_disables_deplodock_capture(monkeypatch):
    import deplodock.commands.run as run_mod
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram

    def _boom(fn):
        raise RuntimeError("forced torch capture failure")

    deplodock_captures: list = []
    orig = CompiledProgram.capture_launch_graphs
    monkeypatch.setattr(run_mod, "_capture_torch_fn", _boom)
    monkeypatch.setattr(CompiledProgram, "capture_launch_graphs", lambda self, bs: deplodock_captures.append(bs) or orig(self, bs))

    fe, lowered = _lowered_rmsnorm()
    results, _, torch_available, captured = run_mod.bench_lowered_vs_torch(
        fe, lowered, CudaBackend(), seed=0, do_bench=True, warmup=2, iters=5, bench_backends="eager,deplodock"
    )
    assert torch_available
    assert captured is False
    assert not deplodock_captures, "torch-side capture failure must also disable deplodock-side capture"
    assert results.get("Eager PyTorch", 0) > 0
    assert results.get("Deplodock", 0) > 0
