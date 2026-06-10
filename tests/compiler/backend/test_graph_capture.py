"""CUDA graph capture of the per-kernel bench (``capture_graphs``).

The per-kernel reproducer bench wraps the measured region in CUDA graphs — cupy stream capture for
deplodock's per-launch batch loop, ``torch.cuda.CUDAGraph`` for the torch closures — so the CUDA
event windows measure dense GPU work instead of per-launch dispatch gaps. These tests cover:

- the captured ``benchmark_program`` path returns the same result shape with sane timings,
- captured replays compute the same outputs as the plain launch loop,
- the all-or-nothing fallback: a capture failure on either side falls the *whole*
  ``bench_lowered_vs_torch`` invocation back to uncaptured timing (``captured=False``), never mixing
  semantics within one table.
"""

from deplodock.compiler.backend.cuda.program import GraphCaptureError, benchmark_program

from ..conftest import requires_cuda
from .test_program import _make_add_graph


@requires_cuda
def test_benchmark_program_capture_graphs_returns_timing():
    captured = benchmark_program(_make_add_graph(1024), warmup=2, num_iters=5, capture_graphs=True)
    assert captured.time_ms > 0
    assert captured.num_launches == 1
    assert captured.per_launch is not None and len(captured.per_launch) == 1


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
