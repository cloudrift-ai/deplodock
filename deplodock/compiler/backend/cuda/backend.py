"""CUDA backend: ``Graph`` → lowered ``Graph[CudaOp]`` → nvcc → GPU.

The compiled artifact is the lowered ``Graph`` itself — every compute
node carries a rendered CUDA kernel source plus its launch geometry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deplodock import config
from deplodock.compiler.backend import Backend, BenchmarkResult, RunResult
from deplodock.compiler.backend.cuda.program import (
    benchmark_program,
    benchmark_program_isolated,
    run_program,
    run_program_debug,
)
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

logger = logging.getLogger(__name__)

# Hard wall-clock cap for ``benchmark()`` calls. Wraps the whole bench
# in a daemon worker thread; if it doesn't return within this budget we
# abandon and raise ``RuntimeError``. Needed because NVRTC compilation
# inside the first kernel launch can take 30+ s on heavily-replicated
# kernels (e.g. autotune variants with FM*FN=256 cells), which would
# otherwise stall the whole sweep on one bad variant. Exposed via the
# inherited ``Backend.bench_wall_timeout_s`` attribute for callers
# (autotune cache reads it to set fail-row latencies).


if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline.dump import CompilerDump


def _resolve_tune_db(value: Path | str | None) -> Path | None:
    """Resolve the ``tune_db=`` constructor argument.

    - ``None`` → ``None`` (no DB; test-isolation default).
    - ``"auto"`` → ``DEPLODOCK_TUNE_DB`` env → ``~/.cache/deplodock/autotune.db``
      (shared resolution in :func:`deplodock.config.tune_db_path`). The result
      is returned regardless of whether the file exists; ``compile()`` skips
      opening it when missing.
    - Explicit ``Path`` / ``str`` → that path (no env lookup).
    """
    if value is None:
        return None
    if value == "auto":
        return config.tune_db_path()
    return Path(value)


class CudaBackend(Backend):
    """CUDA backend.

    When ``debug`` is True (or the ``DEPLODOCK_DEBUG`` env var is set),
    ``run()`` uses the per-launch debug path that dumps every non-input
    buffer after each kernel launch. ``last_debug_result`` is then
    populated with the per-launch snapshots for the last ``run()`` call.
    """

    name = "cuda"

    def __init__(
        self,
        *,
        debug: bool | None = None,
        dump: CompilerDump | None = None,
        bench_wall_timeout_s: float | None = None,
        bench_compile_timeout_s: float = 30.0,
        bench_run_timeout_s: float = 10.0,
        tune_db: Path | str | None = None,
    ) -> None:
        if debug is None:
            debug = config.debug_enabled()
        if dump is None:
            from deplodock.compiler.pipeline.dump import CompilerDump as _CD

            dump = _CD.from_env()
        self.debug = debug
        self.dump = dump
        self.last_debug_result = None
        # When set, ``benchmark()`` runs in a subprocess so the parent
        # can SIGKILL a wedged worker. Defaults to ``None`` (in-process,
        # required when ``on_iter`` callbacks are supplied).
        self.bench_wall_timeout_s = bench_wall_timeout_s
        # Per-stage bench budgets. Default 10 s suits the whole-graph
        # compile/run commands; the ``tune`` command overrides them *down*
        # (it benches isolated single kernels, where a fast-fail on a slow
        # variant matters more than headroom) — see ``commands/tune.py``.
        self.bench_compile_timeout_s = bench_compile_timeout_s
        self.bench_run_timeout_s = bench_run_timeout_s
        # Persistent autotune cache. ``None`` → no DB (test-isolation
        # default; tests construct ``CudaBackend()`` without args and
        # expect deterministic rule-defaults compiles). ``"auto"`` →
        # resolve ``DEPLODOCK_TUNE_DB`` env var → ``~/.cache/deplodock/autotune.db``,
        # open if the file exists. Explicit ``Path`` → use that file
        # (open if it exists; silently skip otherwise).
        self.tune_db = _resolve_tune_db(tune_db)

    def compile(self, graph: Graph) -> Graph:
        """Lower ``Graph`` → ``Graph[LoopOp]`` → ``Graph[TileOp]`` → ``Graph[CudaOp]``."""
        db = None
        if self.tune_db is not None and self.tune_db.exists():
            from deplodock.compiler.pipeline.search.db import SearchDB

            db = SearchDB(path=self.tune_db)
        return Pipeline.build(CUDA_PASSES).run(graph, db=db, dump=self.dump)

    def run(
        self,
        compiled: Graph,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        pre_run=None,
    ) -> tuple[RunResult, object]:
        # ``run_program`` / ``run_program_debug`` hold the GPU lock end
        # to end (compile + alloc + ``pre_run`` + launches + ``.get()``)
        # so peer xdist workers / parallel ``deplodock run`` invocations
        # can never interleave a kernel launch with our work on the
        # shared device. The ``pre_run`` callback runs inside that lock
        # so a torch eager reference computed for comparison sees the
        # same GPU state our kernels do.
        if self.debug:
            debug_result, pre_result = run_program_debug(compiled, input_data=input_data, pre_run=pre_run)
            self.last_debug_result = debug_result
            result_outputs = debug_result.outputs
            time_ms = None
        else:
            self.last_debug_result = None
            result, pre_result = run_program(compiled, input_data=input_data, pre_run=pre_run)
            result_outputs = result.outputs
            time_ms = result.time_ms
        # Symbolic output shapes bind from the supplied input array shapes:
        # walk each input dim's expr.free_vars() and record where each name
        # came from. Output dims (possibly composite Dim exprs) then resolve
        # via ``expr.eval(sym_env)`` — one path covers Literal / Var /
        # BinaryExpr uniformly.
        from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

        sym_env: dict[str, int] = {}
        if input_data is not None:
            for nid in compiled.inputs:
                if nid not in input_data:
                    continue
                for d, dim in enumerate(compiled.nodes[nid].output.shape):
                    if isinstance(dim.expr, Var):
                        sym_env.setdefault(dim.expr.name, int(input_data[nid].shape[d]))
        outputs: dict[str, np.ndarray] = {}
        for name, vals in result_outputs.items():
            shape = tuple(int(d.expr.eval(sym_env)) for d in compiled.nodes[name].output.shape)
            outputs[name] = np.asarray(vals, dtype=compiled.nodes[name].output.dtype.np).reshape(shape)
        return RunResult(outputs=outputs, time_ms=time_ms), pre_result

    def benchmark(
        self,
        compiled: Graph,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int | str = 20,
        on_iter=None,
    ) -> BenchmarkResult:
        del input_data
        # ``bench_wall_timeout_s`` selects between two paths:
        # - Set (autotune sweep): run in a subprocess-isolated worker so
        #   a wedged kernel can be SIGKILLed without leaving the
        #   parent's CUDA stream dirty. Wall-clock backstop on top of the
        #   in-worker ``bench_compile_timeout_s`` / ``bench_run_timeout_s``.
        # - ``None`` (interactive ``deplodock run --bench``): run in-process.
        #   Required when ``on_iter`` callbacks are supplied — they can't
        #   cross the subprocess boundary.
        # ``on_iter`` forces the in-process path even when
        # ``bench_wall_timeout_s`` is set — interleaved benches need to
        # share torch state with the parent and can't cross the
        # subprocess boundary. The autotune sweep never passes
        # ``on_iter``, so this fallback only fires for ``--tune --bench``
        # where the post-tune interleaved bench reuses the same backend.
        if self.bench_wall_timeout_s is not None and on_iter is None:
            result = benchmark_program_isolated(
                compiled,
                wall_timeout_s=self.bench_wall_timeout_s,
                warmup=warmup,
                num_iters=num_iters,
                compile_timeout_s=self.bench_compile_timeout_s,
                run_timeout_s=self.bench_run_timeout_s,
            )
        else:
            result = benchmark_program(
                compiled,
                warmup=warmup,
                num_iters=num_iters,
                on_iter=on_iter,
                compile_timeout_s=self.bench_compile_timeout_s,
                run_timeout_s=self.bench_run_timeout_s,
            )
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
            per_launch=result.per_launch,
        )
