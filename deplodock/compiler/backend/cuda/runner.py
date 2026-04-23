"""Single-kernel cupy dispatch.

Thin cupy-based wrappers used by tuning/diagnostics scripts. The graph
dispatch path lives in ``program.py``; this module exists for
kernel-at-a-time benchmarking (e.g. SGEMM tuning).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from deplodock.compiler.ir.kernel import GpuKernel

logger = logging.getLogger(__name__)


def has_nvcc() -> bool:
    """Check if nvcc is available on PATH (legacy — kernel dispatch uses NVRTC via cupy)."""
    return shutil.which("nvcc") is not None


def has_cuda_gpu() -> bool:
    """Check if cupy is importable and sees at least one CUDA device."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _detect_arch() -> str:
    """Return compute capability as ``sm_XY`` using cupy, falling back to sm_89."""
    try:
        import cupy as cp

        major, minor = cp.cuda.Device().compute_capability[0], cp.cuda.Device().compute_capability[1]
        return f"sm_{major}{minor}"
    except Exception:
        return "sm_89"


@dataclass
class KernelResult:
    """Result of a single-kernel execution."""

    output: list[float]
    kernel_time_ms: float | None = None


@dataclass
class MatmulBenchmarkResult:
    """Result of a SGEMM benchmark with cuBLAS comparison."""

    kernel_time_ms: float  # median
    kernel_min_ms: float
    kernel_max_ms: float
    cublas_time_ms: float | None = None
    cublas_min_ms: float | None = None
    cublas_max_ms: float | None = None
    sm_clock_mhz_pre: int | None = None
    sm_clock_mhz_post: int | None = None
    gpu_temp_c_pre: int | None = None
    gpu_temp_c_post: int | None = None
    gflops: float = 0.0
    cublas_gflops: float | None = None
    efficiency_pct: float | None = None
    dimensions: dict[str, int] | None = None


def _grid(kernel: GpuKernel, dim_args: dict[str, int]) -> tuple[int, int, int]:
    """Replicate the old host-program grid logic for matmul-style kernels."""
    bx, by, _bz = kernel.block_size
    if kernel.tile_m and kernel.tile_n and not kernel.grid_2d:
        n = dim_args.get("N", 1)
        m = dim_args.get("M", 1)
        ntx = (n + kernel.tile_n - 1) // kernel.tile_n
        nty = (m + kernel.tile_m - 1) // kernel.tile_m
        gx = ntx * (((nty + 7) // 8) * 8)
        return (gx, 1, 1)
    if kernel.tile_m and kernel.tile_n and kernel.grid_2d:
        n = dim_args.get("N", 1)
        m = dim_args.get("M", 1)
        return ((n + kernel.tile_n - 1) // kernel.tile_n, (m + kernel.tile_m - 1) // kernel.tile_m, 1)
    n = dim_args.get("N", 1)
    m = dim_args.get("M", 1)
    return ((n + bx - 1) // bx, (m + by - 1) // by, 1)


def _compile_kernel(kernel_source: str, kernel_name: str, smem_bytes: int = 0):
    """Compile kernel_source → cupy.RawKernel."""
    import cupy as cp

    rk = cp.RawKernel(kernel_source, kernel_name, options=("--use_fast_math",))
    if smem_bytes > 48 * 1024:
        # Kernels with >48 KiB of dynamic smem need explicit opt-in on the cupy kernel.
        rk.max_dynamic_shared_size_bytes = smem_bytes
    return rk


def run_kernel(
    kernel: GpuKernel,
    kernel_source: str,
    inputs: dict[str, list[float]],
    output_name: str,
    output_size: int,
    dim_args: dict[str, int],
) -> KernelResult:
    """Compile and run a CUDA kernel via cupy.RawKernel, return output + timing.

    ``kernel.params`` drives argument order: params whose name is in ``inputs``
    or equals ``output_name`` are allocated as cupy arrays; everything else is
    looked up in ``dim_args`` and passed as a scalar.
    """
    import cupy as cp

    rk = _compile_kernel(kernel_source, kernel.name, smem_bytes=getattr(kernel, "extra_smem_bytes", 0))

    arrays: dict[str, cp.ndarray] = {}
    for name, data in inputs.items():
        arrays[name] = cp.asarray(np.asarray(data, dtype=np.float32))
    arrays[output_name] = cp.zeros(output_size, dtype=cp.float32)

    launch_args: list = []
    for p in kernel.params:
        if p.name in arrays:
            launch_args.append(arrays[p.name])
        elif p.name in dim_args:
            dtype = np.int64 if "long" in p.dtype else np.int32
            launch_args.append(dtype(dim_args[p.name]))
        else:
            raise ValueError(f"No value for kernel param {p.name!r}")

    grid = _grid(kernel, dim_args)
    block = kernel.block_size

    # Warmup + timed launch (one iter each).
    rk(grid, block, tuple(launch_args))
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    rk(grid, block, tuple(launch_args))
    stop.record()
    stop.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, stop)

    output = arrays[output_name].get().astype(np.float32, copy=False).tolist()
    return KernelResult(output=output, kernel_time_ms=elapsed_ms)


def _sample_clock_and_temp() -> tuple[int | None, int | None]:
    """Best-effort read of current SM clock (MHz) and GPU temp (°C) via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode != 0:
            return None, None
        line = res.stdout.strip().splitlines()[0]
        clk_str, temp_str = (s.strip() for s in line.split(","))
        return int(clk_str), int(temp_str)
    except Exception:
        return None, None


def run_benchmark(
    kernel: GpuKernel,
    kernel_source: str,
    dim_args: dict[str, int],
    num_iterations: int = 10,
    warmup: int = 3,
    coarsen_cols: int = 1,
    coarsen_rows: int = 1,
    compare_cublas: bool = True,
    cublas_math_mode: str = "default",
) -> MatmulBenchmarkResult:
    """Benchmark a SGEMM kernel against ``cupy.matmul`` (cuBLAS backend).

    ``kernel.params`` is expected to list two ``const float*`` inputs (A, B),
    one ``float*`` output (C), and any number of scalar dim args (M, N, K, …)
    resolved from ``dim_args``. Data is random float32 with fixed seed for
    repeatability.
    """
    del coarsen_cols, coarsen_rows, cublas_math_mode  # reserved for future variants
    import cupy as cp

    m = dim_args.get("M", 1)
    n = dim_args.get("N", 1)
    k = dim_args.get("K", 1)

    rng = np.random.default_rng(0)
    A = cp.asarray(rng.standard_normal((m, k), dtype=np.float32))
    B = cp.asarray(rng.standard_normal((k, n), dtype=np.float32))
    C = cp.zeros((m, n), dtype=cp.float32)

    # Build launch args from kernel.params.
    buffers = {"A": A, "B": B, "C": C}
    launch_args: list = []
    for p in kernel.params:
        if p.name in buffers:
            launch_args.append(buffers[p.name])
        elif p.name in dim_args:
            dtype = np.int64 if "long" in p.dtype else np.int32
            launch_args.append(dtype(dim_args[p.name]))
        else:
            raise ValueError(f"No value for kernel param {p.name!r}")

    rk = _compile_kernel(kernel_source, kernel.name, smem_bytes=getattr(kernel, "extra_smem_bytes", 0))
    grid = _grid(kernel, dim_args)
    block = kernel.block_size

    clk_pre, temp_pre = _sample_clock_and_temp()

    for _ in range(warmup):
        rk(grid, block, tuple(launch_args))
    cp.cuda.runtime.deviceSynchronize()

    kernel_times: list[float] = []
    for _ in range(num_iterations):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        rk(grid, block, tuple(launch_args))
        stop.record()
        stop.synchronize()
        kernel_times.append(cp.cuda.get_elapsed_time(start, stop))

    cublas_times: list[float] = []
    if compare_cublas:
        for _ in range(warmup):
            cp.matmul(A, B)
        cp.cuda.runtime.deviceSynchronize()
        for _ in range(num_iterations):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            cp.matmul(A, B)
            stop.record()
            stop.synchronize()
            cublas_times.append(cp.cuda.get_elapsed_time(start, stop))

    clk_post, temp_post = _sample_clock_and_temp()

    kernel_times.sort()
    k_med = kernel_times[len(kernel_times) // 2]
    k_min, k_max = kernel_times[0], kernel_times[-1]

    flops = 2.0 * m * n * k
    gflops = flops / (k_med * 1e-3) / 1e9

    if cublas_times:
        cublas_times.sort()
        c_med = cublas_times[len(cublas_times) // 2]
        c_min, c_max = cublas_times[0], cublas_times[-1]
        cublas_gflops = flops / (c_med * 1e-3) / 1e9
        eff = gflops / cublas_gflops * 100.0 if cublas_gflops > 0 else None
    else:
        c_med = c_min = c_max = cublas_gflops = eff = None

    return MatmulBenchmarkResult(
        kernel_time_ms=k_med,
        kernel_min_ms=k_min,
        kernel_max_ms=k_max,
        cublas_time_ms=c_med,
        cublas_min_ms=c_min,
        cublas_max_ms=c_max,
        sm_clock_mhz_pre=clk_pre,
        sm_clock_mhz_post=clk_post,
        gpu_temp_c_pre=temp_pre,
        gpu_temp_c_post=temp_post,
        gflops=gflops,
        cublas_gflops=cublas_gflops,
        efficiency_pct=eff,
        dimensions=dict(dim_args),
    )
