"""Compile and run CUDA kernels."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deplodock.compiler.cuda.ir import KernelDef

logger = logging.getLogger(__name__)


def has_nvcc() -> bool:
    """Check if nvcc is available on PATH."""
    return shutil.which("nvcc") is not None


def has_cuda_gpu() -> bool:
    """Check if a CUDA GPU is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def generate_host_program(
    kernel_source: str,
    kernel: KernelDef,
    inputs: dict[str, list[float]],
    output_name: str,
    output_size: int,
    dim_args: dict[str, int],
) -> str:
    """Generate a complete .cu file with kernel + host wrapper.

    Args:
        kernel_source: CUDA kernel source from codegen.
        kernel: KernelDef for block size and parameter info.
        inputs: Mapping of param name → flat float data.
        output_name: Name of the output parameter.
        output_size: Number of elements in the output.
        dim_args: Mapping of dimension param name → int value (e.g. M, N, K).
    """
    lines = [
        "#include <stdio.h>",
        "#include <cuda_runtime.h>",
        "",
        kernel_source,
        "",
        "int main() {",
    ]

    # Declare host input arrays.
    for name, data in inputs.items():
        values = ", ".join(f"{v:.6f}f" for v in data)
        lines.append(f"    float h_{name}[] = {{{values}}};")

    # Declare host output array.
    lines.append(f"    float h_{output_name}[{output_size}];")
    lines.append("")

    # Allocate device memory.
    all_arrays = list(inputs.keys()) + [output_name]
    for name in all_arrays:
        lines.append(f"    float* d_{name};")

    for name, data in inputs.items():
        lines.append(f"    cudaMalloc(&d_{name}, {len(data)} * sizeof(float));")
    lines.append(f"    cudaMalloc(&d_{output_name}, {output_size} * sizeof(float));")
    lines.append("")

    # Copy inputs to device.
    for name, data in inputs.items():
        lines.append(f"    cudaMemcpy(d_{name}, h_{name}, {len(data)} * sizeof(float), cudaMemcpyHostToDevice);")
    lines.append("")

    # Compute grid dimensions.
    bx, by, _bz = kernel.block_size
    # For matmul: grid.x covers N (cols), grid.y covers M (rows).
    grid_x = f"({dim_args.get('N', 1)} + {bx - 1}) / {bx}"
    grid_y = f"({dim_args.get('M', 1)} + {by - 1}) / {by}"
    lines.append(f"    dim3 block({bx}, {by});")
    lines.append(f"    dim3 grid({grid_x}, {grid_y});")
    lines.append("")

    # Build kernel launch arguments.
    launch_args = []
    for p in kernel.params:
        if p.name in inputs or p.name == output_name:
            launch_args.append(f"d_{p.name}")
        elif p.name in dim_args:
            launch_args.append(str(dim_args[p.name]))
        else:
            raise ValueError(f"No value for kernel param {p.name!r}")

    args_str = ", ".join(launch_args)

    # CUDA event timing.
    lines.append("    cudaEvent_t start, stop;")
    lines.append("    cudaEventCreate(&start);")
    lines.append("    cudaEventCreate(&stop);")
    lines.append("")

    # Warmup launch.
    lines.append(f"    {kernel.name}<<<grid, block>>>({args_str});")
    lines.append("    cudaDeviceSynchronize();")
    lines.append("")

    # Timed launch.
    lines.append("    cudaEventRecord(start);")
    lines.append(f"    {kernel.name}<<<grid, block>>>({args_str});")
    lines.append("    cudaEventRecord(stop);")
    lines.append("    cudaEventSynchronize(stop);")
    lines.append("")

    lines.append("    float elapsed_ms = 0.0f;")
    lines.append("    cudaEventElapsedTime(&elapsed_ms, start, stop);")
    lines.append("")

    # Copy result back.
    lines.append(f"    cudaMemcpy(h_{output_name}, d_{output_name}, {output_size} * sizeof(float), cudaMemcpyDeviceToHost);")
    lines.append("")

    # Print result line, then timing line (separate lines for easy parsing).
    lines.append(f"    for (int i = 0; i < {output_size}; i++) {{")
    lines.append(f'        printf("%.6f ", h_{output_name}[i]);')
    lines.append("    }")
    lines.append('    printf("\\n");')
    lines.append('    printf("KERNEL_TIME_MS=%.6f\\n", elapsed_ms);')
    lines.append("")
    lines.append("    cudaEventDestroy(start);")
    lines.append("    cudaEventDestroy(stop);")
    lines.append("")

    # Cleanup.
    for name in all_arrays:
        lines.append(f"    cudaFree(d_{name});")
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


@dataclass
class KernelResult:
    """Result of a kernel execution."""

    output: list[float]
    kernel_time_ms: float | None = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run across iterations."""

    kernel_time_ms: float  # median
    kernel_min_ms: float
    kernel_max_ms: float
    cublas_time_ms: float | None = None
    gflops: float = 0.0
    cublas_gflops: float | None = None
    efficiency_pct: float | None = None
    dimensions: dict[str, int] | None = None


def run_kernel(
    kernel: KernelDef,
    kernel_source: str,
    inputs: dict[str, list[float]],
    output_name: str,
    output_size: int,
    dim_args: dict[str, int],
) -> KernelResult:
    """Compile and run a CUDA kernel, return output and timing.

    Args:
        kernel: KernelDef for metadata (block size, params).
        kernel_source: CUDA kernel source from codegen.
        inputs: Mapping of param name → flat float data.
        output_name: Name of the output parameter.
        output_size: Number of elements in the output.
        dim_args: Mapping of dimension param name → int value.

    Returns:
        KernelResult with output floats and optional GPU timing.
    """
    program = generate_host_program(kernel_source, kernel, inputs, output_name, output_size, dim_args)

    with tempfile.TemporaryDirectory(prefix="deplodock_cuda_") as tmpdir:
        src_path = Path(tmpdir) / "kernel.cu"
        bin_path = Path(tmpdir) / "kernel"

        src_path.write_text(program)
        logger.debug("CUDA source written to %s", src_path)

        # Compile.
        compile_result = subprocess.run(
            ["nvcc", "-o", str(bin_path), str(src_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"nvcc compilation failed:\n{compile_result.stderr}")

        # Run.
        run_result = subprocess.run(
            [str(bin_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"Kernel execution failed:\n{run_result.stderr}")

        # Parse output: first line = data, second line = timing.
        lines = run_result.stdout.strip().splitlines()
        output = [float(x) for x in lines[0].split()]
        kernel_time_ms = None
        for line in lines[1:]:
            if line.startswith("KERNEL_TIME_MS="):
                kernel_time_ms = float(line.split("=", 1)[1])
        return KernelResult(output=output, kernel_time_ms=kernel_time_ms)


def _detect_arch() -> str | None:
    """Detect best supported nvcc arch for the installed GPU.

    Queries the GPU compute capability and clamps to what nvcc supports.
    """
    try:
        # Get supported architectures from nvcc.
        nvcc_result = subprocess.run(
            ["nvcc", "--list-gpu-arch"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        supported = set()
        if nvcc_result.returncode == 0:
            for line in nvcc_result.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("compute_"):
                    supported.add(line.replace("compute_", "sm_"))

        if not supported:
            return None

        # Get GPU compute capability.
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if gpu_result.returncode != 0:
            return None

        cap = gpu_result.stdout.strip().split("\n")[0].strip()
        major, minor = cap.split(".")
        target = f"sm_{major}{minor}"

        if target in supported:
            return target

        # Fall back to highest supported arch.
        sorted_archs = sorted(supported, key=lambda s: int(s.replace("sm_", "")))
        return sorted_archs[-1] if sorted_archs else None

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def generate_benchmark_program(
    kernel_source: str,
    kernel: KernelDef,
    dim_args: dict[str, int],
    num_iterations: int = 10,
    compare_cublas: bool = True,
    coarsen_cols: int = 1,
    coarsen_rows: int = 1,
    cublas_math_mode: str = "default",
) -> str:
    """Generate a .cu benchmark program with random data and cuBLAS comparison.

    Uses curand for random initialization so we don't embed large arrays.
    Runs multiple timed iterations and reports median, min, max.
    coarsen_cols/rows: how many output elements each thread computes.
    """
    m = dim_args.get("M", 1)
    n = dim_args.get("N", 1)
    k = dim_args.get("K", 1)

    bx, by, _bz = kernel.block_size
    if kernel.tile_m and kernel.tile_n:
        grid_x = f"({n} + {kernel.tile_n - 1}) / {kernel.tile_n}"
        grid_y = f"({m} + {kernel.tile_m - 1}) / {kernel.tile_m}"
    else:
        effective_n = f"(({n} + {coarsen_cols - 1}) / {coarsen_cols})"
        effective_m = f"(({m} + {coarsen_rows - 1}) / {coarsen_rows})"
        grid_x = f"({effective_n} + {bx - 1}) / {bx}"
        grid_y = f"({effective_m} + {by - 1}) / {by}"

    launch_args = []
    for p in kernel.params:
        if p.dtype.endswith("*"):
            launch_args.append(f"d_{p.name}")
        elif p.name in dim_args:
            launch_args.append(str(dim_args[p.name]))
    args_str = ", ".join(launch_args)

    cublas_includes = ""
    if compare_cublas:
        cublas_includes = "#include <cublas_v2.h>\n"

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
{cublas_includes}
{kernel_source}

int main() {{
    int M = {m}, N = {n}, K = {k};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Random initialization
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_A, M * K);
    curandGenerateUniform(gen, d_B, K * N);
    curandDestroyGenerator(gen);

    dim3 block({bx}, {by});
    dim3 grid({grid_x}, {grid_y});

    // Warmup both kernels
    {kernel.name}<<<grid, block>>>({args_str});
    cudaDeviceSynchronize();

{
        f'''    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    {"cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);" if cublas_math_mode == "pedantic" else "// default math mode"}
    float *d_C_ref;
    cudaMalloc(&d_C_ref, {m * n} * sizeof(float));
    float alpha = 1.0f, beta_val = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                {n}, {m}, {k}, &alpha,
                d_B, {n}, d_A, {k}, &beta_val, d_C_ref, {n});
    cudaDeviceSynchronize();
'''
        if compare_cublas
        else ""
    }
    // Timed iterations — INTERLEAVED for thermal fairness
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
{
        '''    cudaEvent_t cb_start, cb_stop;
    cudaEventCreate(&cb_start);
    cudaEventCreate(&cb_stop);
'''
        if compare_cublas
        else ""
    }
    float times[{num_iterations}];
{f"    float cublas_times[{num_iterations}];" if compare_cublas else ""}
    for (int iter = 0; iter < {num_iterations}; iter++) {{
        // Our kernel
        cudaEventRecord(start);
        {kernel.name}<<<grid, block>>>({args_str});
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[iter], start, stop);
{
        f'''
        // cuBLAS (same thermal state as our kernel)
        cudaEventRecord(cb_start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    {n}, {m}, {k}, &alpha,
                    d_B, {n}, d_A, {k}, &beta_val, d_C_ref, {n});
        cudaEventRecord(cb_stop);
        cudaEventSynchronize(cb_stop);
        cudaEventElapsedTime(&cublas_times[iter], cb_start, cb_stop);
'''
        if compare_cublas
        else ""
    }    }}

    // Sort for median
    for (int i = 0; i < {num_iterations} - 1; i++)
        for (int j = i + 1; j < {num_iterations}; j++)
            if (times[j] < times[i]) {{
                float tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }}

    printf("KERNEL_MEDIAN_MS=%.6f\\n", times[{num_iterations} / 2]);
    printf("KERNEL_MIN_MS=%.6f\\n", times[0]);
    printf("KERNEL_MAX_MS=%.6f\\n", times[{num_iterations} - 1]);

{
        f'''    // Sort cublas times
    for (int i = 0; i < {num_iterations} - 1; i++)
        for (int j = i + 1; j < {num_iterations}; j++)
            if (cublas_times[j] < cublas_times[i]) {{
                float tmp = cublas_times[i];
                cublas_times[i] = cublas_times[j];
                cublas_times[j] = tmp;
            }}
    printf("CUBLAS_MEDIAN_MS=%.6f\\n", cublas_times[{num_iterations} / 2]);
    printf("CUBLAS_MIN_MS=%.6f\\n", cublas_times[0]);
    printf("CUBLAS_MAX_MS=%.6f\\n", cublas_times[{num_iterations} - 1]);

    // Correctness check
    float *h_ours = (float*)malloc({m * n} * sizeof(float));
    float *h_ref = (float*)malloc({m * n} * sizeof(float));
    cudaMemcpy(h_ours, d_C, {m * n} * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, d_C_ref, {m * n} * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < {m * n}; i++) {{
        float err = fabsf(h_ours[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        float rel = err / fmaxf(fabsf(h_ref[i]), 1e-6f);
        if (rel > max_rel_err) max_rel_err = rel;
    }}
    printf("MAX_ERROR=%.6f\\n", max_err);
    printf("MAX_REL_ERROR=%.6f\\n", max_rel_err);
    printf("CORRECT=%d\\n", max_err < 1e-2 ? 1 : 0);
    free(h_ours);
    free(h_ref);
    cudaEventDestroy(cb_start);
    cudaEventDestroy(cb_stop);
    cudaFree(d_C_ref);
    cublasDestroy(handle);
'''
        if compare_cublas
        else ""
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}}
"""


def run_benchmark(
    kernel: KernelDef,
    kernel_source: str,
    dim_args: dict[str, int],
    num_iterations: int = 10,
    compare_cublas: bool = True,
    coarsen_cols: int = 1,
    coarsen_rows: int = 1,
    cublas_math_mode: str = "default",
    maxrregcount: int | None = None,
    fast_math: bool = True,
) -> BenchmarkResult:
    """Run a benchmark: compile and execute, return timing results."""
    program = generate_benchmark_program(
        kernel_source,
        kernel,
        dim_args,
        num_iterations,
        compare_cublas,
        coarsen_cols=coarsen_cols,
        coarsen_rows=coarsen_rows,
        cublas_math_mode=cublas_math_mode,
    )

    arch = _detect_arch()
    nvcc_cmd = ["nvcc", "-O3"]
    if fast_math:
        nvcc_cmd.append("--use_fast_math")
    if arch:
        nvcc_cmd.extend(["-arch", arch])
    if maxrregcount is not None:
        nvcc_cmd.extend(["--maxrregcount", str(maxrregcount)])
    if "#include <mma.h>" in program or "wmma::" in program:
        nvcc_cmd.extend(["--std=c++17"])
    if compare_cublas:
        nvcc_cmd.extend(["-lcublas", "-lcurand"])
    else:
        nvcc_cmd.append("-lcurand")

    with tempfile.TemporaryDirectory(prefix="deplodock_bench_") as tmpdir:
        src_path = Path(tmpdir) / "bench.cu"
        bin_path = Path(tmpdir) / "bench"

        src_path.write_text(program)
        logger.debug("Benchmark source written to %s", src_path)

        # Compile.
        compile_result = subprocess.run(
            [*nvcc_cmd, "-o", str(bin_path), str(src_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"nvcc compilation failed:\n{compile_result.stderr}")

        # Run.
        run_result = subprocess.run(
            [str(bin_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"Benchmark execution failed:\n{run_result.stderr}")

        # Parse output.
        vals: dict[str, float] = {}
        for line in run_result.stdout.strip().splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                try:
                    vals[key] = float(val)
                except ValueError:
                    pass

        kernel_median = vals.get("KERNEL_MEDIAN_MS", 0.0)
        kernel_min = vals.get("KERNEL_MIN_MS", 0.0)
        kernel_max = vals.get("KERNEL_MAX_MS", 0.0)
        cublas_median = vals.get("CUBLAS_MEDIAN_MS")
        max_error = vals.get("MAX_ERROR")
        correct = vals.get("CORRECT")

        if correct is not None and correct < 1:
            logger.warning("Kernel produced incorrect results! max_error=%.6f", max_error)

        m = dim_args.get("M", 1)
        n = dim_args.get("N", 1)
        k = dim_args.get("K", 1)
        flops = 2.0 * m * n * k
        gflops = (flops / (kernel_median * 1e-3)) / 1e9 if kernel_median > 0 else 0.0
        cublas_gflops = None
        efficiency = None
        if cublas_median is not None and cublas_median > 0:
            cublas_gflops = (flops / (cublas_median * 1e-3)) / 1e9
            if cublas_gflops > 0:
                efficiency = (gflops / cublas_gflops) * 100.0

        return BenchmarkResult(
            kernel_time_ms=kernel_median,
            kernel_min_ms=kernel_min,
            kernel_max_ms=kernel_max,
            cublas_time_ms=cublas_median,
            gflops=gflops,
            cublas_gflops=cublas_gflops,
            efficiency_pct=efficiency,
            dimensions=dim_args,
        )
