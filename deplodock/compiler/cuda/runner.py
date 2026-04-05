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
