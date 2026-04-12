"""Unified GPU program abstraction.

A Program is a self-contained description of a CUDA computation:
buffers, kernel sources, and an ordered launch sequence. One runner
generates a .cu file from any Program, compiles it, and executes it.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.backend import BenchmarkResult, ProgramResult

logger = logging.getLogger(__name__)


@dataclass
class Buffer:
    """GPU buffer specification."""

    name: str
    size: int  # total number of elements
    dtype: str = "float"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"


@dataclass
class TmaDescriptorSpec:
    """Spec for creating a CUtensorMap descriptor at runtime.

    Each spec produces one CUtensorMap via cuTensorMapEncodeTiled().
    The descriptor is passed to the kernel as a __grid_constant__ param.
    """

    param_name: str  # kernel parameter name (e.g., "A_tma")
    buffer: str  # device buffer name (e.g., "mul_1")
    dims: list[str]  # [dim0, dim1] expressions for tensor dimensions
    strides: list[str]  # [stride0] expressions for byte strides
    tile: list[int]  # [tile0, tile1] block tile sizes


@dataclass
class Launch:
    """One kernel invocation."""

    kernel_source: str  # complete __global__ function
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[str]  # buffer names and scalar literals in param order
    smem_bytes: int = 0
    tma_descriptors: list[TmaDescriptorSpec] = field(default_factory=list)


@dataclass
class Program:
    """A complete GPU program: buffers + kernels + launch order."""

    name: str
    buffers: list[Buffer]
    launches: list[Launch]
    defines: dict[str, str] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def generate_source(program: Program, mode: str = "benchmark", num_iters: int = 10, warmup: int = 3) -> str:
    """Generate a complete .cu program from a Program spec.

    Args:
        program: The program to generate.
        mode: "run" (print outputs) or "benchmark" (timed iterations).
        num_iters: Number of timed iterations (benchmark mode).
        warmup: Number of warmup iterations.
    """
    has_tma = any(launch.tma_descriptors for launch in program.launches)
    parts: list[str] = []

    # Includes.
    parts.append("#include <cstdio>")
    parts.append("#include <cstdlib>")
    parts.append("#include <cmath>")
    parts.append("#include <cuda_runtime.h>")
    if has_tma:
        parts.append("#include <cuda.h>")
    if mode == "benchmark":
        parts.append("#include <float.h>")
    for inc in program.includes:
        parts.append(f"#include <{inc}>")
    parts.append("")

    # Defines.
    for name, value in program.defines.items():
        parts.append(f"#define {name} {value}")
    if program.defines:
        parts.append("")

    # Kernel sources (deduplicate by name).
    seen_kernels: set[str] = set()
    for launch in program.launches:
        if launch.kernel_name not in seen_kernels:
            parts.append(launch.kernel_source)
            seen_kernels.add(launch.kernel_name)
    parts.append("")

    # Host main.
    parts.append("int main() {")

    # Allocate buffers.
    for buf in program.buffers:
        parts.append(f"    {buf.dtype}* d_{buf.name};")
        parts.append(f"    cudaMalloc(&d_{buf.name}, {buf.size} * sizeof({buf.dtype}));")
    parts.append("")

    # Initialize inputs and constants with pseudorandom data.
    for buf in program.buffers:
        if buf.role in ("input", "constant"):
            parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
            parts.append(f"      for (int i = 0; i < {buf.size}; i++) h[i] = 0.01f * ((i * 7 + 13) % 101 - 50);")
            parts.append(f"      cudaMemcpy(d_{buf.name}, h, {buf.size} * sizeof({buf.dtype}), cudaMemcpyHostToDevice);")
            parts.append("      free(h); }")
    parts.append("")

    # TMA descriptor setup (once, before any launches).
    if has_tma:
        parts.append(_generate_tma_setup(program))
        parts.append("")

    # Build launch code block.
    launch_code = _generate_launches(program)

    if mode == "benchmark":
        # Warmup.
        parts.append(f"    for (int _w = 0; _w < {warmup}; _w++) {{")
        parts.append(launch_code)
        parts.append("    }")
        parts.append("    cudaDeviceSynchronize();")
        parts.append("")

        # Timed iterations.
        parts.append("    cudaEvent_t _start, _stop;")
        parts.append("    cudaEventCreate(&_start);")
        parts.append("    cudaEventCreate(&_stop);")
        parts.append("")
        parts.append("    cudaEventRecord(_start);")
        parts.append(f"    for (int _iter = 0; _iter < {num_iters}; _iter++) {{")
        parts.append(launch_code)
        parts.append("    }")
        parts.append("    cudaEventRecord(_stop);")
        parts.append("    cudaEventSynchronize(_stop);")
        parts.append("")
        parts.append("    float _total_ms;")
        parts.append("    cudaEventElapsedTime(&_total_ms, _start, _stop);")
        parts.append(f'    printf("PROGRAM_TIME_MS=%.4f\\n", _total_ms / {num_iters});')
        parts.append(f'    printf("PROGRAM_LAUNCHES={len(program.launches)}\\n");')
        parts.append("")
        parts.append("    cudaEventDestroy(_start);")
        parts.append("    cudaEventDestroy(_stop);")
    else:
        # Single run.
        parts.append(launch_code)
        parts.append("    cudaDeviceSynchronize();")
    parts.append("")

    # Read back outputs.
    for buf in program.buffers:
        if buf.role == "output":
            parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
            parts.append(f"      cudaMemcpy(h, d_{buf.name}, {buf.size} * sizeof({buf.dtype}), cudaMemcpyDeviceToHost);")
            parts.append(f'      for (int i = 0; i < {buf.size}; i++) printf("OUT {buf.name} %.6f\\n", h[i]);')
            parts.append("      free(h); }")
    parts.append("")

    # Free.
    for buf in program.buffers:
        parts.append(f"    cudaFree(d_{buf.name});")
    parts.append("    return 0;")
    parts.append("}")

    return "\n".join(parts)


def _tma_var(launch: Launch, desc: TmaDescriptorSpec) -> str:
    """Unique variable name for a TMA descriptor: kernel_param_desc."""
    return f"{launch.kernel_name}_{desc.param_name}_desc"


def _generate_tma_setup(program: Program) -> str:
    """Generate CUtensorMap descriptor creation code for all TMA launches."""
    lines: list[str] = []

    for launch in program.launches:
        for desc in launch.tma_descriptors:
            var = _tma_var(launch, desc)
            buf = _format_arg(desc.buffer, program)
            d0, d1 = desc.dims
            s0 = desc.strides[0]
            t0, t1 = desc.tile

            lines.append(f"    CUtensorMap {var};")
            lines.append("    {")
            lines.append(f"        uint64_t d[2]={{(uint64_t)({d0}),(uint64_t)({d1})}};")
            lines.append(f"        uint64_t s[1]={{(uint64_t)(({s0})*sizeof(float))}};")
            lines.append(f"        uint32_t b[2]={{{t0},{t1}}};")
            lines.append("        uint32_t e[2]={1,1};")
            lines.append(
                f"        cuTensorMapEncodeTiled(&{var},CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,"
            )
            lines.append(
                f"            {buf},d,s,b,e,"
            )
            lines.append(
                "            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,"
            )
            lines.append(
                "            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,"
                "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);"
            )
            lines.append("    }")

        # Dynamic shared memory attribute.
        if launch.tma_descriptors and launch.smem_bytes > 0:
            lines.append(
                f"    cudaFuncSetAttribute({launch.kernel_name},"
                f"cudaFuncAttributeMaxDynamicSharedMemorySize,{launch.smem_bytes});"
            )

    return "\n".join(lines)


def _generate_launches(program: Program) -> str:
    """Generate the kernel launch statements."""
    lines: list[str] = []
    for launch in program.launches:
        # TMA descriptors are prepended to the regular args.
        tma_args = [_tma_var(launch, d) for d in launch.tma_descriptors]
        regular_args = [_format_arg(a, program) for a in launch.args]
        all_args = tma_args + regular_args
        args_str = ", ".join(all_args)

        gx, gy, gz = launch.grid
        bx, by, bz = launch.block
        if gz == 1 and bz == 1:
            grid_str = f"dim3({gx}, {gy})" if gy > 1 else str(gx)
            block_str = f"dim3({bx}, {by})" if by > 1 else str(bx)
        else:
            grid_str = f"dim3({gx}, {gy}, {gz})"
            block_str = f"dim3({bx}, {by}, {bz})"
        smem = f", {launch.smem_bytes}" if launch.smem_bytes > 0 else ""
        lines.append(f"        {launch.kernel_name}<<<{grid_str}, {block_str}{smem}>>>({args_str});")
    return "\n".join(lines)


def _format_arg(arg: str, program: Program) -> str:
    """Format a launch argument: buffer names become d_name, others pass through."""
    buf_names = {b.name for b in program.buffers}
    if arg in buf_names:
        return f"d_{arg}"
    return arg  # scalar literal like "256" or "1e-5f"


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "deplodock" / "kernels"


def compile_program(source: str, arch: str | None = None) -> Path:
    """Compile CUDA source to a binary, with content-hash caching."""
    content_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    binary = _CACHE_DIR / f"prog_{content_hash}"

    if binary.exists():
        return binary

    cu_path = _CACHE_DIR / f"prog_{content_hash}.cu"
    cu_path.write_text(source)

    if arch is None:
        arch = _detect_arch()

    cmd = ["nvcc", "-O2", "--use_fast_math", f"-arch={arch}", str(cu_path), "-o", str(binary)]
    # Link libcuda when using TMA (CUtensorMap / cuTensorMapEncodeTiled).
    if "CUtensorMap" in source:
        cmd.append("-lcuda")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")

    return binary


def run_program(program: Program) -> ProgramResult:
    """Generate, compile, and run a Program. Returns outputs + timing."""
    source = generate_source(program, mode="run")
    binary = compile_program(source)

    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Program execution failed:\n{result.stderr}")

    return _parse_run_output(result.stdout, program)


def benchmark_program(
    program: Program,
    warmup: int = 5,
    num_iters: int = 20,
) -> BenchmarkResult:
    """Generate, compile, and benchmark a Program. Returns timing."""
    source = generate_source(program, mode="benchmark", num_iters=num_iters, warmup=warmup)
    binary = compile_program(source)

    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Program execution failed:\n{result.stderr}")

    return _parse_benchmark_output(result.stdout, program)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_run_output(stdout: str, program: Program) -> ProgramResult:
    """Parse output from a run-mode program."""
    outputs: dict[str, list[float]] = {}
    time_ms = None

    for line in stdout.strip().split("\n"):
        if line.startswith("OUT "):
            parts = line.split()
            buf_name = parts[1]
            value = float(parts[2])
            outputs.setdefault(buf_name, []).append(value)
        elif line.startswith("PROGRAM_TIME_MS="):
            time_ms = float(line.split("=")[1])

    return ProgramResult(outputs=outputs, time_ms=time_ms)


def _parse_benchmark_output(stdout: str, program: Program) -> BenchmarkResult:
    """Parse output from a benchmark-mode program."""
    time_ms = 0.0
    num_launches = 0

    for line in stdout.strip().split("\n"):
        if line.startswith("PROGRAM_TIME_MS="):
            time_ms = float(line.split("=")[1])
        elif line.startswith("PROGRAM_LAUNCHES="):
            num_launches = int(line.split("=")[1])

    return BenchmarkResult(time_ms=time_ms, num_launches=num_launches)


def _detect_arch() -> str:
    """Detect the GPU architecture for nvcc."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split("\n")[0].strip()
            return f"sm_{cap.replace('.', '')}"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "sm_89"  # fallback
