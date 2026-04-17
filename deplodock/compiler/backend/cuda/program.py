"""CUDA program: source generation, compilation, and execution.

Extends the backend-agnostic ``GpuProgram`` / ``GpuLaunch`` / ``GpuBuffer``
with CUDA-specific features (TMA descriptors, nvcc compilation, GPU
execution).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from deplodock.compiler.backend import BenchmarkResult, ProgramResult
from deplodock.compiler.program.gpu import GpuBuffer, GpuLaunch, GpuProgram  # noqa: F401 — re-export

logger = logging.getLogger(__name__)


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
class CudaLaunch(GpuLaunch):
    """CUDA kernel launch with optional TMA descriptor metadata."""

    tma_descriptors: list[TmaDescriptorSpec] = field(default_factory=list)
    loop_ir: object | None = None  # LoopProgram, if generated via LoopIR pipeline
    schedule: object | None = None  # Schedule, if generated via LoopIR pipeline


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def generate_source(
    program: GpuProgram,
    mode: str = "benchmark",
    num_iters: int = 10,
    warmup: int = 3,
    input_data: dict[str, list[float]] | None = None,
) -> str:
    """Generate a complete .cu program from a Program spec.

    Args:
        program: The program to generate.
        mode: "run" (print outputs) or "benchmark" (timed iterations).
        num_iters: Number of timed iterations (benchmark mode).
        warmup: Number of warmup iterations.
    """
    has_tma = any(getattr(launch, "tma_descriptors", None) for launch in program.launches)
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

    # Allocate buffers. Aliased buffers share the target's device pointer.
    aliased = set(program.aliases.keys())
    for buf in program.buffers:
        if buf.name in aliased:
            continue  # will be assigned after all allocations
        parts.append(f"    {buf.dtype}* d_{buf.name};")
        parts.append(f"    cudaMalloc(&d_{buf.name}, {buf.size} * sizeof({buf.dtype}));")
    for alias_name, target_name in program.aliases.items():
        buf = next(b for b in program.buffers if b.name == alias_name)
        parts.append(f"    {buf.dtype}* d_{alias_name} = d_{target_name};")
    parts.append("")

    # Initialize inputs and constants.
    for buf in program.buffers:
        if buf.role in ("input", "constant") and buf.name not in aliased:
            if input_data and buf.name in input_data:
                # Use provided data: read from binary file at runtime.
                parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
                # Write data via binary file read for large buffers.
                parts.append(f'      FILE* fp = fopen("{buf.name}.bin", "rb");')
                parts.append(f"      fread(h, sizeof({buf.dtype}), {buf.size}, fp); fclose(fp);")
                parts.append(f"      cudaMemcpy(d_{buf.name}, h, {buf.size} * sizeof({buf.dtype}), cudaMemcpyHostToDevice);")
                parts.append("      free(h); }")
            else:
                # Pseudorandom fallback.
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
        # Single run with GPU-event timing (kernel-only, excludes subprocess/nvcc overhead).
        parts.append("    cudaEvent_t _start, _stop;")
        parts.append("    cudaEventCreate(&_start);")
        parts.append("    cudaEventCreate(&_stop);")
        parts.append("")
        parts.append("    cudaEventRecord(_start);")
        parts.append(launch_code)
        parts.append("    cudaEventRecord(_stop);")
        parts.append("    cudaEventSynchronize(_stop);")
        parts.append("")
        parts.append("    float _total_ms;")
        parts.append("    cudaEventElapsedTime(&_total_ms, _start, _stop);")
        parts.append('    printf("PROGRAM_TIME_MS=%.4f\\n", _total_ms);')
        parts.append("")
        parts.append("    cudaEventDestroy(_start);")
        parts.append("    cudaEventDestroy(_stop);")
    parts.append("")

    # Read back outputs.
    for buf in program.buffers:
        if buf.role == "output":
            parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
            parts.append(f"      cudaMemcpy(h, d_{buf.name}, {buf.size} * sizeof({buf.dtype}), cudaMemcpyDeviceToHost);")
            parts.append(f'      for (int i = 0; i < {buf.size}; i++) printf("OUT {buf.name} %.6f\\n", h[i]);')
            parts.append("      free(h); }")
    parts.append("")

    # Free (skip aliases — they share another buffer's allocation).
    for buf in program.buffers:
        if buf.name not in aliased:
            parts.append(f"    cudaFree(d_{buf.name});")
    parts.append("    return 0;")
    parts.append("}")

    return "\n".join(parts)


def _tma_var(launch: GpuLaunch, desc: TmaDescriptorSpec) -> str:
    """Unique variable name for a TMA descriptor: kernel_param_desc."""
    return f"{launch.kernel_name}_{desc.param_name}_desc"


def _generate_tma_setup(program: GpuProgram) -> str:
    """Generate CUtensorMap descriptor creation code for all TMA launches.

    Each TMA descriptor encodes a 2D tile view of a global-memory buffer.
    cuTensorMapEncodeTiled parameters:
      - d[2]: global tensor dimensions (innermost-first, column-major order).
      - s[1]: stride between rows, in bytes (only 1 stride for 2D).
      - b[2]: block (tile) sizes that cp.async.bulk will load per TMA op.
      - e[2]: element strides (always 1 — no strided access within a tile).

    Flag choices:
      - INTERLEAVE_NONE: elements are contiguous in memory (no channel
        interleaving). Required for standard row-major FP32 layouts.
      - SWIZZLE_NONE: no shared-memory swizzle pattern. Swizzling reduces
        bank conflicts for certain tile shapes, but the double-buffer
        strategy already avoids conflicts via staging through registers.
      - L2_PROMOTION_L2_256B: hint the L2 cache to keep 256B lines for
        TMA traffic. Improves reuse when multiple CTAs read overlapping
        K-slices of the same matrix.
      - FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA: out-of-bounds tile elements
        are filled with zero for FMA operations (not NaN). Allows the
        kernel to use uniform tile sizes without explicit boundary checks;
        OOB elements contribute zero to the accumulator.
    """
    lines: list[str] = []

    for launch in program.launches:
        tma_descs = getattr(launch, "tma_descriptors", [])
        for desc in tma_descs:
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
            lines.append(f"        cuTensorMapEncodeTiled(&{var},CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,")
            lines.append(f"            {buf},d,s,b,e,")
            lines.append("            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,")
            lines.append("            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);")
            lines.append("    }")

        # Dynamic shared memory attribute.
        if tma_descs and launch.smem_bytes > 0:
            lines.append(f"    cudaFuncSetAttribute({launch.kernel_name},cudaFuncAttributeMaxDynamicSharedMemorySize,{launch.smem_bytes});")

    return "\n".join(lines)


def _generate_launches(program: GpuProgram) -> str:
    """Generate the kernel launch statements."""
    lines: list[str] = []
    buf_sizes = {b.name: b.size for b in program.buffers}
    for launch in program.launches:
        # Zero output buffers if needed (k_splits atomicAdd).
        for buf_name in launch.zero_outputs:
            size = buf_sizes.get(buf_name, 0)
            lines.append(f"        cudaMemset(d_{buf_name}, 0, {size} * sizeof(float));")

        # TMA descriptors are prepended to the regular args.
        tma_descs = getattr(launch, "tma_descriptors", [])
        tma_args = [_tma_var(launch, d) for d in tma_descs]
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


def _format_arg(arg: str, program: GpuProgram) -> str:
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


def run_program(program: GpuProgram, input_data: dict[str, list[float]] | None = None) -> ProgramResult:
    """Generate, compile, and run a Program. Returns outputs + timing."""
    source = generate_source(program, mode="run", input_data=input_data)
    binary = compile_program(source)

    # Write input data files next to the binary if provided.
    if input_data:
        import struct

        for buf_name, vals in input_data.items():
            data_path = binary.parent / f"{buf_name}.bin"
            data_path.write_bytes(struct.pack(f"{len(vals)}f", *vals))

    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120, cwd=str(binary.parent))
    if result.returncode != 0:
        raise RuntimeError(f"Program execution failed:\n{result.stderr}")

    return _parse_run_output(result.stdout, program)


def benchmark_program(
    program: GpuProgram,
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


def _parse_run_output(stdout: str, program: GpuProgram) -> ProgramResult:
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


def _parse_benchmark_output(stdout: str, program: GpuProgram) -> BenchmarkResult:
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
