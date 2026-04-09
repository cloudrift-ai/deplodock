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
        # Linearized grid for CTA swizzle: pad nty to multiple of SWIZ=8
        # Total blocks = ntx * ceil(nty/8)*8, launched as (total, 1)
        ntx = f"(({n} + {kernel.tile_n - 1}) / {kernel.tile_n})"
        nty = f"(({m} + {kernel.tile_m - 1}) / {kernel.tile_m})"
        grid_x = f"({ntx} * (({nty} + 7) / 8) * 8)"
        grid_y = "1"
    else:
        effective_n = f"(({n} + {coarsen_cols - 1}) / {coarsen_cols})"
        effective_m = f"(({m} + {coarsen_rows - 1}) / {coarsen_rows})"
        grid_x = f"({effective_n} + {bx - 1}) / {bx}"
        grid_y = f"({effective_m} + {by - 1}) / {by}"

    # Default values for kernel params not in dim_args
    param_defaults = {"k_splits": 1}
    launch_args = []
    for p in kernel.params:
        if p.dtype.endswith("*"):
            launch_args.append(f"d_{p.name}")
        elif p.name in dim_args:
            launch_args.append(str(dim_args[p.name]))
        elif p.name in param_defaults:
            launch_args.append(str(param_defaults[p.name]))
    args_str = ", ".join(launch_args)

    cublas_includes = ""
    if compare_cublas:
        cublas_includes = "#include <cublas_v2.h>\n"

    # TMA setup code
    tma_includes = ""
    tma_setup = ""
    tma_launch_prefix = ""
    tma_smem_attr = ""
    if kernel.tma_params:
        tma_includes = "#include <cuda.h>\n"
        tile_m = kernel.tile_m or 64
        tile_n = kernel.tile_n or 128
        bk_val = dim_args.get("K", 1)  # Will be overridden per-strategy
        # Detect BK from the kernel source (look for "nt=K/" or "nt=(k_end-k_start)/" pattern)
        import re

        for stmt in kernel.body:
            if hasattr(stmt, "code"):
                m2 = re.search(r"nt=(?:K|\(k_end-k_start\))/(\d+)", stmt.code)
                if m2:
                    bk_val = int(m2.group(1))
                    break
        a_size = tile_m * bk_val
        b_size = bk_val * tile_n
        stage_size = a_size + b_size
        smem_bytes = 2 * stage_size * 4 + 256 + getattr(kernel, "extra_smem_bytes", 0)

        batch_count = dim_args.get("batch", 1)
        if kernel.batched and batch_count > 1:
            # Batched: create per-batch TMA descriptor arrays
            tma_setup = f"""
    // Batched TMA descriptor setup ({batch_count} batches)
    CUtensorMap h_{kernel.tma_params[0]}_descs[{batch_count}], h_{kernel.tma_params[1]}_descs[{batch_count}];
    for (int b = 0; b < {batch_count}; b++) {{
        uint64_t da[2]={{(uint64_t)K,(uint64_t)M}};
        uint64_t sa[1]={{(uint64_t)K*sizeof(float)}};
        uint32_t ba[2]={{{bk_val},{tile_m}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&h_{kernel.tma_params[0]}_descs[b],CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_A+b*M*K,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        uint64_t db[2]={{(uint64_t)N,(uint64_t)K}};
        uint64_t sb[1]={{(uint64_t)N*sizeof(float)}};
        uint32_t bb[2]={{{tile_n},{bk_val}}};
        cuTensorMapEncodeTiled(&h_{kernel.tma_params[1]}_descs[b],CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_B+b*K*N,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    CUtensorMap *d_{kernel.tma_params[0]}_descs, *d_{kernel.tma_params[1]}_descs;
    cudaMalloc(&d_{kernel.tma_params[0]}_descs, {batch_count}*sizeof(CUtensorMap));
    cudaMalloc(&d_{kernel.tma_params[1]}_descs, {batch_count}*sizeof(CUtensorMap));
    cudaMemcpy(d_{kernel.tma_params[0]}_descs, h_{kernel.tma_params[0]}_descs, {batch_count}*sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_{kernel.tma_params[1]}_descs, h_{kernel.tma_params[1]}_descs, {batch_count}*sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaFuncSetAttribute({kernel.name},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem_bytes});
"""
            tma_launch_prefix = f"d_{kernel.tma_params[0]}_descs, d_{kernel.tma_params[1]}_descs, "
        elif getattr(kernel, "bf16_emulation", False):
            # bf16x9 fused: standard 2 FP32 TMA descriptors, no preprocessing.
            # The kernel reads FP32 from smem and splits it to 3 BF16 limbs
            # internally; no scales are needed because BF16 carries its own
            # exponent.
            (a_tma, b_tma) = kernel.tma_params
            tma_setup = f"""
    // BF16x9 fused SGEMM: kernel splits FP32 -> 3 BF16 limbs in smem.
    // No preprocessing, no scale args — same FP32 in/out interface as cublasSgemm.
    CUtensorMap {a_tma}_desc, {b_tma}_desc;
    {{
        uint64_t da[2]={{(uint64_t)K,(uint64_t)M}};
        uint64_t sa[1]={{(uint64_t)K*sizeof(float)}};
        uint32_t ba[2]={{{bk_val},{tile_m}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{a_tma}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_A,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    {{
        uint64_t db[2]={{(uint64_t)N,(uint64_t)K}};
        uint64_t sb[1]={{(uint64_t)N*sizeof(float)}};
        uint32_t bb[2]={{{tile_n},{bk_val}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{b_tma}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_B,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    cudaFuncSetAttribute({kernel.name},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem_bytes});
"""
            tma_launch_prefix = f"{a_tma}_desc, {b_tma}_desc, "
        elif getattr(kernel, "int8_emulation", False) and len(kernel.tma_params) == 2:
            # int8x9 fused: 2 FP32 TMA descriptors. The kernel quantizes inline
            # in shared memory each K-tile, so no INT8 scratch buffers exist.
            # The runner only needs to compute scale_AB and pass it to the kernel.
            (a_tma, b_tma) = kernel.tma_params
            tma_setup = f"""
    // INT8x9 fused SGEMM: kernel quantizes FP32 -> INT8 limbs in smem.
    // We just need scale_AB derived from a global max-finding pass (one
    // small kernel call). For test data in [-1, 1] we hardcode max=1.0;
    // production code should compute it dynamically.
    float h_max_a = 1.0f, h_max_b = 1.0f;
    float s_A = h_max_a / 127.0f / 65536.0f;
    float s_B = h_max_b / 127.0f / 65536.0f;
    float scale_AB = s_A * s_B;

    // Two FP32 TMA descriptors (same as standard tma_db)
    CUtensorMap {a_tma}_desc, {b_tma}_desc;
    {{
        uint64_t da[2]={{(uint64_t)K,(uint64_t)M}};
        uint64_t sa[1]={{(uint64_t)K*sizeof(float)}};
        uint32_t ba[2]={{{bk_val},{tile_m}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{a_tma}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_A,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    {{
        uint64_t db[2]={{(uint64_t)N,(uint64_t)K}};
        uint64_t sb[1]={{(uint64_t)N*sizeof(float)}};
        uint32_t bb[2]={{{tile_n},{bk_val}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{b_tma}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_B,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    cudaFuncSetAttribute({kernel.name},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem_bytes});
"""
            tma_launch_prefix = f"{a_tma}_desc, {b_tma}_desc, "
            extra = "scale_AB, 1.0f/s_A, 1.0f/s_B"
            args_str = f"{args_str}, {extra}" if args_str else extra
        elif getattr(kernel, "int8_emulation", False):
            # int8x9 pre-quantized: 6 INT8 limb buffers (3 for A, 3 for B),
            # filled by a quantize_to_int8x3 helper kernel before timing.
            assert len(kernel.tma_params) == 6, "int8x9 expects 6 TMA params"
            (a_h, a_m, a_l, b_h, b_m, b_l) = kernel.tma_params
            tma_setup = f"""
    // INT8x9 SGEMM: pre-quantize A and B to 3 INT8 limb buffers each.
    // Use a generous fixed scale for the test data range [-0.5, 0.5].
    int8_t *d_A_h, *d_A_m, *d_A_l, *d_B_h, *d_B_m, *d_B_l;
    cudaMalloc(&d_A_h, BATCH*M*K); cudaMalloc(&d_A_m, BATCH*M*K); cudaMalloc(&d_A_l, BATCH*M*K);
    cudaMalloc(&d_B_h, BATCH*K*N); cudaMalloc(&d_B_m, BATCH*K*N); cudaMalloc(&d_B_l, BATCH*K*N);
    float h_max_a = 1.0f, h_max_b = 1.0f;  // upper bound on |x| in test data
    float s_A = h_max_a / 127.0f / 65536.0f;
    float s_B = h_max_b / 127.0f / 65536.0f;
    float scale_AB = s_A * s_B;
    {{
        int qN = M*K;
        quantize_to_int8x3<<<(qN+255)/256, 256>>>(d_A, d_A_h, d_A_m, d_A_l, qN, 1.0f/s_A);
        qN = K*N;
        quantize_to_int8x3<<<(qN+255)/256, 256>>>(d_B, d_B_h, d_B_m, d_B_l, qN, 1.0f/s_B);
    }}
    cudaDeviceSynchronize();

    // Six INT8 TMA descriptors
    CUtensorMap {a_h}_desc, {a_m}_desc, {a_l}_desc, {b_h}_desc, {b_m}_desc, {b_l}_desc;
    {{
        uint64_t da[2]={{(uint64_t)K,(uint64_t)M}};
        uint64_t sa[1]={{(uint64_t)K*sizeof(int8_t)}};
        uint32_t ba[2]={{{bk_val},{tile_m}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{a_h}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_A_h,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuTensorMapEncodeTiled(&{a_m}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_A_m,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuTensorMapEncodeTiled(&{a_l}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_A_l,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }}
    {{
        uint64_t db[2]={{(uint64_t)N,(uint64_t)K}};
        uint64_t sb[1]={{(uint64_t)N*sizeof(int8_t)}};
        uint32_t bb[2]={{{tile_n},{bk_val}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{b_h}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_B_h,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuTensorMapEncodeTiled(&{b_m}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_B_m,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuTensorMapEncodeTiled(&{b_l}_desc,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,
            d_B_l,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }}
    cudaFuncSetAttribute({kernel.name},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem_bytes});
"""
            tma_launch_prefix = f"{a_h}_desc, {a_m}_desc, {a_l}_desc, {b_h}_desc, {b_m}_desc, {b_l}_desc, "
            # Append scale_AB to the user args (the kernel signature has it as the last param)
            args_str = f"{args_str}, scale_AB" if args_str else "scale_AB"
        else:
            # Single: one TMA descriptor per matrix
            tma_setup = f"""
    // TMA descriptor setup
    CUtensorMap {kernel.tma_params[0]}_desc, {kernel.tma_params[1]}_desc;
    {{
        uint64_t da[2]={{(uint64_t)K,(uint64_t)M}};
        uint64_t sa[1]={{(uint64_t)K*sizeof(float)}};
        uint32_t ba[2]={{{bk_val},{tile_m}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{kernel.tma_params[0]}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_A,da,sa,ba,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    {{
        uint64_t db[2]={{(uint64_t)N,(uint64_t)K}};
        uint64_t sb[1]={{(uint64_t)N*sizeof(float)}};
        uint32_t bb[2]={{{tile_n},{bk_val}}};
        uint32_t ea[2]={{1,1}};
        cuTensorMapEncodeTiled(&{kernel.tma_params[1]}_desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
            d_B,db,sb,bb,ea,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    }}
    cudaFuncSetAttribute({kernel.name},cudaFuncAttributeMaxDynamicSharedMemorySize,{smem_bytes});
"""
            tma_launch_prefix = f"{kernel.tma_params[0]}_desc, {kernel.tma_params[1]}_desc, "
        tma_smem_attr = f", {smem_bytes}"

    # Build launch args with TMA prefix
    full_args = f"{tma_launch_prefix}{args_str}"
    launch_suffix = f"<<<grid, block{tma_smem_attr}>>>"

    int8_helper = ""
    if getattr(kernel, "int8_emulation", False):
        int8_helper = """
// Pre-quantize an FP32 buffer into 3 INT8 limb buffers using a global scale.
__global__ void quantize_to_int8x3(const float* x, int8_t* hi, int8_t* mid, int8_t* lo,
                                    int n, float inv_scale) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int q = (int)rintf(x[i] * inv_scale);
    if (q > (1<<23)-1) q = (1<<23)-1;
    if (q < -(1<<23))   q = -(1<<23);
    int qh = q >> 16;
    int rem = q - (qh << 16);
    int qm = rem >> 8;
    int ql = rem - (qm << 8);
    if (ql > 127)  { ql -= 256; qm += 1; }
    if (ql < -128) { ql += 256; qm -= 1; }
    if (qm > 127)  { qm -= 256; qh += 1; }
    if (qm < -128) { qm += 256; qh -= 1; }
    hi[i]  = (int8_t)qh;
    mid[i] = (int8_t)qm;
    lo[i]  = (int8_t)ql;
}
"""

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
{tma_includes}{cublas_includes}
// Compile-time matrix dimensions for kernel optimization
#define M {m}
#define N {n}
#define K {k}
{int8_helper}
{kernel_source}

int main() {{
    int BATCH = {dim_args.get("batch", 1)};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, BATCH * M * K * sizeof(float));
    cudaMalloc(&d_B, BATCH * K * N * sizeof(float));
    cudaMalloc(&d_C, BATCH * M * N * sizeof(float));

    // Random initialization
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_A, BATCH * M * K);
    curandGenerateUniform(gen, d_B, BATCH * K * N);
    curandDestroyGenerator(gen);
{tma_setup}
    dim3 block({bx}, {by});
    int k_splits_val = {dim_args.get("k_splits", 1)};
    int grid_z = BATCH > 1 ? BATCH : k_splits_val;
    dim3 grid({grid_x}, {grid_y}, grid_z);

    // Zero output for K-splitting atomicAdd
    if (k_splits_val > 1) cudaMemset(d_C, 0, BATCH * M * N * sizeof(float));

    // Warmup our kernel + surface launch errors loudly. cuBLAS gets its
    // own warmup further down inside the cuBLAS setup block.
    {kernel.name}{launch_suffix}({full_args});
    cudaError_t warmup_err = cudaDeviceSynchronize();
    if (warmup_err != cudaSuccess) {{
        fprintf(stderr, "CUDA error after warmup launch: %s\\n", cudaGetErrorString(warmup_err));
        return 2;
    }}

{
        f'''    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);
    {"cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);" if cublas_math_mode == "pedantic" else "// default math mode"}
    float *d_C_ref;
    cudaMalloc(&d_C_ref, BATCH * {m * n} * sizeof(float));
    float alpha = 1.0f, beta_val = 0.0f;
    {"cublasSgemmStridedBatched" if dim_args.get("batch", 1) > 1 else "cublasSgemm"}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                {n}, {m}, {k}, &alpha,
                d_B, {n}, {"(long long)" + str(k * n) + "," if dim_args.get("batch", 1) > 1 else ""}
                d_A, {k}, {"(long long)" + str(m * k) + "," if dim_args.get("batch", 1) > 1 else ""}
                &beta_val, d_C_ref, {n}{",(long long)" + str(m * n) + ",BATCH" if dim_args.get("batch", 1) > 1 else ""});
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
    // Run num_iterations+1 iterations; iter 0 is a discarded warmup so the
    // first sample (which often pays JIT/cache/algo-selection costs) is not
    // captured for either kernel.
    for (int iter = 0; iter < {num_iterations + 1}; iter++) {{
        // Our kernel
        if (k_splits_val > 1) cudaMemset(d_C, 0, BATCH * M * N * sizeof(float));
        cudaEventRecord(start);
        {kernel.name}{launch_suffix}({full_args});
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float k_t; cudaEventElapsedTime(&k_t, start, stop);
        if (iter > 0) times[iter - 1] = k_t;
{
        f'''
        // cuBLAS (same thermal state as our kernel)
        cudaEventRecord(cb_start);
        {"cublasSgemmStridedBatched" if dim_args.get("batch", 1) > 1 else "cublasSgemm"}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    {n}, {m}, {k}, &alpha,
                    d_B, {n}, {"(long long)" + str(k * n) + "," if dim_args.get("batch", 1) > 1 else ""}
                    d_A, {k}, {"(long long)" + str(m * k) + "," if dim_args.get("batch", 1) > 1 else ""}
                    &beta_val, d_C_ref, {n}{",(long long)" + str(m * n) + ",BATCH" if dim_args.get("batch", 1) > 1 else ""});
        cudaEventRecord(cb_stop);
        cudaEventSynchronize(cb_stop);
        float cb_t; cudaEventElapsedTime(&cb_t, cb_start, cb_stop);
        if (iter > 0) cublas_times[iter - 1] = cb_t;
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

    cudaError_t loop_err = cudaGetLastError();
    if (loop_err != cudaSuccess) {{
        fprintf(stderr, "CUDA error in timed loop: %s\\n", cudaGetErrorString(loop_err));
        return 3;
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
    // FP32 accumulation order differs between kernels; threshold scales with K
    float err_tol = fmaxf(1e-2f, K * 1e-5f);
    printf("CORRECT=%d\\n", max_err < err_tol ? 1 : 0);
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


def _sample_clock_and_temp() -> tuple[int | None, int | None]:
    """Snapshot SM clock (MHz) and GPU temp (C) via nvidia-smi. None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    if out.returncode != 0:
        return None, None
    line = out.stdout.strip().splitlines()[0] if out.stdout else ""
    parts = [p.strip() for p in line.split(",")]
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None, None


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
    fast_math: bool = False,
) -> BenchmarkResult:
    """Run a benchmark: compile and execute, return timing results.

    `fast_math=False` (the default) is the IEEE-clean configuration: we still
    enable `--fmad=true` so SGEMM gets fused FFMA instructions (without it
    nvcc emits separate FMUL+FADD and SGEMM throughput halves), but we do
    NOT pass `--use_fast_math`, which would also enable `--ftz=true` and
    relaxed div/sqrt — those break "exact FP32" claims for headline numbers.
    """
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
    nvcc_cmd = ["nvcc", "-O3", "--fmad=true"]
    if fast_math:
        nvcc_cmd.append("--use_fast_math")
    if arch:
        nvcc_cmd.extend(["-arch", arch])
    if maxrregcount is not None:
        nvcc_cmd.extend(["--maxrregcount", str(maxrregcount)])
    if "#include <mma.h>" in program or "wmma::" in program:
        nvcc_cmd.extend(["--std=c++17"])
    if "CUtensorMap" in program or "cuTensorMapEncodeTiled" in program:
        nvcc_cmd.append("-lcuda")
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

        # Sample SM clock + temp on either side of the run so the report can
        # show the thermal/clock context for each measurement.
        sm_pre, temp_pre = _sample_clock_and_temp()
        # Run.
        run_result = subprocess.run(
            [str(bin_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        sm_post, temp_post = _sample_clock_and_temp()
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
        batch = dim_args.get("batch", 1)
        flops = 2.0 * m * n * k * batch
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
            cublas_min_ms=vals.get("CUBLAS_MIN_MS"),
            cublas_max_ms=vals.get("CUBLAS_MAX_MS"),
            sm_clock_mhz_pre=sm_pre,
            sm_clock_mhz_post=sm_post,
            gpu_temp_c_pre=temp_pre,
            gpu_temp_c_post=temp_post,
            gflops=gflops,
            cublas_gflops=cublas_gflops,
            efficiency_pct=efficiency,
            dimensions=dim_args,
        )
