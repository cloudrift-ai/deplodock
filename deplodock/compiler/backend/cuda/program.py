"""CUDA codegen + execution driver for ``Graph[CudaOp]``.

Walks the lowered graph to emit a full ``.cu`` program: buffer
allocation, input/constant initialization, kernel launches, output
readback. The graph is the single source of truth — buffer roles are
derived from ``graph.inputs`` / ``graph.outputs`` / ``ConstantOp``
membership, shapes from ``node.output.shape``, and launch order from
topological order.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deplodock.compiler.backend import BenchmarkResult, LaunchTime, ProgramResult
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.graph import Graph, Node

logger = logging.getLogger(__name__)


@dataclass
class TmaDescriptorSpec:
    """Spec for creating a CUtensorMap descriptor at runtime."""

    param_name: str
    buffer: str
    dims: list[str]
    strides: list[str]
    tile: list[int]


# ---------------------------------------------------------------------------
# Buffer / launch classification
# ---------------------------------------------------------------------------


@dataclass
class _Buffer:
    name: str
    shape: tuple[int | str, ...]
    dtype: str
    role: str

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


def _buffers(graph: Graph) -> list[_Buffer]:
    input_set = set(graph.inputs)
    output_set = set(graph.outputs)
    bufs: list[_Buffer] = []
    for nid in graph.nodes:
        node = graph.nodes[nid]
        if nid in input_set:
            role = "input"
        elif isinstance(node.op, ConstantOp):
            role = "constant"
        elif nid in output_set:
            role = "output"
        else:
            role = "scratch"
        bufs.append(_Buffer(name=nid, shape=tuple(node.output.shape), dtype="float", role=role))
    return bufs


def _constant_values(graph: Graph) -> dict[str, float]:
    out: dict[str, float] = {}
    for nid, node in graph.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.value is not None:
            out[nid] = node.op.value
    return out


def _launches(graph: Graph) -> list[Node]:
    nodes: list[Node] = []
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if isinstance(node.op, (InputOp, ConstantOp)):
            continue
        if not isinstance(node.op, CudaOp):
            raise TypeError(
                f"CudaBackend: node {nid!r} has non-CudaOp {type(node.op).__name__!r}; "
                "lowering passes must produce Graph[CudaOp] before codegen."
            )
        nodes.append(node)
    return nodes


def _buffer_size_map(bufs: list[_Buffer]) -> dict[str, int]:
    return {b.name: b.size for b in bufs}


def graph_name(graph: Graph) -> str:
    return getattr(graph, "name", None) or "prog"


def graph_shape(graph: Graph, name: str) -> tuple:
    node = graph.nodes.get(name)
    if node is None:
        raise KeyError(f"Buffer {name!r} not in graph")
    return tuple(node.output.shape)


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def generate_source(
    graph: Graph,
    mode: str = "benchmark",
    num_iters: int = 10,
    warmup: int = 3,
    input_data: dict[str, np.ndarray] | None = None,
    debug: bool = False,
) -> str:
    """Generate a complete .cu program from a lowered graph.

    Args:
        graph: ``Graph[CudaOp + InputOp + ConstantOp]``.
        mode: "run" (print outputs) or "benchmark" (timed iterations).
        num_iters: Number of timed iterations (benchmark mode).
        warmup: Number of warmup iterations.
        debug: If True, dump every non-input buffer after each kernel launch.
    """
    bufs = _buffers(graph)
    launches = _launches(graph)
    constant_values = _constant_values(graph)

    has_tma = any(getattr(n.op, "tma_descriptors", ()) for n in launches)
    parts: list[str] = []

    parts.append("#include <cstdio>")
    parts.append("#include <cstdlib>")
    parts.append("#include <cmath>")
    parts.append("#include <cuda_runtime.h>")
    if has_tma:
        parts.append("#include <cuda.h>")
    if mode == "benchmark":
        parts.append("#include <float.h>")
    parts.append("")

    seen_kernels: set[str] = set()
    for node in launches:
        op: CudaOp = node.op
        if op.kernel_name not in seen_kernels:
            parts.append(op.kernel_source)
            seen_kernels.add(op.kernel_name)
    parts.append("")

    parts.append("int main() {")

    for buf in bufs:
        parts.append(f"    {buf.dtype}* d_{buf.name};")
        parts.append(f"    cudaMalloc(&d_{buf.name}, {buf.size} * sizeof({buf.dtype}));")
    parts.append("")

    for buf in bufs:
        if buf.role in ("input", "constant"):
            if input_data and buf.name in input_data:
                parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
                parts.append(f'      FILE* fp = fopen("{buf.name}.bin", "rb");')
                parts.append(f"      fread(h, sizeof({buf.dtype}), {buf.size}, fp); fclose(fp);")
                parts.append(f"      cudaMemcpy(d_{buf.name}, h, {buf.size} * sizeof({buf.dtype}), cudaMemcpyHostToDevice);")
                parts.append("      free(h); }")
            elif buf.role == "constant" and buf.name in constant_values:
                value = constant_values[buf.name]
                parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
                parts.append(f"      for (int i = 0; i < {buf.size}; i++) h[i] = ({buf.dtype}){value!r}f;")
                parts.append(f"      cudaMemcpy(d_{buf.name}, h, {buf.size} * sizeof({buf.dtype}), cudaMemcpyHostToDevice);")
                parts.append("      free(h); }")
            else:
                parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
                parts.append(f"      for (int i = 0; i < {buf.size}; i++) h[i] = 0.01f * ((i * 7 + 13) % 101 - 50);")
                parts.append(f"      cudaMemcpy(d_{buf.name}, h, {buf.size} * sizeof({buf.dtype}), cudaMemcpyHostToDevice);")
                parts.append("      free(h); }")
    parts.append("")

    if has_tma:
        parts.append(_generate_tma_setup(launches, bufs))
        parts.append("")

    buf_names = {b.name for b in bufs}
    buf_sizes = _buffer_size_map(bufs)

    if debug:
        buf_by_name = {b.name: b for b in bufs}
        parts.append('    printf("DEBUG_START\\n");')
        for li, node in enumerate(launches):
            single = _generate_single_launch(node, buf_names, buf_sizes)
            parts.append(f"    // --- launch {li}: {node.op.kernel_name} ---")
            parts.append(single)
            parts.append(f"    cudaError_t _err{li} = cudaDeviceSynchronize();")
            parts.append(f"    if (_err{li} != cudaSuccess) {{")
            parts.append(f'        fprintf(stderr, "LAUNCH_FAIL {li} {node.op.kernel_name}: %s\\n", cudaGetErrorString(_err{li}));')
            parts.append("        return 1;")
            parts.append("    }")
            out_name = _launch_output_name(node, buf_names)
            if out_name is not None and out_name in buf_by_name:
                buf = buf_by_name[out_name]
                parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
                parts.append(f"      cudaMemcpy(h, d_{buf.name}, {buf.size} * sizeof({buf.dtype}), cudaMemcpyDeviceToHost);")
                parts.append(f'      FILE* fp = fopen("launch_{li:03d}.bin", "wb");')
                parts.append(f"      fwrite(h, sizeof({buf.dtype}), {buf.size}, fp); fclose(fp);")
                parts.append(f'      printf("DBUF {li} {buf.name} {buf.size}\\n");')
                parts.append("      free(h); }")
        parts.append('    printf("DEBUG_END\\n");')
        parts.append("")
    else:
        if mode == "benchmark":
            parts.append(f"    for (int _w = 0; _w < {warmup}; _w++) {{")
            parts.append(_generate_launches(launches, buf_names, buf_sizes))
            parts.append("    }")
            parts.append("    cudaDeviceSynchronize();")
            parts.append("")

            n_launches = len(launches)
            parts.append(f"    const int NL = {n_launches};")
            parts.append("    cudaEvent_t _ks[NL], _ke[NL];")
            parts.append("    for (int i = 0; i < NL; i++) { cudaEventCreate(&_ks[i]); cudaEventCreate(&_ke[i]); }")
            parts.append("    float _kacc[NL] = {0};")
            parts.append("    cudaEvent_t _start, _stop;")
            parts.append("    cudaEventCreate(&_start);")
            parts.append("    cudaEventCreate(&_stop);")
            parts.append("")
            parts.append("    cudaEventRecord(_start);")
            parts.append(f"    for (int _iter = 0; _iter < {num_iters}; _iter++) {{")
            parts.append(_generate_timed_launches(launches, buf_names, buf_sizes))
            parts.append("        cudaEventSynchronize(_ke[NL - 1]);")
            parts.append("        for (int i = 0; i < NL; i++) {")
            parts.append("            float _ms; cudaEventElapsedTime(&_ms, _ks[i], _ke[i]); _kacc[i] += _ms;")
            parts.append("        }")
            parts.append("    }")
            parts.append("    cudaEventRecord(_stop);")
            parts.append("    cudaEventSynchronize(_stop);")
            parts.append("")
            parts.append("    float _total_ms;")
            parts.append("    cudaEventElapsedTime(&_total_ms, _start, _stop);")
            parts.append(f'    printf("PROGRAM_TIME_MS=%.4f\\n", _total_ms / {num_iters});')
            parts.append(f'    printf("PROGRAM_LAUNCHES={n_launches}\\n");')
            for li, node in enumerate(launches):
                parts.append(f'    printf("KERNEL_TIME_MS {li} {node.op.kernel_name} %.6f\\n", _kacc[{li}] / {num_iters});')
            parts.append("")
            parts.append("    for (int i = 0; i < NL; i++) { cudaEventDestroy(_ks[i]); cudaEventDestroy(_ke[i]); }")
            parts.append("    cudaEventDestroy(_start);")
            parts.append("    cudaEventDestroy(_stop);")
        else:
            launch_code = _generate_launches(launches, buf_names, buf_sizes)
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

    for buf in bufs:
        if buf.role == "output":
            parts.append(f"    {{ {buf.dtype}* h = ({buf.dtype}*)malloc({buf.size} * sizeof({buf.dtype}));")
            parts.append(f"      cudaMemcpy(h, d_{buf.name}, {buf.size} * sizeof({buf.dtype}), cudaMemcpyDeviceToHost);")
            parts.append(f'      for (int i = 0; i < {buf.size}; i++) printf("OUT {buf.name} %.6f\\n", h[i]);')
            parts.append("      free(h); }")
    parts.append("")

    for buf in bufs:
        parts.append(f"    cudaFree(d_{buf.name});")
    parts.append("    return 0;")
    parts.append("}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Launch emission
# ---------------------------------------------------------------------------


def _tma_var(kernel_name: str, desc: TmaDescriptorSpec) -> str:
    return f"{kernel_name}_{desc.param_name}_desc"


def _generate_tma_setup(launches: list[Node], bufs: list[_Buffer]) -> str:
    lines: list[str] = []
    buf_names = {b.name for b in bufs}
    for node in launches:
        op: CudaOp = node.op
        tma_descs = getattr(op, "tma_descriptors", ())
        for desc in tma_descs:
            var = _tma_var(op.kernel_name, desc)
            buf = _format_arg(desc.buffer, buf_names)
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

        if tma_descs and op.smem_bytes > 0:
            lines.append(f"    cudaFuncSetAttribute({op.kernel_name},cudaFuncAttributeMaxDynamicSharedMemorySize,{op.smem_bytes});")

    return "\n".join(lines)


def _generate_launches(launches: list[Node], buf_names: set[str], buf_sizes: dict[str, int]) -> str:
    lines: list[str] = []
    for node in launches:
        op: CudaOp = node.op
        for buf_name in op.zero_outputs:
            size = buf_sizes.get(buf_name, 0)
            lines.append(f"        cudaMemset(d_{buf_name}, 0, {size} * sizeof(float));")

        tma_descs = getattr(op, "tma_descriptors", ())
        tma_args = [_tma_var(op.kernel_name, d) for d in tma_descs]
        regular_args = [_format_arg(a, buf_names) for a in op.arg_order]
        args_str = ", ".join(tma_args + regular_args)

        gx, gy, gz = op.grid
        bx, by, bz = op.block
        if gz == 1 and bz == 1:
            grid_str = f"dim3({gx}, {gy})" if gy > 1 else str(gx)
            block_str = f"dim3({bx}, {by})" if by > 1 else str(bx)
        else:
            grid_str = f"dim3({gx}, {gy}, {gz})"
            block_str = f"dim3({bx}, {by}, {bz})"
        smem = f", {op.smem_bytes}" if op.smem_bytes > 0 else ""
        lines.append(f"        {op.kernel_name}<<<{grid_str}, {block_str}{smem}>>>({args_str});")
    return "\n".join(lines)


def _generate_timed_launches(launches: list[Node], buf_names: set[str], buf_sizes: dict[str, int]) -> str:
    lines: list[str] = []
    for li, node in enumerate(launches):
        op: CudaOp = node.op
        for buf_name in op.zero_outputs:
            size = buf_sizes.get(buf_name, 0)
            lines.append(f"        cudaMemset(d_{buf_name}, 0, {size} * sizeof(float));")

        tma_descs = getattr(op, "tma_descriptors", ())
        tma_args = [_tma_var(op.kernel_name, d) for d in tma_descs]
        regular_args = [_format_arg(a, buf_names) for a in op.arg_order]
        args_str = ", ".join(tma_args + regular_args)

        gx, gy, gz = op.grid
        bx, by, bz = op.block
        if gz == 1 and bz == 1:
            grid_str = f"dim3({gx}, {gy})" if gy > 1 else str(gx)
            block_str = f"dim3({bx}, {by})" if by > 1 else str(bx)
        else:
            grid_str = f"dim3({gx}, {gy}, {gz})"
            block_str = f"dim3({bx}, {by}, {bz})"
        smem = f", {op.smem_bytes}" if op.smem_bytes > 0 else ""
        lines.append(f"        cudaEventRecord(_ks[{li}]);")
        lines.append(f"        {op.kernel_name}<<<{grid_str}, {block_str}{smem}>>>({args_str});")
        lines.append(f"        cudaEventRecord(_ke[{li}]);")
    return "\n".join(lines)


def _generate_single_launch(node: Node, buf_names: set[str], buf_sizes: dict[str, int]) -> str:
    op: CudaOp = node.op
    lines: list[str] = []
    for buf_name in op.zero_outputs:
        size = buf_sizes.get(buf_name, 0)
        lines.append(f"    cudaMemset(d_{buf_name}, 0, {size} * sizeof(float));")

    tma_descs = getattr(op, "tma_descriptors", ())
    tma_args = [_tma_var(op.kernel_name, d) for d in tma_descs]
    regular_args = [_format_arg(a, buf_names) for a in op.arg_order]
    args_str = ", ".join(tma_args + regular_args)

    gx, gy, gz = op.grid
    bx, by, bz = op.block
    if gz == 1 and bz == 1:
        grid_str = f"dim3({gx}, {gy})" if gy > 1 else str(gx)
        block_str = f"dim3({bx}, {by})" if by > 1 else str(bx)
    else:
        grid_str = f"dim3({gx}, {gy}, {gz})"
        block_str = f"dim3({bx}, {by}, {bz})"
    smem = f", {op.smem_bytes}" if op.smem_bytes > 0 else ""
    lines.append(f"    {op.kernel_name}<<<{grid_str}, {block_str}{smem}>>>({args_str});")
    return "\n".join(lines)


def _launch_output_name(node: Node, buf_names: set[str]) -> str | None:
    for arg in reversed(node.op.arg_order):
        if arg in buf_names:
            return arg
    return None


def _format_arg(arg: str, buf_names: set[str]) -> str:
    if arg in buf_names:
        return f"d_{arg}"
    return arg


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "deplodock" / "kernels"


def compile_program(source: str, arch: str | None = None) -> Path:
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
    if "CUtensorMap" in source:
        cmd.append("-lcuda")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")

    return binary


def run_program(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
    source = generate_source(graph, mode="run", input_data=input_data)
    binary = compile_program(source)

    with tempfile.TemporaryDirectory(prefix="deplodock_run_") as rundir:
        if input_data:
            for buf_name, vals in input_data.items():
                data_path = Path(rundir) / f"{buf_name}.bin"
                data_path.write_bytes(np.ascontiguousarray(vals, dtype=np.float32).tobytes())

        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120, cwd=rundir)
        if result.returncode != 0:
            raise RuntimeError(f"Program execution failed:\n{result.stderr}")

        return _parse_run_output(result.stdout)


@dataclass
class DebugResult:
    outputs: dict[str, list[float]]
    per_launch: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def run_program_debug(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> DebugResult:
    source = generate_source(graph, mode="run", input_data=input_data, debug=True)
    binary = compile_program(source)

    with tempfile.TemporaryDirectory(prefix="deplodock_dbg_") as rundir:
        rundir_path = Path(rundir)
        if input_data:
            for buf_name, vals in input_data.items():
                data_path = rundir_path / f"{buf_name}.bin"
                data_path.write_bytes(np.ascontiguousarray(vals, dtype=np.float32).tobytes())

        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300, cwd=rundir)
        if result.returncode != 0:
            raise RuntimeError(f"Debug program execution failed:\n{result.stderr}\nstdout:\n{result.stdout}")

        return _parse_debug_output(result.stdout, graph, rundir_path)


def benchmark_program(graph: Graph, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
    source = generate_source(graph, mode="benchmark", num_iters=num_iters, warmup=warmup)
    binary = compile_program(source)

    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Program execution failed:\n{result.stderr}")

    return _parse_benchmark_output(result.stdout)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_run_output(stdout: str) -> ProgramResult:
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


def _parse_benchmark_output(stdout: str) -> BenchmarkResult:
    time_ms = 0.0
    num_launches = 0
    per_launch: list[LaunchTime] = []

    for line in stdout.strip().split("\n"):
        if line.startswith("PROGRAM_TIME_MS="):
            time_ms = float(line.split("=")[1])
        elif line.startswith("PROGRAM_LAUNCHES="):
            num_launches = int(line.split("=")[1])
        elif line.startswith("KERNEL_TIME_MS "):
            parts = line.split()
            per_launch.append(LaunchTime(idx=int(parts[1]), kernel_name=parts[2], time_ms=float(parts[3])))

    per_launch.sort(key=lambda lt: lt.idx)
    return BenchmarkResult(
        time_ms=time_ms,
        num_launches=num_launches,
        per_launch=per_launch if per_launch else None,
    )


def _parse_debug_output(stdout: str, graph: Graph, workdir: Path) -> DebugResult:
    outputs: dict[str, list[float]] = {}
    buf_shapes = {nid: tuple(node.output.shape) for nid, node in graph.nodes.items()}
    per_launch_np: dict[int, dict[str, np.ndarray]] = {}

    for line in stdout.strip().split("\n"):
        if line.startswith("DBUF "):
            parts = line.split()
            launch_idx = int(parts[1])
            buf_name = parts[2]
            numel = int(parts[3])
            bin_path = workdir / f"launch_{launch_idx:03d}.bin"
            if not bin_path.exists():
                continue
            raw = np.fromfile(bin_path, dtype=np.float32, count=numel)
            shape = buf_shapes.get(buf_name, (numel,))
            per_launch_np.setdefault(launch_idx, {})[buf_name] = raw.reshape(shape)
        elif line.startswith("OUT "):
            parts = line.split()
            outputs.setdefault(parts[1], []).append(float(parts[2]))

    return DebugResult(outputs=outputs, per_launch=per_launch_np)


def _detect_arch() -> str:
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
    return "sm_89"
