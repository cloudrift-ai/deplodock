"""CUDA runtime dispatch for ``Graph[CudaOp]``.

Compiles each unique kernel source via NVRTC (through ``cupy.RawKernel``),
allocates a ``cupy.ndarray`` for every buffer in the graph, and walks
compute nodes in topological order launching kernels directly from Python.
No host ``.cu`` is generated — the only codegen that survives is the
per-kernel ``__global__`` function itself, emitted by ``ir/cuda/emit.py``.

Buffer roles come from the graph: ``graph.inputs`` → input,
``ConstantOp`` → constant, ``graph.outputs`` → output, everything else →
scratch. Launch order is ``graph.topological_order()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import BenchmarkResult, LaunchTime, RunResult
from deplodock.compiler.backend.cuda import _tma
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.cuda import CudaOp, TmaDescMeta

if TYPE_CHECKING:
    import cupy as cp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Buffer / launch classification
# ---------------------------------------------------------------------------


@dataclass
class _Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str  # "input" | "constant" | "output" | "scratch"

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
    for nid, node in graph.nodes.items():
        if nid in input_set:
            role = "input"
        elif isinstance(node.op, ConstantOp):
            role = "constant"
        elif nid in output_set:
            role = "output"
        else:
            role = "scratch"
        shape = tuple(int(d) for d in node.output.shape)
        bufs.append(_Buffer(name=nid, shape=shape, dtype="float32", role=role))
    return bufs


def _constant_values(graph: Graph) -> dict[str, float]:
    return {nid: node.op.value for nid, node in graph.nodes.items() if isinstance(node.op, ConstantOp) and node.op.value is not None}


def _launches(graph: Graph) -> list[Node]:
    nodes: list[Node] = []
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if isinstance(node.op, (InputOp, ConstantOp)):
            continue
        if not isinstance(node.op, CudaOp):
            raise TypeError(f"CudaBackend: node {nid!r} has non-CudaOp {type(node.op).__name__!r}; lowering must produce Graph[CudaOp].")
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Compiled program: RawKernels + buffer plan + launch list
# ---------------------------------------------------------------------------


@dataclass
class _Launch:
    node_id: str
    kernel_name: str
    arg_names: tuple[str, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem_bytes: int
    zero_outputs: tuple[str, ...]
    tma_descriptors: tuple[TmaDescMeta, ...] = ()


@dataclass
class _Compiled:
    bufs: list[_Buffer]
    buf_by_name: dict[str, _Buffer]
    constants: dict[str, float]
    kernels: dict[str, cp.RawKernel]  # kernel_name → RawKernel
    launches: list[_Launch]


def _compile(graph: Graph) -> _Compiled:
    import cupy as cp

    bufs = _buffers(graph)
    buf_by_name = {b.name: b for b in bufs}
    constants = _constant_values(graph)
    launches_nodes = _launches(graph)

    kernels: dict[str, cp.RawKernel] = {}
    seen_source: dict[str, str] = {}
    launches: list[_Launch] = []
    for node in launches_nodes:
        op: CudaOp = node.op
        kname = op.kernel_name
        if kname not in kernels:
            prev_src = seen_source.get(kname)
            if prev_src is not None and prev_src != op.kernel_source:
                raise ValueError(f"kernel name {kname!r} used by two distinct sources")
            seen_source[kname] = op.kernel_source
            options = _nvrtc_options(uses_tma=bool(op.tma_descriptors))
            kernels[kname] = cp.RawKernel(op.kernel_source, kname, options=options)
        launches.append(
            _Launch(
                node_id=node.id,
                kernel_name=kname,
                arg_names=tuple(op.arg_order),
                grid=op.grid,
                block=op.block,
                smem_bytes=op.smem_bytes,
                zero_outputs=tuple(op.zero_outputs),
                tma_descriptors=tuple(op.tma_descriptors),
            )
        )
    return _Compiled(bufs=bufs, buf_by_name=buf_by_name, constants=constants, kernels=kernels, launches=launches)


def _nvrtc_options(*, uses_tma: bool) -> tuple[str, ...]:
    """NVRTC compile options. TMA-using kernels need ``sm_<major><minor>a``
    (the ``a`` arch unlocks ``cp.async.bulk.tensor`` PTX). Non-TMA
    kernels keep the cupy default (capability inferred at runtime)."""
    base = ("--use_fast_math",)
    if not uses_tma:
        return base
    import cupy as cp

    cap_str = str(cp.cuda.Device().compute_capability)  # e.g. "120" for sm_12.0
    major, minor = cap_str[:-1], cap_str[-1]
    return (*base, f"--gpu-architecture=sm_{major}{minor}a")


# ---------------------------------------------------------------------------
# Buffer materialization
# ---------------------------------------------------------------------------


def _allocate(compiled: _Compiled, input_data: dict[str, np.ndarray] | None) -> dict[str, cp.ndarray]:
    import cupy as cp

    input_data = input_data or {}
    arrays: dict[str, cp.ndarray] = {}
    for buf in compiled.bufs:
        shape = buf.shape or (1,)
        src = input_data.get(buf.name)
        if src is not None:
            arr = cp.asarray(np.ascontiguousarray(src, dtype=np.float32).reshape(shape))
        elif buf.role == "constant" and buf.name in compiled.constants:
            arr = cp.full(shape, float(compiled.constants[buf.name]), dtype=cp.float32)
        elif buf.role == "input":
            # Pseudo-random fill for un-supplied inputs (matches old generated program).
            idx = np.arange(buf.size, dtype=np.float32)
            vals = 0.01 * ((idx * 7 + 13) % 101 - 50)
            arr = cp.asarray(vals.reshape(shape))
        else:
            arr = cp.zeros(shape, dtype=cp.float32)
        arrays[buf.name] = arr
    return arrays


def _launch(launch: _Launch, compiled: _Compiled, arrays: dict[str, cp.ndarray]) -> None:
    for zname in launch.zero_outputs:
        arrays[zname].fill(0)
    kernel = compiled.kernels[launch.kernel_name]
    desc_args = _build_descriptor_args(launch, arrays) if launch.tma_descriptors else {}
    args = tuple(desc_args.get(name) if name in desc_args else arrays[name] for name in launch.arg_names)
    kernel(launch.grid, launch.block, args, shared_mem=0)


def _build_descriptor_args(launch: _Launch, arrays: dict[str, cp.ndarray]) -> dict[str, cp.ndarray]:
    """Encode one ``CUtensorMap`` per descriptor referenced by ``launch``
    and stage it in a 128-byte device buffer.

    The kernel signature takes ``const CUtensorMap*`` (not a by-value
    ``__grid_constant__`` parameter) because cupy's arg-packing doesn't
    guarantee the 64-byte alignment required for by-value descriptors.
    Placing the descriptor in device memory and passing a pointer
    sidesteps the alignment concern — the TMA load PTX dereferences
    via a generic 64-bit pointer either way."""
    import cupy as cp

    out: dict[str, cp.ndarray] = {}
    for desc in launch.tma_descriptors:
        arr = arrays[desc.src_buf]
        desc_bytes = _tma.encode_tiled(
            global_address=int(arr.data.ptr),
            src_shape=tuple(int(d) for d in arr.shape),
            box_extents=desc.box_extents,
            elem_size=arr.itemsize,
            swizzle=desc.swizzle,
        )
        # Stage the 128 B descriptor in device memory. uint64 view keeps
        # the buffer 8-byte aligned (cupy.ndarray allocations are
        # typically 256-byte aligned, well above CUtensorMap's 64 B
        # requirement).
        d_arr = cp.asarray(np.frombuffer(desc_bytes, dtype=np.uint64))
        out[desc.name] = d_arr
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_program(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
    """Run the lowered graph once, return output ndarrays + total time."""
    import cupy as cp

    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for launch in compiled.launches:
        _launch(launch, compiled, arrays)
    stop.record()
    stop.synchronize()
    time_ms = cp.cuda.get_elapsed_time(start, stop)

    outputs: dict[str, np.ndarray] = {}
    for buf in compiled.bufs:
        if buf.role == "output":
            outputs[buf.name] = arrays[buf.name].get().astype(np.float32, copy=False)
    return RunResult(outputs=outputs, time_ms=time_ms)


@dataclass
class DebugResult:
    outputs: dict[str, np.ndarray]
    per_launch: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def run_program_debug(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> DebugResult:
    """Run the graph once, snapshotting every non-input buffer after each launch."""
    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)

    input_names = {b.name for b in compiled.bufs if b.role == "input"}
    per_launch_np: dict[int, dict[str, np.ndarray]] = {}

    for li, launch in enumerate(compiled.launches):
        _launch(launch, compiled, arrays)
        snap: dict[str, np.ndarray] = {}
        for name, arr in arrays.items():
            if name in input_names:
                continue
            snap[name] = arr.get().astype(np.float32, copy=False)
        per_launch_np[li] = snap

    outputs: dict[str, np.ndarray] = {}
    for buf in compiled.bufs:
        if buf.role == "output":
            outputs[buf.name] = arrays[buf.name].get().astype(np.float32, copy=False)
    return DebugResult(outputs=outputs, per_launch=per_launch_np)


def benchmark_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    warmup: int = 5,
    num_iters: int = 20,
) -> BenchmarkResult:
    """Time the graph's launches with per-kernel CUDA events."""
    import cupy as cp

    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)

    for _ in range(warmup):
        for launch in compiled.launches:
            _launch(launch, compiled, arrays)
    cp.cuda.runtime.deviceSynchronize()

    n = len(compiled.launches)
    starts = [cp.cuda.Event() for _ in range(n)]
    stops = [cp.cuda.Event() for _ in range(n)]
    acc = [0.0] * n

    prog_start = cp.cuda.Event()
    prog_stop = cp.cuda.Event()
    prog_start.record()
    for _ in range(num_iters):
        for i, launch in enumerate(compiled.launches):
            starts[i].record()
            _launch(launch, compiled, arrays)
            stops[i].record()
        stops[n - 1].synchronize()
        for i in range(n):
            acc[i] += cp.cuda.get_elapsed_time(starts[i], stops[i])
    prog_stop.record()
    prog_stop.synchronize()
    total_ms = cp.cuda.get_elapsed_time(prog_start, prog_stop)

    per_launch = [LaunchTime(idx=i, kernel_name=compiled.launches[i].kernel_name, time_ms=acc[i] / num_iters) for i in range(n)]
    return BenchmarkResult(
        time_ms=total_ms / num_iters,
        num_launches=n,
        per_launch=per_launch if per_launch else None,
    )


def make_runner(graph: Graph, input_data: dict[str, np.ndarray] | None = None):
    """Compile the graph, allocate buffers once, and return a zero-arg
    ``run_once()`` callable that issues one full pass of the kernel sequence.

    Used by interleaved benchmarking: callers warm up once, then alternate
    backends iteration-by-iteration so each backend sees the same warm GPU
    state (same clocks, same caches).
    """
    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)

    def run_once() -> None:
        for launch in compiled.launches:
            _launch(launch, compiled, arrays)

    return run_once
