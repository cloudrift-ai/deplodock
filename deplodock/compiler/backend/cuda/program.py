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

# Mirror of ``ir.kernel.render.STATIC_SMEM_CAP`` — kept here to avoid
# pulling the renderer into the runtime path.
_STATIC_SMEM_CAP = 48 * 1024


def _ensure_dynamic_smem_attr(kernel: cp.RawKernel, smem_bytes: int) -> None:
    """Opt this kernel into the device's max dynamic-smem allowance.

    Required when ``smem_bytes`` exceeds the 48 KB static cap. cupy's
    ``RawKernel.max_dynamic_shared_size_bytes`` setter calls
    ``cuFuncSetAttribute(MaxDynamicSharedMemorySize)``; the driver
    clamps to the device's per-block dynamic max (e.g. ~99 KB on
    sm_120). Already-set kernels are skipped.
    """
    if kernel.max_dynamic_shared_size_bytes >= smem_bytes:
        return
    kernel.max_dynamic_shared_size_bytes = smem_bytes


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


def _launch(
    launch: _Launch,
    compiled: _Compiled,
    arrays: dict[str, cp.ndarray],
    desc_args: dict[str, cp.ndarray] | None = None,
) -> None:
    for zname in launch.zero_outputs:
        arrays[zname].fill(0)
    kernel = compiled.kernels[launch.kernel_name]
    desc_args = desc_args or {}
    args = tuple(desc_args.get(name) if name in desc_args else arrays[name] for name in launch.arg_names)
    # Kernels whose Smem footprint exceeds the 48 KB static cap declare
    # an ``extern __shared__`` pool; the launch supplies the byte size
    # via ``shared_mem=`` and (for footprints above 48 KB) opts into the
    # device's larger dynamic-smem allowance via ``cudaFuncSetAttribute``.
    smem_bytes = launch.smem_bytes
    if smem_bytes > _STATIC_SMEM_CAP:
        _ensure_dynamic_smem_attr(kernel, smem_bytes)
        kernel(launch.grid, launch.block, args, shared_mem=smem_bytes)
    else:
        kernel(launch.grid, launch.block, args, shared_mem=0)


def _collapse_inert_dims(arr_shape: tuple[int, ...], box_extents: tuple[int, ...]) -> tuple[int, ...]:
    """Reconstruct the materializer's gap-singleton drop from runtime info.

    The materializer drops gap source dims that are extent-1 singletons
    with literal-0 origin coords (a literal-0 origin can only arise for
    a singleton arr dim, since otherwise IR construction would have
    emitted a ``Var`` or expression). At runtime we don't carry that
    decision explicitly — instead we walk ``arr_shape`` and
    ``box_extents`` innermost-first and drop any arr dim of extent 1
    that lines up with a box dim of extent > 1. Leading singletons
    pair with their (kept) box==1 entry and stay; gap singletons fall
    out exactly where the materializer dropped them.

    The materializer's swizzle-split path may emit a rank-(N+1) box on
    a rank-N source by splitting an inner dim. Reinterpret the array's
    last dim as the matching split before walking, so the rank-match
    check below succeeds and ``encode_tiled`` sees a consistent view."""
    arr_rev = list(reversed(arr_shape))
    box_rev = list(reversed(box_extents))
    if len(box_rev) == len(arr_rev) + 1 and arr_rev and box_rev[0] != 0 and arr_rev[0] % box_rev[0] == 0:
        arr_rev = [box_rev[0], arr_rev[0] // box_rev[0], *arr_rev[1:]]
    kept: list[int] = []
    bi = 0
    for a in arr_rev:
        if bi < len(box_rev) and a == 1 and box_rev[bi] != 1:
            continue  # dropped gap singleton
        kept.append(a)
        bi += 1
    if bi != len(box_rev) or len(kept) != len(box_rev):
        raise ValueError(f"TMA descriptor rank mismatch: arr_shape={arr_shape!r} cannot be collapsed to match box_extents={box_extents!r}")
    return tuple(reversed(kept))


def _prebuild_descriptors(compiled: _Compiled, arrays: dict[str, cp.ndarray]) -> dict[int, dict[str, cp.ndarray]]:
    """Encode every TMA ``CUtensorMap`` for ``compiled`` up-front.

    The kernel signature takes ``const CUtensorMap*`` (not a by-value
    ``__grid_constant__`` parameter) because cupy's arg-packing doesn't
    guarantee the 64-byte alignment required for by-value descriptors.
    Placing the descriptor in device memory and passing a pointer
    sidesteps the alignment concern — the TMA load PTX dereferences
    via a generic 64-bit pointer either way.

    Why eagerly: ``cp.asarray(np.frombuffer(...))`` queues an H2D copy on
    the current stream. Building descriptors lazily inside ``_launch``
    means each fresh kernel's H2D races against in-flight TMA loads from
    *previous* launches sharing the same descriptor allocator slab —
    the next allocation can land on cupy-pool memory the prior kernel's
    cp.async.bulk.tensor is still reading, corrupting the descriptor
    and deadlocking the wait. Pre-building once after ``_allocate``
    removes the race entirely; the returned dict is held alive for the
    whole program lifetime, so cupy never reclaims the slab."""
    import cupy as cp

    out: dict[int, dict[str, cp.ndarray]] = {}
    for li, launch in enumerate(compiled.launches):
        if not launch.tma_descriptors:
            continue
        per_launch: dict[str, cp.ndarray] = {}
        for desc in launch.tma_descriptors:
            arr = arrays[desc.src_buf]
            src_shape = _collapse_inert_dims(tuple(int(d) for d in arr.shape), desc.box_extents)
            desc_bytes = _tma.encode_tiled(
                global_address=int(arr.data.ptr),
                src_shape=src_shape,
                box_extents=desc.box_extents,
                elem_size=int(arr.itemsize),
                swizzle=desc.swizzle,
            )
            per_launch[desc.name] = cp.asarray(np.frombuffer(desc_bytes, dtype=np.uint64))
        out[li] = per_launch
    if out:
        cp.cuda.runtime.deviceSynchronize()
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_program(graph: Graph, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
    """Run the lowered graph once, return output ndarrays + total time."""
    import cupy as cp

    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)
    descs = _prebuild_descriptors(compiled, arrays)

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for li, launch in enumerate(compiled.launches):
        _launch(launch, compiled, arrays, descs.get(li))
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
    descs = _prebuild_descriptors(compiled, arrays)

    input_names = {b.name for b in compiled.bufs if b.role == "input"}
    per_launch_np: dict[int, dict[str, np.ndarray]] = {}

    for li, launch in enumerate(compiled.launches):
        _launch(launch, compiled, arrays, descs.get(li))
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


_AUTO_BUDGET_MS = 100.0
_AUTO_MIN_ITERS = 10
_AUTO_MAX_ITERS = 100_000


def benchmark_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    on_iter=None,
) -> BenchmarkResult:
    """Time the graph's launches with per-kernel CUDA events.

    ``num_iters`` accepts an explicit count or the string ``"auto"``. In
    auto mode the loop runs at least ``_AUTO_MIN_ITERS`` iterations and
    keeps going until the accumulated *GPU* time across all launches
    (sum of per-kernel event deltas) reaches ``_AUTO_BUDGET_MS``, capped
    at ``_AUTO_MAX_ITERS``. This adapts the iter count to the kernel
    cost: a 7-µs RMSNorm gets ~14k iters at 100 ms budget, a 1-ms matmul
    gets ~100. The result's ``time_ms`` is the mean of the actually-run
    iterations.

    ``on_iter``, if provided, is a no-arg callable invoked **before**
    each iteration of both the warmup and the measured loop. Used by
    the perf suite to interleave a PyTorch reference run with the
    Deplodock launches so both backends see the same warm GPU state
    (same clocks, same caches) — comparing one-after-the-other can
    give the second-run backend a thermal/cache advantage worth tens
    of percent on small kernels. The callable is responsible for any
    timing of its own work."""
    import cupy as cp

    if isinstance(num_iters, str):
        if num_iters != "auto":
            raise ValueError(f"num_iters must be int or 'auto', got {num_iters!r}")
        target_total_ms = _AUTO_BUDGET_MS
        min_iters = _AUTO_MIN_ITERS
        max_iters = _AUTO_MAX_ITERS
        auto = True
    else:
        target_total_ms = float("inf")
        min_iters = int(num_iters)
        max_iters = int(num_iters)
        auto = False

    compiled = _compile(graph)
    arrays = _allocate(compiled, input_data)
    descs = _prebuild_descriptors(compiled, arrays)

    for _ in range(warmup):
        if on_iter is not None:
            on_iter()
        for li, launch in enumerate(compiled.launches):
            _launch(launch, compiled, arrays, descs.get(li))
    cp.cuda.runtime.deviceSynchronize()

    n = len(compiled.launches)
    starts = [cp.cuda.Event() for _ in range(n)]
    stops = [cp.cuda.Event() for _ in range(n)]
    acc = [0.0] * n

    iters_run = 0
    cumulative_gpu_ms = 0.0
    while iters_run < max_iters:
        if on_iter is not None:
            on_iter()
        for i, launch in enumerate(compiled.launches):
            starts[i].record()
            _launch(launch, compiled, arrays, descs.get(i))
            stops[i].record()
            # Per-kernel sync to make per-launch attribution accurate.
            # Without it, back-to-back launches let one kernel's stop
            # event slide into a downstream kernel's scheduling window,
            # which ends up attributing 0.5-0.8 ms of phantom time to
            # whichever sub-100µs kernel happens to land at a stream-
            # stall position. The added sync overhead is negligible vs
            # the kernels' work and only affects the benchmark window,
            # not what ``time_ms`` reports (which is the sum of per-
            # kernel events — pure GPU work, no sync overhead).
            stops[i].synchronize()
        iter_total = 0.0
        for i in range(n):
            dt = cp.cuda.get_elapsed_time(starts[i], stops[i])
            acc[i] += dt
            iter_total += dt
        cumulative_gpu_ms += iter_total
        iters_run += 1
        if auto and iters_run >= min_iters and cumulative_gpu_ms >= target_total_ms:
            break

    iters = max(iters_run, 1)
    per_launch = [LaunchTime(idx=i, kernel_name=compiled.launches[i].kernel_name, time_ms=acc[i] / iters) for i in range(n)]
    return BenchmarkResult(
        time_ms=sum(acc) / iters,
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
    descs = _prebuild_descriptors(compiled, arrays)

    def run_once() -> None:
        for li, launch in enumerate(compiled.launches):
            _launch(launch, compiled, arrays, descs.get(li))

    return run_once
