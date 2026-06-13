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

import asyncio
import logging
import math
import os as _os
import pickle
import sys as _sys
import time as _time_module
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from deplodock.compiler.backend import BenchmarkResult, LaunchTime, RunResult
from deplodock.compiler.backend.cuda import _tma, nvcc
from deplodock.compiler.backend.cuda.dtype import cupy_dtype
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import DataType
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
    # ``Dim``-valued shape. Static dims carry a ``Literal`` expr, atomic
    # symbolic carry a ``Var``, composite (e.g. ``S * 2`` from a reshape
    # or cat) carry a ``BinaryExpr``. ``resolve_shape`` calls ``expr.eval``
    # to turn any of those into a runtime int.
    shape: tuple[Dim, ...]
    dtype: DataType
    role: str  # "input" | "constant" | "output" | "scratch"

    @property
    def is_symbolic(self) -> bool:
        return any(not d.is_static for d in self.shape)

    def resolve_shape(self, sym_values: dict[str, int]) -> tuple[int, ...]:
        return tuple(int(d.expr.eval(sym_values)) for d in self.shape)


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
        bufs.append(_Buffer(name=nid, shape=tuple(node.output.shape), dtype=node.output.dtype, role=role))
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
    # ``grid`` / ``block`` are per-dim ``GridDimSpec`` tuples whose factors may
    # include symbolic axis names — resolved at launch time via the
    # parent ``_Compiled.symbolic_bindings`` env. Static kernels collapse
    # to single-int factors (``((128,), (1,), (1,))``).
    grid: tuple
    block: tuple
    smem_bytes: int
    zero_outputs: tuple[str, ...]
    tma_descriptors: tuple[TmaDescMeta, ...] = ()
    runtime_args: tuple[str, ...] = ()


@dataclass
class _Compiled:
    bufs: list[_Buffer]
    buf_by_name: dict[str, _Buffer]
    constants: dict[str, float]
    kernels: dict[str, cp.RawKernel]  # kernel_name → RawKernel
    launches: list[_Launch]
    # Symbolic axis name → (input_buf_name, dim_index). Resolved from input
    # array shapes at run-time; empty when the graph has no symbolic dims.
    symbolic_bindings: dict[str, tuple[str, int]] = field(default_factory=dict)
    # Per-symbolic-name buffers whose shape carries that name — used to
    # reshape ``input_data`` to the actual runtime shape before upload.
    symbolic_buf_shape: dict[str, tuple] = field(default_factory=dict)
    # Symbolic axis name → its ``Dim`` hint (default expected size). Used as a
    # fallback concrete value when no ``input_data`` is supplied (the autotuner
    # benches a symbolic graph at the hint size).
    symbolic_hints: dict[str, int] = field(default_factory=dict)


def _compile(graph: Graph) -> _Compiled:
    bufs = _buffers(graph)
    buf_by_name = {b.name: b for b in bufs}
    constants = _constant_values(graph)
    launches_nodes = _launches(graph)
    symbolic_bindings = _symbolic_bindings(graph)
    symbolic_hints = _symbolic_hints(graph)

    # ``nvcc.load_function`` returns a cupy ``Function`` (or a ``RawKernel`` on
    # NVRTC fallback) — both launch-callable and smem-attr settable.
    kernels: dict[str, object] = {}
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
            uses_tma = bool(op.tma_descriptors)
            options = _nvrtc_options(uses_tma=uses_tma)
            # Compile via offline nvcc (faster, cache-warmable) with an NVRTC
            # fallback; returns a Function usable exactly like a RawKernel.
            kernels[kname] = nvcc.load_function(op.kernel_source, kname, options, uses_tma=uses_tma)
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
                runtime_args=tuple(getattr(op, "runtime_args", ())),
            )
        )
    return _Compiled(
        bufs=bufs,
        buf_by_name=buf_by_name,
        constants=constants,
        kernels=kernels,
        launches=launches,
        symbolic_bindings=symbolic_bindings,
        symbolic_hints=symbolic_hints,
    )


def _symbolic_bindings(graph: Graph) -> dict[str, tuple[str, int]]:
    """Walk graph inputs to map every symbolic dim name to its source
    ``(input_buf, dim_index)`` — the launch resolver reads the runtime
    value from ``input_arrays[buf].shape[dim_index]``. First-seen position
    wins on conflicts so each name resolves deterministically.

    Input dims are atomic ``Var``-backed when symbolic (composite Dim exprs
    appear only on derived tensors). We collect via ``expr.free_vars()`` so
    each free name binds to the first input axis where it appears."""
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

    bindings: dict[str, tuple[str, int]] = {}
    for nid in graph.inputs:
        for d, dim in enumerate(graph.nodes[nid].output.shape):
            if dim.is_static:
                continue
            if not isinstance(dim.expr, Var):
                raise ValueError(
                    f"input {nid!r} axis {d} has a composite symbolic dim {dim!r}; "
                    "inputs must carry atomic Var-backed symbolic dims so the launch "
                    "resolver can recover each name from a single shape axis"
                )
            bindings.setdefault(dim.expr.name, (nid, d))
    return bindings


def _symbolic_hints(graph: Graph) -> dict[str, int]:
    """Map every symbolic input-dim name to its ``Dim`` hint (default expected
    size). Read straight off the input ``Dim`` — convenient here since this
    walks the same input shapes as ``_symbolic_bindings``. Used by
    ``_resolve_symbolic`` as the fallback bench size when no ``input_data`` is
    supplied (the autotuner case)."""
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

    hints: dict[str, int] = {}
    for nid in graph.inputs:
        for dim in graph.nodes[nid].output.shape:
            if isinstance(dim.expr, Var) and dim.hint is not None:
                hints.setdefault(dim.expr.name, dim.hint)
    return hints


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


def _materialize(buf: _Buffer, shape: tuple[int, ...], src: np.ndarray | None, constants: dict[str, float]) -> cp.ndarray:
    """Build one device array for ``buf`` at ``shape`` — the single fill
    policy shared by :func:`_allocate` and :meth:`CompiledProgram.rebind`."""
    import cupy as cp

    cp_dtype = cupy_dtype(buf.dtype)
    np_dtype = buf.dtype.np
    if src is not None:
        return cp.asarray(np.ascontiguousarray(src, dtype=np_dtype).reshape(shape))
    if buf.role == "constant" and buf.name in constants:
        return cp.full(shape, float(constants[buf.name]), dtype=cp_dtype)
    if buf.role == "input":
        # Pseudo-random fill for un-supplied inputs (matches old generated program).
        n = 1
        for d in shape:
            n *= int(d)
        # Build the index ramp in int64, not ``np_dtype``: a float16 buffer
        # past 65504 elements would overflow to ``inf`` (then ``inf % 101``
        # → ``nan``). Compute in fp32 and cast the final values — always in
        # ``[-0.5, 0.5]``, so fp16-safe.
        idx = np.arange(n, dtype=np.int64)
        vals = (0.01 * ((idx.astype(np.float32) * 7 + 13) % 101 - 50)).astype(np_dtype)
        return cp.asarray(vals.reshape(shape))
    return cp.zeros(shape, dtype=cp_dtype)


def _allocate(compiled: _Compiled, input_data: dict[str, np.ndarray] | None) -> dict[str, cp.ndarray]:
    input_data = input_data or {}
    sym_values = _resolve_symbolic(compiled, input_data)
    arrays: dict[str, cp.ndarray] = {}
    # Saturating casts here are intended, not bugs: e.g. an SDPA mask-fill
    # constant (``-1e9``) is meant to become ``-inf`` in fp16 (masked → 0 after
    # softmax). Ignore the over/invalid warnings so genuine output stays clean.
    with np.errstate(over="ignore", invalid="ignore"):
        for buf in compiled.bufs:
            shape = buf.resolve_shape(sym_values) or (1,)
            arrays[buf.name] = _materialize(buf, shape, input_data.get(buf.name), compiled.constants)
    return arrays


def _resolve_symbolic(compiled: _Compiled, input_data: dict[str, np.ndarray]) -> dict[str, int]:
    """Bind every symbolic axis name to a concrete ``int``. Reads the runtime
    value from the supplied input array shape (``compiled.symbolic_bindings``
    says which input + dim each name reads from). When no array is supplied for
    that input — the autotuner benches without real inputs — falls back to the
    ``Dim`` hint so the graph runs at its expected (tuned) size."""
    env: dict[str, int] = {}
    for name, (buf, dim_idx) in compiled.symbolic_bindings.items():
        arr = input_data.get(buf)
        if arr is not None:
            env[name] = int(arr.shape[dim_idx])
        elif name in compiled.symbolic_hints:
            env[name] = compiled.symbolic_hints[name]
        else:
            raise ValueError(
                f"symbolic dim {name!r} reads from input {buf!r}.shape[{dim_idx}] but no array was supplied and the dim carries no hint"
            )
    return env


def _launch(
    launch: _Launch,
    compiled: _Compiled,
    arrays: dict[str, cp.ndarray],
    desc_args: dict[str, cp.ndarray] | None = None,
    sym_values: dict[str, int] | None = None,
) -> None:
    from deplodock.compiler.ir.cuda.ir import resolve_dim  # noqa: PLC0415

    for zname in launch.zero_outputs:
        arrays[zname].fill(0)
    kernel = compiled.kernels[launch.kernel_name]
    desc_args = desc_args or {}
    sym_values = sym_values or {}
    args = tuple(desc_args.get(name) if name in desc_args else arrays[name] for name in launch.arg_names)
    # Symbolic axes appear as ``int`` kernel params after buffers + TMA
    # descriptors — append their resolved values to the arg pack.
    if launch.runtime_args:
        args = (*args, *(sym_values[name] for name in launch.runtime_args))
    grid = tuple(resolve_dim(spec, sym_values) for spec in launch.grid)
    block = tuple(resolve_dim(spec, sym_values) for spec in launch.block)
    # Kernels whose Smem footprint exceeds the 48 KB static cap declare
    # an ``extern __shared__`` pool; the launch supplies the byte size
    # via ``shared_mem=`` and (for footprints above 48 KB) opts into the
    # device's larger dynamic-smem allowance via ``cudaFuncSetAttribute``.
    smem_bytes = launch.smem_bytes
    if smem_bytes > _STATIC_SMEM_CAP:
        _ensure_dynamic_smem_attr(kernel, smem_bytes)
        kernel(grid, block, args, shared_mem=smem_bytes)
    else:
        kernel(grid, block, args, shared_mem=0)


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
# Iter-loop policy constants + per-event watchdog
# ---------------------------------------------------------------------------


# Per-launch wall-clock cap. Any single kernel launch exceeding this is
# considered "broken" — too many threads, infinite loop, hung GPU — and
# we bail out via ``HungKernelError`` so the autotune sweep doesn't stall
# on one bad variant.
_KERNEL_TIMEOUT_MS = 1000.0


class HungKernelError(RuntimeError):
    """A kernel launch did not complete within the per-launch watchdog window.

    Distinct from a plain ``RuntimeError`` (a slow-but-completing variant) because a hung
    kernel stays **resident on the device** after we give up polling for it — the in-process
    bench has no way to evict it (only the SIGKILL-isolated tuning worker can reset the
    device). A caller that runs further benches on the same device after catching this must
    treat the device as poisoned and stop, or the next blocking ``synchronize()`` (e.g. the
    torch peer-bench) will block behind the still-running kernel. Subclasses ``RuntimeError``
    so existing ``except RuntimeError`` handlers (the autotune sweep) keep marking the variant
    ``bench_fail`` unchanged."""


class GraphCaptureError(RuntimeError):
    """CUDA graph capture of the bench launch loop failed.

    Raised by :meth:`CompiledProgram.capture_launch_graphs` after draining any
    in-progress capture state, so the stream is clean and the caller can simply
    retry the bench uncaptured. Only the per-kernel reproducer bench enables
    capture (the autotune sweep never does), so this can't be misclassified as
    a ``bench_fail`` there."""


_AUTO_BUDGET_MS = 100.0
# Iter-count cap on ``num_iters="auto"``. Combined with the GPU-time
# target above: whichever fires first wins. The cap is the binding
# constraint for fast kernels (sub-ms / launch, where 100 ms target
# would otherwise mean 100s of iters and the corresponding atomic /
# clock-state pressure on heavy-fanout K-split kernels); the GPU-time
# target is the binding constraint for slow kernels (>= 1 ms / launch,
# where 100 iters would over-measure relative to confidence needs).
_AUTO_MAX_ITERS = 100
# Target per-kernel-position timing window. Sub-millisecond kernels are
# dominated by per-iter Python/cupy framing overhead (~100 µs); we
# amortize it by repeating each launch ``batch_size`` times inside one
# CUDA event window, where ``batch_size = ceil(_BATCH_TARGET_MS /
# per_launch_ms)``. Calibrated after warmup from the last-warmup iter's
# per-launch timings, then held fixed during measurement.
_BATCH_TARGET_MS = 1.0
# Minimum total GPU time the warmup window should cover. sm_120 (and
# other consumer GPUs with auto-boost) take several ms to ramp clocks
# from idle. For tiny kernels the requested ``warmup`` iters may sum
# to << 1 ms — the first measured iters then see mid-ramp clocks and
# the median jitters across runs. After the post-warmup batch-size
# calibration we extend ``warmup`` so total warmup GPU time clears
# this threshold.
_WARMUP_TARGET_MS = 10.0


def _wait_for_event(event, timeout_ms: float, label: str) -> None:
    """Block until ``event`` completes, polling rather than calling the
    blocking ``synchronize()``. Raises ``HungKernelError`` on timeout —
    necessary because once a CUDA kernel is hung, ``synchronize()``
    blocks indefinitely (the driver only resets after minutes), which
    stalls the autotune sweep on a single bad variant.

    Caveat: a hung kernel is still queued on the device after we give
    up here, so the *next* launch queues behind it and may also be
    slow. That's still vastly better than blocking forever in this
    one bench."""
    import time as _time

    import cupy as _cp  # noqa: PLC0415

    start = _time.perf_counter()
    deadline = start + timeout_ms / 1000.0
    next_warn = start + 0.2  # surface kernels stuck >200ms even if they eventually finish
    warned = False
    while not event.done:
        now = _time.perf_counter()
        if now > deadline:
            raise HungKernelError(f"kernel {label!r} did not complete within {timeout_ms:.0f} ms — variant marked bench_fail")
        if now > next_warn:
            logger.warning("[cuda] kernel %r still pending after %.2fs (timeout %.1fs)", label, now - start, timeout_ms / 1000.0)
            warned = True
            next_warn = now + 1.0  # subsequent log every 1s while still stuck
        _time.sleep(0.001)
    elapsed = _time.perf_counter() - start
    if warned:
        logger.warning("[cuda] kernel %r completed after %.2fs of waiting", label, elapsed)
    _cp.cuda.runtime.eventSynchronize(event.ptr)  # cheap post-completion sync


# ---------------------------------------------------------------------------
# CompiledProgram: post-compile GPU state + uniform iter loop
# ---------------------------------------------------------------------------


@dataclass
class CompiledProgram:
    """Post-compile GPU state for one graph: kernels, allocated buffers,
    pre-built TMA descriptors.

    Constructed inside ``gpu_lock()`` by the public entry points
    (:func:`run_program`, :func:`run_program_debug`,
    :func:`benchmark_program`) so every CUDA-touching phase — NVRTC
    compile, cupy alloc, descriptor H2D, kernel-launch loop, output
    ``.get()`` — runs with the lock held. Peer xdist workers never
    interleave with us on the device, which previously surfaced as
    small numerical divergence in multi-kernel attention tests when
    the suite ran in parallel.

    All three entry points walk launches through the same
    :meth:`iter_once`. What differs between them — single pass vs
    warmup+measure vs snapshot-every-launch — collapses to which
    optional callbacks they pass."""

    compiled: _Compiled
    arrays: dict[str, cp.ndarray]
    descs: dict[int, dict[str, cp.ndarray]]
    # Per-symbolic-axis runtime ``int`` resolved at ``build`` time from the
    # supplied input shapes — fed straight to ``_launch`` for grid /
    # block resolution and the runtime-arg tail. Empty for fully-static
    # graphs.
    sym_values: dict[str, int] = field(default_factory=dict)
    # Per-launch timing events, lazily created on first ``iter_once``
    # and reused across every subsequent call so multi-iter bench loops
    # don't churn the cupy ``Event`` pool (the pre-unification
    # ``benchmark_program`` allocated events once outside the while
    # loop; thrashing them per iter perturbs the tuner's variant
    # ranking — close-latency siblings get reordered run-to-run, which
    # caused ``test_tuned_variant_matches_reference`` to flake ~30%).
    _starts: list = field(default_factory=list, repr=False)
    _stops: list = field(default_factory=list, repr=False)
    # Per-launch CUDA graphs (one per launch position, each containing that
    # launch's whole batch) captured by :meth:`capture_launch_graphs`. When
    # set, ``iter_once`` replays ``_graphs[i]`` with one host call instead of
    # the ``batch_sizes[i]``-long Python launch loop, so the CUDA event window
    # measures dense GPU work rather than per-launch dispatch gaps. ``None``
    # (the default) keeps the plain launch loop — ``run_program`` /
    # ``run_program_debug`` and the autotune sweep never capture.
    _graphs: list | None = field(default=None, repr=False)
    _graph_batch_sizes: list[int] | None = field(default=None, repr=False)
    # One CUDA graph holding EVERY launch in program order (batch 1 each),
    # captured by :meth:`capture_program_graph` for the whole-program (e2e)
    # timing windows — the deplodock analogue of timing a captured torch
    # forward, so the backend-comparison table is like-for-like.
    _e2e_graph: Any | None = field(default=None, repr=False)
    _e2e_start: Any | None = field(default=None, repr=False)
    _e2e_stop: Any | None = field(default=None, repr=False)

    @classmethod
    def build(
        cls,
        graph: Graph,
        input_data: dict[str, np.ndarray] | None = None,
        *,
        compile_timeout_s: float | None = None,
    ) -> CompiledProgram:
        """Compile ``graph``, allocate every buffer, pre-build TMA
        descriptors. ``compile_timeout_s`` bounds the setup phase at a
        C-call boundary: if NVRTC + alloc + descriptor work overruns,
        raise ``RuntimeError`` before the caller proceeds to launches
        so no in-flight kernels are left queued.

        Caller is expected to hold ``gpu_lock()`` around this call and
        every subsequent method on the returned program."""
        t0 = _time_module.monotonic()
        compiled = _compile(graph)
        sym_values = _resolve_symbolic(compiled, input_data or {})
        arrays = _allocate(compiled, input_data)
        descs = _prebuild_descriptors(compiled, arrays)
        elapsed = _time_module.monotonic() - t0
        if compile_timeout_s is not None and elapsed > compile_timeout_s:
            raise RuntimeError(f"compile stage exceeded {compile_timeout_s:.1f}s budget ({elapsed:.2f}s) — variant marked bench_fail")
        logger.info(
            "[cuda] CompiledProgram.build: %d launch(es) compile+alloc=%.2fs kernels=[%s]",
            len(compiled.launches),
            elapsed,
            ", ".join(f"{li}:{lc.kernel_name}" for li, lc in enumerate(compiled.launches)),
        )
        return cls(compiled=compiled, arrays=arrays, descs=descs, sym_values=sym_values)

    def rebind(self, input_data: dict[str, np.ndarray]) -> None:
        """Re-bind ``input_data`` on an already-built program, re-sizing
        symbolic-shaped buffers to the new runtime dims — the serving path,
        where one compiled dynamic-seq_len program runs request after request.

        Supplied buffers are re-uploaded: in place (``arr.set``) when the
        resolved shape is unchanged, re-allocated otherwise. Un-supplied
        buffers whose shape carries a symbolic dim (scratch/outputs sized by
        seq_len) re-materialize at the new shape under the same fill policy
        as ``build``; static-shaped un-supplied buffers — the weights — keep
        their device arrays untouched (no re-upload). When any array was
        re-allocated, TMA descriptors are rebuilt (they embed device pointers
        and shapes) and captured CUDA graphs are dropped (they bake old
        pointers). Caller must hold ``gpu_lock()``."""
        new_sym = _resolve_symbolic(self.compiled, input_data)
        realloc = False
        with np.errstate(over="ignore", invalid="ignore"):
            for buf in self.compiled.bufs:
                src = input_data.get(buf.name)
                if src is None and not buf.is_symbolic:
                    continue
                shape = buf.resolve_shape(new_sym) or (1,)
                arr = self.arrays[buf.name]
                if tuple(arr.shape) != shape:
                    self.arrays[buf.name] = _materialize(buf, shape, src, self.compiled.constants)
                    realloc = True
                elif src is not None:
                    arr.set(np.ascontiguousarray(src, dtype=buf.dtype.np).reshape(shape))
        self.sym_values = new_sym
        if realloc:
            self.descs = _prebuild_descriptors(self.compiled, self.arrays)
            self._graphs = None
            self._graph_batch_sizes = None
            self._e2e_graph = None

    def run_once(self) -> None:
        """Launch every kernel once in program order with no per-launch event
        record/sync/watchdog — the serving hot path (timing semantics live in
        :meth:`iter_once`). The default stream orders the launches; the
        caller's subsequent ``outputs()`` ``.get()`` synchronizes."""
        for i, launch in enumerate(self.compiled.launches):
            _launch(launch, self.compiled, self.arrays, self.descs.get(i), self.sym_values)

    def capture_launch_graphs(self, batch_sizes: list[int]) -> None:
        """Capture each launch position's batch into one CUDA graph.

        Stream capture is illegal on the legacy default stream, so each batch is
        captured on a temporary side stream; ``iter_once`` then replays the graph
        on the default stream (``Graph.launch`` targets the *current* stream), so
        the existing cupy/torch event interleaving is untouched. The capture
        window holds only ``_launch`` work — output zeroing + kernel launches on
        prebuilt buffers/descriptors — no allocations, no sync, no event records
        (the dynamic-smem attribute is already set by the uncaptured warmup iters
        that always precede capture).

        Safe to call again when batch sizes change (warmup extension re-fires the
        calibration); a no-op when they match the captured ones. Raises
        :class:`GraphCaptureError` on any failure, after draining the capture
        state so the stream isn't left wedged — the caller retries uncaptured."""
        import cupy as cp

        if self._graphs is not None and self._graph_batch_sizes == list(batch_sizes):
            return
        self._graphs = None
        side = cp.cuda.Stream(non_blocking=True)
        graphs = []
        for i, launch in enumerate(self.compiled.launches):
            try:
                with side:
                    side.begin_capture()
                    for _ in range(batch_sizes[i]):
                        _launch(launch, self.compiled, self.arrays, self.descs.get(i), self.sym_values)
                    graphs.append(side.end_capture())
            except Exception as exc:
                if side.is_capturing():
                    try:
                        with side:
                            side.end_capture()  # drain capture state; discard the partial graph
                    except Exception:  # noqa: BLE001, S110 — already raising the original failure
                        pass
                raise GraphCaptureError(f"capture failed for launch {i} ({launch.kernel_name!r}): {exc}") from exc
        # One throwaway replay per graph absorbs graphExec instantiation /
        # upload cost so the first measured iter is clean. Buffers get
        # clobbered, which is fine: accuracy was checked before benching.
        for g in graphs:
            g.launch()
        cp.cuda.runtime.deviceSynchronize()
        self._graphs = graphs
        self._graph_batch_sizes = list(batch_sizes)

    def capture_program_graph(self) -> None:
        """Capture every launch (batch 1, program order) into ONE CUDA graph.

        The whole-program graph backs :meth:`time_program_window` — an event
        window around N replays of the full program, which is the same
        semantics the captured torch closures get in the interleaved bench
        (one graph per forward, replayed back-to-back inside one window).
        Per-launch capture (:meth:`capture_launch_graphs`) can't provide this:
        summing isolated single-kernel windows drops cross-kernel cache
        effects and inter-kernel gaps, so the sum is not an end-to-end time.

        No-op when already captured. Same error contract as
        :meth:`capture_launch_graphs`: raises :class:`GraphCaptureError` after
        draining any partial capture state."""
        import cupy as cp

        if self._e2e_graph is not None:
            return
        side = cp.cuda.Stream(non_blocking=True)
        try:
            with side:
                side.begin_capture()
                for i, launch in enumerate(self.compiled.launches):
                    _launch(launch, self.compiled, self.arrays, self.descs.get(i), self.sym_values)
                graph = side.end_capture()
        except Exception as exc:
            if side.is_capturing():
                try:
                    with side:
                        side.end_capture()  # drain capture state; discard the partial graph
                except Exception:  # noqa: BLE001, S110 — already raising the original failure
                    pass
            raise GraphCaptureError(f"whole-program capture failed: {exc}") from exc
        # Throwaway replay absorbs graphExec instantiation/upload cost.
        graph.launch()
        cp.cuda.runtime.deviceSynchronize()
        self._e2e_graph = graph

    def time_program_window(self, replays: int) -> float:
        """One event window around ``replays`` back-to-back whole-program
        replays of the captured e2e graph; returns per-replay ms. Caller
        must have run :meth:`capture_program_graph` first."""
        import cupy as cp

        if self._e2e_start is None:
            self._e2e_start, self._e2e_stop = cp.cuda.Event(), cp.cuda.Event()
        self._e2e_start.record()
        for _ in range(replays):
            self._e2e_graph.launch()
        self._e2e_stop.record()
        n = max(1, len(self.compiled.launches))
        _wait_for_event(self._e2e_stop, _KERNEL_TIMEOUT_MS * n * replays, "<whole-program e2e window>")
        return cp.cuda.get_elapsed_time(self._e2e_start, self._e2e_stop) / replays

    def iter_once(
        self,
        *,
        batch_sizes: list[int] | None = None,
        pre_iter=None,
        per_launch_hook=None,
    ) -> list[float]:
        """Run every launch once. Returns per-launch wall time in ms,
        already event-synced before return.

        ``batch_sizes[i]`` repeats launch ``i`` ``N`` times inside one
        CUDA event window so per-iter Python/cupy framing overhead
        amortizes across launches when the kernel is faster than the
        framing (a 9 µs kernel measured one iter at a time is mostly
        framing noise). Returned dt is divided by the batch size so
        callers always see per-call ms.

        ``pre_iter(max_batch_size)`` runs once before the launch loop
        and inside the GPU lock the caller is holding — that's where
        ``_bench_interleaved`` issues its peer torch backends so they
        share the same warm GPU state deplodock measures from.

        ``per_launch_hook(i, launch)`` runs after each launch's stop
        event has synced. :func:`run_program_debug` uses it to
        snapshot every non-input buffer after each launch.

        Per-kernel sync (the ``_wait_for_event(stop_i, ...)``) makes
        per-launch attribution accurate — without it, one kernel's
        stop event can slide into a downstream kernel's scheduling
        window and the timing for a sub-100µs kernel ends up
        contaminated by 0.5-0.8 ms of phantom stream-stall time. The
        watchdog also catches hung kernels independently per launch."""
        import cupy as cp

        n = len(self.compiled.launches)
        if batch_sizes is None:
            batch_sizes = [1] * n
        if pre_iter is not None:
            pre_iter(max(batch_sizes))
        if not self._starts:
            self._starts = [cp.cuda.Event() for _ in range(n)]
            self._stops = [cp.cuda.Event() for _ in range(n)]
        starts, stops = self._starts, self._stops
        dts = [0.0] * n
        for i, launch in enumerate(self.compiled.launches):
            b = batch_sizes[i]
            starts[i].record()
            if self._graphs is not None:
                # Captured-graph replay: one host call enqueues the whole
                # batch, so the event window measures dense GPU work. The
                # caller (``benchmark_program``) re-captures whenever the
                # batch sizes change, so ``_graphs[i]`` always matches ``b``.
                self._graphs[i].launch()
            else:
                for _ in range(b):
                    _launch(launch, self.compiled, self.arrays, self.descs.get(i), self.sym_values)
            stops[i].record()
            _wait_for_event(stops[i], _KERNEL_TIMEOUT_MS * b, launch.kernel_name)
            elapsed_ms = cp.cuda.get_elapsed_time(starts[i], stops[i])
            # CUDA event timing has sub-µs resolution and a real launch must
            # consume at least one device cycle — a 0.0 reading means the
            # launch was a no-op (degenerate grid like BM=1×BN=128 with the
            # M tile entirely masked out, or a kernel that was fused into
            # nothing). Pinning a 0µs "win" in the autotune DB would lock
            # that variant in as the unbeatable best across re-runs. Treat
            # as bench_fail instead — the existing worker → parent → DB
            # path then records a normal sentinel row.
            if elapsed_ms <= 0.0:
                raise RuntimeError(
                    f"kernel {launch.kernel_name!r} reported {elapsed_ms:.3f}ms elapsed — "
                    "degenerate / no-op launch, variant marked bench_fail"
                )
            dts[i] = elapsed_ms / b
            if per_launch_hook is not None:
                per_launch_hook(i, launch)
        return dts

    def outputs(self) -> dict[str, np.ndarray]:
        """Copy every output buffer back to host. Caller must hold the
        GPU lock — ``.get()`` is an async D2H copy on the default
        stream, so peer workers' kernels would otherwise interleave
        with our D2H on the shared device."""
        return {b.name: self.arrays[b.name].get() for b in self.compiled.bufs if b.role == "output"}

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy every non-input buffer (scratch + constants + outputs)
        to host. Used by :func:`run_program_debug` to capture every
        intermediate state for per-launch comparison against a
        reference backend."""
        input_names = {b.name for b in self.compiled.bufs if b.role == "input"}
        return {name: arr.get() for name, arr in self.arrays.items() if name not in input_names}


# ---------------------------------------------------------------------------
# Public entry points: thin shells around CompiledProgram
# ---------------------------------------------------------------------------


def run_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    *,
    pre_run=None,
) -> tuple[RunResult, Any]:
    """Run the lowered graph once, return ``(RunResult, pre_run_result)``.

    ``pre_run`` runs once inside the GPU lock, before deplodock's
    kernel launches. Its return value flows through as the tuple's
    second element. Tests use this to compute a torch eager reference
    on the same GPU window the deplodock launches will see, so peer-
    worker CUDA activity can't interleave the eager forward with the
    deplodock comparison."""
    from deplodock.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        prog = CompiledProgram.build(graph, input_data)
        dts = prog.iter_once()
        outputs = prog.outputs()
    return RunResult(outputs=outputs, time_ms=sum(dts)), pre_result


@dataclass
class DebugResult:
    outputs: dict[str, np.ndarray]
    per_launch: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def run_program_debug(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    *,
    pre_run=None,
) -> tuple[DebugResult, Any]:
    """Run the graph once, snapshotting every non-input buffer after
    each launch. Returns ``(DebugResult, pre_run_result)`` — same
    ``pre_run`` semantics as :func:`run_program`."""
    from deplodock.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    per_launch: dict[int, dict[str, np.ndarray]] = {}
    with gpu_lock():
        pre_result = pre_run() if pre_run is not None else None
        prog = CompiledProgram.build(graph, input_data)
        prog.iter_once(per_launch_hook=lambda li, _lc: per_launch.__setitem__(li, prog.snapshot()))
        outputs = prog.outputs()
    return DebugResult(outputs=outputs, per_launch=per_launch), pre_result


def benchmark_program(
    graph: Graph,
    input_data: dict[str, np.ndarray] | None = None,
    warmup: int = 5,
    num_iters: int | str = 20,
    on_iter=None,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Time the graph's launches with per-kernel CUDA events.

    Single loop covers warmup + measurement: the first ``warmup`` iters
    are discarded, the rest are counted toward the result. The
    per-launch ``_KERNEL_TIMEOUT_MS`` watchdog (inside
    :meth:`CompiledProgram.iter_once`) runs every iter — warmup or
    measured — so a single hung kernel raises cleanly instead of
    stalling the whole sweep.

    ``num_iters`` accepts an explicit count or the string ``"auto"``.
    In auto mode the loop accumulates measured GPU time until it
    reaches ``_AUTO_BUDGET_MS`` (capped at ``_AUTO_MAX_ITERS`` measured
    iters). For a 7-µs RMSNorm that's ~14k iters; a 1-ms matmul gets
    ~100. The result's per-launch ``time_ms`` is the *median* of
    measured iters (mean was sensitive to single-iter outliers from
    thermal blips and GPU-lock-contention spikes — the autotune
    ``_pick_best_candidate`` selects on the lowest summed latency, so
    noise-driven dips made it pick variants whose post-tune bench was
    slower than the heuristic). Total ``time_ms`` is the sum of
    per-launch medians.

    ``on_iter(batch_size)`` is invoked once at the top of every iter
    inside the GPU lock — that's where ``_bench_interleaved`` runs
    peer torch backends so they time the same number of back-to-back
    calls deplodock does per CUDA event window, no warm-vs-cold
    asymmetry.

    ``compile_timeout_s`` bounds NVRTC + alloc + descriptor setup;
    raised inside :meth:`CompiledProgram.build`.

    ``run_timeout_s`` bounds the iter loop on **accumulated GPU time**
    (sum of per-launch CUDA-event measurements), not wall-clock — so
    Python/cupy framing overhead doesn't shrink the budget for tiny
    ops. Catches the gap left by the per-launch ``_KERNEL_TIMEOUT_MS``
    watchdog: a variant where every launch fits under the watchdog but
    summed across iters exceeds the budget (e.g. 999 ms × N iters).
    Checked between iters so no in-flight launch is mid-kernel when
    the function raises.

    ``capture_graphs`` (default on) captures each launch's batch into a CUDA
    graph (:meth:`CompiledProgram.capture_launch_graphs`) once batch sizes are
    calibrated, so the event windows measure dense GPU work instead of
    per-launch dispatch gaps. Warmup iters always run uncaptured, which keeps
    the zero-elapsed degenerate-launch guard and the hung-kernel watchdog
    probing real launches before any graph is built. A capture failure is
    non-fatal: the bench logs a warning and continues uncaptured, reporting it
    via ``BenchmarkResult.captured`` — callers pairing this result with peer
    torch timings (``bench_lowered_vs_torch`` / the e2e comparison) use that
    flag to re-run all-or-nothing so one table never mixes semantics, and the
    tune sweep persists it on each ``perf`` row (captured measurements
    supersede wall-semantics ones on write — see ``SearchDB.record_perf``).

    Multi-launch programs additionally get a WHOLE-program time per measured
    iter — one event window around back-to-back replays of a single CUDA
    graph holding every launch in program order
    (:meth:`CompiledProgram.time_program_window`) — reported as
    ``BenchmarkResult.e2e_ms`` / ``e2e_min_ms``. The per-launch windows each
    replay one kernel solo, so their sum misses cross-kernel cache effects;
    only the whole-program window is comparable against a captured torch
    forward. Automatic when capture holds and the program has more than one
    launch (for a single launch the solo window IS the program time, so the
    fields stay ``None`` and nothing is measured twice — the common case for
    the autotune sweep's single-node slices); also ``None`` when capture is
    off or fell back."""
    from deplodock.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415

    target_total_ms, max_measured, auto = _resolve_iter_budget(num_iters)

    with gpu_lock():
        prog = CompiledProgram.build(graph, input_data, compile_timeout_s=compile_timeout_s)
        n = len(prog.compiled.launches)
        batch_sizes = [1] * n
        # Per-launch sample list — kept around to compute the median
        # across measured iters (more robust than the arithmetic mean
        # against thermal blips, GPU-lock-contention spikes, and other
        # one-off outliers the autotune's variant ranking previously
        # got confused by; see ``project_..._noise`` write-ups).
        samples: list[list[float]] = [[] for _ in range(n)]
        measure_e2e = n > 1  # single launch: the solo window IS the program time
        e2e_samples: list[float] = []
        e2e_replays = 0  # calibrated lazily on the first measured iter
        iters_run = 0
        measured = 0
        cumulative_gpu_ms = 0.0  # measured-iter GPU time, for the "auto" stop target
        total_gpu_ms = 0.0  # all-iter GPU time (incl. warmup), for the run-stage budget

        def _try_capture(sizes: list[int]) -> bool:
            """Best-effort capture; a failure logs + continues uncaptured."""
            try:
                prog.capture_launch_graphs(sizes)
            except GraphCaptureError as exc:
                logger.warning("[cuda] %s — continuing with uncaptured (dispatch-inclusive) timing", exc)
                return False
            return True

        if capture_graphs and warmup == 0:
            # No warmup → the calibration below never fires; capture the
            # uncalibrated all-1 batches so measurement is still dense.
            capture_graphs = _try_capture(batch_sizes)
        while True:
            iter_dts = prog.iter_once(batch_sizes=batch_sizes, pre_iter=on_iter)
            iters_run += 1
            total_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            # GPU-time run budget: bail if the cumulative GPU time
            # across all iters (warmup + measured) exceeds
            # ``run_timeout_s``. Catches the "every launch is just
            # under the per-launch watchdog" pathology. Counts warmup
            # iters too so a slow kernel can't hide behind warmup
            # discards.
            if run_timeout_s is not None and total_gpu_ms > run_timeout_s * 1000.0:
                raise RuntimeError(f"benchmark run stage exceeded {run_timeout_s:.1f}s of GPU time — variant marked bench_fail")
            if iters_run == warmup:
                batch_sizes = _calibrate_batch_sizes(iter_dts)
                if capture_graphs:
                    # Capture (or re-capture) at the calibrated batch sizes.
                    # The warmup extension below can re-fire this calibration
                    # branch with new batch sizes — ``capture_launch_graphs``
                    # no-ops when they're unchanged and re-captures when not,
                    # so graphs and batches never go out of sync.
                    capture_graphs = _try_capture(batch_sizes)
                # Extend warmup until total warmup GPU time clears the
                # clock-ramp floor. Post-batching, each subsequent
                # warmup iter spends roughly
                # ``sum(iter_dts[i] * batch_sizes[i])`` of GPU time —
                # use the just-measured per-launch dts to estimate how
                # many extra iters are needed.
                if total_gpu_ms < _WARMUP_TARGET_MS:
                    per_iter_ms = sum(iter_dts[i] * batch_sizes[i] for i in range(n))
                    if per_iter_ms > 0:
                        warmup += int(math.ceil((_WARMUP_TARGET_MS - total_gpu_ms) / per_iter_ms))
            if iters_run <= warmup:
                continue
            # Measured iter: store per-launch sample (already
            # normalized to per-launch ms inside ``iter_once``).
            # Reduced via median at the end so a single outlier iter
            # can't shift the result.
            for i in range(n):
                samples[i].append(iter_dts[i])
            cumulative_gpu_ms += sum(iter_dts[i] * batch_sizes[i] for i in range(n))
            measured += 1
            # Whole-program window, one per measured iter — shares the same
            # warm GPU state as the per-launch windows and the ``on_iter``
            # torch closures. Counted toward the run-stage GPU budget but NOT
            # the auto-stop target, so it never starves per-launch sampling.
            if measure_e2e and capture_graphs:
                if e2e_replays == 0:
                    try:
                        prog.capture_program_graph()
                    except GraphCaptureError as exc:
                        logger.warning("[cuda] %s — skipping whole-program e2e timing", exc)
                        measure_e2e = False
                    else:
                        iter_ms = sum(iter_dts)
                        e2e_replays = max(1, int(round(_BATCH_TARGET_MS / iter_ms))) if 0 < iter_ms < _BATCH_TARGET_MS else 1
                if measure_e2e:
                    e2e_dt = prog.time_program_window(e2e_replays)
                    e2e_samples.append(e2e_dt)
                    total_gpu_ms += e2e_dt * e2e_replays
            if measured >= max_measured:
                break
            if auto and cumulative_gpu_ms >= target_total_ms:
                break

    return _samples_to_result(samples, prog.compiled.launches, captured=capture_graphs, e2e_samples=e2e_samples)


def _resolve_iter_budget(num_iters: int | str) -> tuple[float, int, bool]:
    """Resolve ``num_iters`` to ``(target_total_ms, max_measured, auto)``."""
    if isinstance(num_iters, str):
        if num_iters != "auto":
            raise ValueError(f"num_iters must be int or 'auto', got {num_iters!r}")
        return (_AUTO_BUDGET_MS, _AUTO_MAX_ITERS, True)
    return (float("inf"), int(num_iters), False)


def _calibrate_batch_sizes(iter_dts: list[float]) -> list[int]:
    """Pick per-position batch sizes so each CUDA event window covers
    ~``_BATCH_TARGET_MS`` of GPU time. Per-position 1 when the kernel
    already exceeds the target — no benefit to batching there."""
    return [max(1, int(round(_BATCH_TARGET_MS / dt))) if 0 < dt < _BATCH_TARGET_MS else 1 for dt in iter_dts]


def _samples_to_result(
    samples: list[list[float]], launches: list[_Launch], *, captured: bool = False, e2e_samples: list[float] | None = None
) -> BenchmarkResult:
    """Collapse per-launch sample lists to a ``BenchmarkResult`` keyed
    on the median of each launch's measured iters."""
    import statistics as _stats  # noqa: PLC0415

    n = len(launches)
    medians = [(_stats.median(samples[i]) if samples[i] else 0.0) for i in range(n)]
    mins = [(min(samples[i]) if samples[i] else 0.0) for i in range(n)]
    per_launch = [
        LaunchTime(
            idx=i,
            kernel_name=launches[i].kernel_name,
            time_ms=medians[i],
            samples=tuple(samples[i]) if samples[i] else None,
        )
        for i in range(n)
    ]
    # ``time_ms`` is the per-launch median (stable for tune's ranking); ``min_ms``
    # is the per-launch best-case (least OS/thermal noise — what ``run --bench``
    # reports, matching tune's min-over-variants reporting).
    return BenchmarkResult(
        time_ms=sum(medians),
        min_ms=sum(mins),
        num_launches=n,
        per_launch=per_launch if per_launch else None,
        captured=captured,
        e2e_ms=_stats.median(e2e_samples) if e2e_samples else None,
        e2e_min_ms=min(e2e_samples) if e2e_samples else None,
    )


# ---------------------------------------------------------------------------
# Subprocess-isolated benchmark worker (single async transport)
# ---------------------------------------------------------------------------


class _AsyncBenchWorker:
    """The parent-side transport for the SIGKILL-able ``_bench_worker`` subprocess —
    the **single** isolated-bench transport (the old sync ``_BenchWorker`` is gone).

    Lets the parent enforce a hard wall-clock cap on a bench: if the worker doesn't
    respond within ``wall_timeout_s``, the parent SIGKILLs it. The dirty CUDA stream
    (and any kernels still queued behind a hung launch) dies with the process, so the
    *next* bench starts on a clean device — fixing the "autotune hangs on the variant
    AFTER a bench_fail" pathology. The worker imports cupy lazily on its first
    request, so spawn cost is just Python startup (~0.2 s).

    Drives the ``_bench_worker`` protocol (``<8-byte LE length><pickle>``, both
    directions) over asyncio streams, so one event loop can keep N device-pinned
    workers benching concurrently — the per-kernel multi-GPU autotune path
    (``two_level.inner_reward``). The sync ``benchmark_program_isolated`` /
    ``benchmark_compare_isolated`` entry points are thin ``asyncio.run`` bridges over
    a one-shot instance (deployable ``--bench``); the autotune sweep awaits a
    persistent instance per GPU directly via ``benchmark_program_isolated_async``.

    Pin a worker to a physical GPU with ``device_id``: the spawn env gets
    ``CUDA_VISIBLE_DEVICES=<id>`` (so the child's logical device 0 *is* that
    GPU — every argumentless ``cp.cuda.Device()`` in the child resolves
    correctly with no other call-site change) and, when a base
    ``DEPLODOCK_GPU_LOCK`` is set, a per-device lock path so workers on
    different GPUs take distinct ``FileLock``s instead of serialising. The
    env overlay rides the child only — the parent's ``os.environ`` is never
    mutated (it's shared by every slot on the one event-loop thread).

    The wall-clock cap is :func:`asyncio.wait_for`; on overrun the child is
    SIGKILLed and respawned on the next bench."""

    _WORKER_MODULE = "deplodock.compiler.backend.cuda._bench_worker"

    def __init__(self, *, device_id: int | None = None) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._device_id = device_id

    def _child_env(self) -> dict:
        env = dict(_os.environ)
        if self._device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
            from deplodock import config  # noqa: PLC0415

            base = config.gpu_lock_path()
            if base:
                # Per-device lock so concurrent device-pinned workers don't
                # serialise on one FileLock (the lock is taken inside the child).
                env["DEPLODOCK_GPU_LOCK"] = f"{base}-{self._device_id}"
        return env

    async def _spawn(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            _sys.executable,
            "-m",
            self._WORKER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_env(),
        )
        logger.info("[bench-worker] spawned (async) pid=%s device=%s", self._proc.pid, self._device_id)

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    def close(self) -> None:
        """Terminate the worker (driver teardown). The subprocess transport is
        reaped when the event loop closes."""
        self._kill()

    async def aclose(self) -> None:
        """Terminate the worker and await its reap — for one-shot bridges that
        spawn + run + tear down within a single ``asyncio.run`` (so no orphaned
        subprocess transport survives the loop)."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass

    async def _read_stderr_tail(self, proc: asyncio.subprocess.Process) -> str:
        try:
            data = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
        except (TimeoutError, Exception):  # noqa: BLE001 — stderr drain is best-effort
            return ""
        return data.decode(errors="replace")[-500:]

    async def run_job(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        """Send one request, read the response within ``wall_timeout_s`` (else SIGKILL
        + raise ``RuntimeError``), and return the unpickled response. A stale-worker
        race on send respawns and retries once; a response-side timeout / EOF is a
        hard error."""
        request = pickle.dumps(request_obj, protocol=pickle.HIGHEST_PROTOCOL)
        frame = len(request).to_bytes(8, "little") + request
        deadline = _time_module.perf_counter() + wall_timeout_s
        for attempt in (0, 1):
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn()
            assert self._proc is not None  # for type narrowing
            proc = self._proc
            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                proc.stdin.write(frame)
                await asyncio.wait_for(proc.stdin.drain(), timeout=remaining)
            except TimeoutError as exc:
                await self.aclose()
                raise RuntimeError(
                    f"bench worker did not accept the request within {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned"
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self.aclose()
                if attempt == 1:
                    raise RuntimeError(f"bench worker died during request send: {exc}") from exc
                logger.info("[bench-worker] stale async worker on send (%s) — respawning", exc)
                continue

            try:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                header = await asyncio.wait_for(proc.stdout.readexactly(8), timeout=remaining)
                n = int.from_bytes(header, "little")
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                body = await asyncio.wait_for(proc.stdout.readexactly(n), timeout=remaining)
            except TimeoutError as exc:
                await self.aclose()
                raise RuntimeError(f"bench worker exceeded {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned") from exc
            except asyncio.IncompleteReadError as exc:
                stderr_tail = await self._read_stderr_tail(proc)
                await self.aclose()
                raise RuntimeError(f"bench worker EOF before response; stderr tail: {stderr_tail}") from exc

            resp = pickle.loads(body)
            if not resp.get("ok"):
                raise RuntimeError(f"bench worker error: {resp.get('error', '?')}")
            return resp
        raise RuntimeError("bench worker unreachable")  # both attempts exhausted (defensive)


async def benchmark_program_isolated_async(
    graph: Graph,
    *,
    worker: _AsyncBenchWorker,
    wall_timeout_s: float,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    nvcc_flags: str | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Async mirror of :func:`benchmark_program_isolated`, benching through a
    caller-supplied device-pinned ``worker`` so one event loop can drive N GPUs
    concurrently. Same in-worker budgets + ``wall_timeout_s`` SIGKILL backstop."""
    resp = await worker.run_job(
        {
            "graph": graph,
            "nvcc_flags": nvcc_flags,
            "torch_spec": None,  # no torch comparison — pure deplodock bench
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
                "capture_graphs": capture_graphs,
            },
        },
        wall_timeout_s=wall_timeout_s,
    )
    return resp["result"]


async def _run_job_oneshot(request_obj: dict, *, wall_timeout_s: float) -> dict:
    """Spawn a fresh (unpinned) ``_AsyncBenchWorker``, run one job, tear it down.
    The transport for the synchronous one-shot bridges below — they each wrap this
    in ``asyncio.run`` (the worker's streams bind to the loop, so it can't persist
    across ``asyncio.run`` calls; the per-call ~0.2 s spawn is negligible against a
    deployable ``--bench``)."""
    worker = _AsyncBenchWorker()
    try:
        return await worker.run_job(request_obj, wall_timeout_s=wall_timeout_s)
    finally:
        await worker.aclose()


def benchmark_program_isolated(
    graph: Graph,
    *,
    wall_timeout_s: float,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    nvcc_flags: str | None = None,
    capture_graphs: bool = True,
) -> BenchmarkResult:
    """Synchronous wall-time-bounded ``benchmark_program`` — a thin ``asyncio.run``
    bridge over :class:`_AsyncBenchWorker` (the one transport). Runs the bench in a
    SIGKILL-able subprocess; the in-worker ``compile_timeout_s`` / ``run_timeout_s``
    budgets still apply, and ``wall_timeout_s`` is the backstop for a kernel that
    keeps the GPU busy past them (cupy's next ``deviceSynchronize`` would block).

    ``on_iter`` isn't supported here — interleaved benchmarking (``deplodock run
    --bench``) shares a Python process with torch and uses ``benchmark_program``
    directly. The autotune sweep awaits ``benchmark_program_isolated_async`` instead
    (a persistent per-GPU worker); this sync entry serves ``CudaBackend.benchmark``'s
    isolated path."""
    resp = asyncio.run(
        _run_job_oneshot(
            {
                "graph": graph,
                "nvcc_flags": nvcc_flags,
                "torch_spec": None,  # no torch comparison — pure deplodock bench
                "kwargs": {
                    "warmup": warmup,
                    "num_iters": num_iters,
                    "compile_timeout_s": compile_timeout_s,
                    "run_timeout_s": run_timeout_s,
                    "capture_graphs": capture_graphs,
                },
            },
            wall_timeout_s=wall_timeout_s,
        )
    )
    return resp["result"]


def benchmark_compare_isolated(
    *,
    lowered: Graph,
    torch_spec: tuple,
    bench_backends: str,
    wall_timeout_s: float,
    warmup: int,
    iters: int,
    seed: int,
    nvcc_flags: str | None = None,
) -> tuple:
    """Run the deployable eager / torch.compile / deplodock comparison in the
    SIGKILL-able worker — a synchronous ``asyncio.run`` bridge over
    :class:`_AsyncBenchWorker` (the one transport).

    Unlike the deplodock-only autotune bench, the deployable comparison interleaves deplodock with
    real torch in one process and so couldn't be isolated before — a hung generated kernel wedged
    the whole run. This ships the *entire* comparison to the worker: a hung kernel hangs the child,
    which the parent SIGKILLs on ``wall_timeout_s``, freeing the device and leaving the parent clean.

    ``torch_spec`` rebuilds the torch side **in the child** (no live module crosses the pipe), reusing
    the same core functions the in-process path uses:

    - ``("trace_args", {code, input, layer, seq_len, dynamic})`` → ``load_or_trace`` rebuilds the real
      module (an HF model id or a ``--code`` expression), benched via ``bench_full_model_real``.
    - ``("frontend_graph", Graph | None)`` → ``bench_lowered_vs_torch`` (per-kernel reproducer; ``None``
      benches deplodock-only when the graph isn't torch-runnable).

    Returns ``(results, bench, torch_available, captured)`` — the shape ``bench_lowered_vs_torch``
    returns (``captured``: all backends were timed under CUDA graph capture; False means the
    all-or-nothing fallback ran and the timings include host dispatch)."""
    resp = asyncio.run(
        _run_job_oneshot(
            {
                "graph": lowered,
                "nvcc_flags": nvcc_flags,
                "torch_spec": torch_spec,
                "bench_backends": bench_backends,
                "warmup": warmup,
                "iters": iters,
                "seed": seed,
            },
            wall_timeout_s=wall_timeout_s,
        )
    )
    return resp["results"], resp["result"], resp["torch_available"], resp.get("captured", False)
