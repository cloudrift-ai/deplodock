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
import math
import os as _os
import pickle
import select
import subprocess
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


def _allocate(compiled: _Compiled, input_data: dict[str, np.ndarray] | None) -> dict[str, cp.ndarray]:
    import cupy as cp

    input_data = input_data or {}
    sym_values = _resolve_symbolic(compiled, input_data)
    arrays: dict[str, cp.ndarray] = {}
    # Saturating casts here are intended, not bugs: e.g. an SDPA mask-fill
    # constant (``-1e9``) is meant to become ``-inf`` in fp16 (masked → 0 after
    # softmax). Ignore the over/invalid warnings so genuine output stays clean.
    with np.errstate(over="ignore", invalid="ignore"):
        for buf in compiled.bufs:
            resolved = buf.resolve_shape(sym_values)
            shape = resolved or (1,)
            cp_dtype = cupy_dtype(buf.dtype)
            np_dtype = buf.dtype.np
            src = input_data.get(buf.name)
            if src is not None:
                arr = cp.asarray(np.ascontiguousarray(src, dtype=np_dtype).reshape(shape))
            elif buf.role == "constant" and buf.name in compiled.constants:
                arr = cp.full(shape, float(compiled.constants[buf.name]), dtype=cp_dtype)
            elif buf.role == "input":
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
                arr = cp.asarray(vals.reshape(shape))
            else:
                arr = cp.zeros(shape, dtype=cp_dtype)
            arrays[buf.name] = arr
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
    the function raises."""
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
        iters_run = 0
        measured = 0
        cumulative_gpu_ms = 0.0  # measured-iter GPU time, for the "auto" stop target
        total_gpu_ms = 0.0  # all-iter GPU time (incl. warmup), for the run-stage budget
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
            if measured >= max_measured:
                break
            if auto and cumulative_gpu_ms >= target_total_ms:
                break

    return _samples_to_result(samples, prog.compiled.launches)


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


def _samples_to_result(samples: list[list[float]], launches: list[_Launch]) -> BenchmarkResult:
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
    return BenchmarkResult(time_ms=sum(medians), min_ms=sum(mins), num_launches=n, per_launch=per_launch if per_launch else None)


# ---------------------------------------------------------------------------
# Subprocess-isolated benchmark worker
# ---------------------------------------------------------------------------


class _BenchWorker:
    """Long-lived ``deplodock.compiler.backend.cuda._bench_worker`` subprocess.

    Lets the parent enforce a hard wall-clock cap on a bench: if the worker
    doesn't respond within ``wall_timeout_s``, the parent SIGKILLs it. The
    dirty CUDA stream (and any kernels still queued behind a hung launch)
    dies with the process, so the *next* bench starts on a clean device —
    fixes the "autotune hangs on the variant AFTER a bench_fail" pathology
    (see ``Pipeline._bench_terminal``).

    Lifecycle: one worker per ``CudaBackend`` instance, lazily spawned on
    the first ``bench`` call and respawned on timeout / EOF / error. The
    worker imports cupy lazily on its first request — until then no CUDA
    context is initialized in the worker, so the spawn cost is just Python
    startup (~0.2 s).
    """

    _WORKER_MODULE = "deplodock.compiler.backend.cuda._bench_worker"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None  # noqa: UP007 — lazy-init sentinel

    def _spawn(self) -> None:
        import sys as _sys  # noqa: PLC0415

        self._proc = subprocess.Popen(  # noqa: S603
            [_sys.executable, "-m", self._WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(_os.environ),
            bufsize=0,
        )
        logger.info("[bench-worker] spawned pid=%s", self._proc.pid)

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                # Process is in uninterruptible-kernel state (rare —
                # NVRTC stuck in a driver call). Nothing more we can do;
                # init will reap once the syscall returns. Release the
                # Python handle so the next bench respawns.
                logger.warning("[bench-worker] pid=%s did not reap within 2s of SIGKILL", self._proc.pid)
        except ProcessLookupError:
            pass
        self._proc = None

    def __del__(self) -> None:
        self._kill()

    def run_job(self, request_obj: dict, *, wall_timeout_s: float) -> dict:
        """Send one length-prefixed pickle request, read the response within ``wall_timeout_s``
        (else SIGKILL the worker and raise ``RuntimeError``), and return the unpickled response.
        The single transport for both the deplodock-only bench and the deployable comparison — the
        request's ``torch_spec`` decides which (see ``_bench_worker._run_job``)."""
        request = pickle.dumps(request_obj, protocol=pickle.HIGHEST_PROTOCOL)
        # A previous worker may have exited between our last interaction
        # and this request — most commonly through the dirty-context exit
        # path in ``_bench_worker.main``. ``poll()`` has a brief window
        # where it still reports the worker as alive (the kernel hasn't
        # reaped yet), so the first stdin write can hit ``BrokenPipeError``
        # even though our state says "worker is up". Treat that as a
        # stale-worker race: respawn and retry once. A second write
        # failure on a fresh worker is a real bug and surfaces.
        for attempt in (0, 1):
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            assert self._proc is not None  # for type narrowing
            proc = self._proc
            try:
                proc.stdin.write(len(request).to_bytes(8, "little"))
                proc.stdin.write(request)
                proc.stdin.flush()
                break
            except (BrokenPipeError, OSError) as exc:
                self._kill()
                if attempt == 1:
                    raise RuntimeError(f"bench worker died during request send: {exc}") from exc
                logger.info("[bench-worker] stale worker on send (%s) — respawning", exc)

        deadline = _time_module.perf_counter() + wall_timeout_s
        out_fd = proc.stdout.fileno()

        def _read_with_deadline(n: int) -> bytes:
            buf = bytearray()
            while len(buf) < n:
                remaining = deadline - _time_module.perf_counter()
                if remaining <= 0:
                    raise _BenchTimeout()
                r, _, _ = select.select([out_fd], [], [], remaining)
                if not r:
                    raise _BenchTimeout()
                chunk = _os.read(out_fd, n - len(buf))
                if not chunk:
                    raise _BenchWorkerEOF()
                buf.extend(chunk)
            return bytes(buf)

        try:
            header = _read_with_deadline(8)
            n = int.from_bytes(header, "little")
            body = _read_with_deadline(n)
        except _BenchTimeout as exc:
            self._kill()
            raise RuntimeError(f"bench worker exceeded {wall_timeout_s:.1f}s wall budget — SIGKILL'd, stream cleaned") from exc
        except _BenchWorkerEOF as exc:
            stderr_tail = b""
            try:
                stderr_tail = proc.stderr.read() or b""
            except Exception:  # noqa: BLE001 — stderr drain is best-effort
                pass
            self._kill()
            raise RuntimeError(f"bench worker EOF before response; stderr tail: {stderr_tail.decode(errors='replace')[-500:]}") from exc

        resp = pickle.loads(body)
        if not resp.get("ok"):
            # Surface the worker-side exception verbatim. ``Pipeline._bench_terminal`` treats a
            # failed deplodock bench as ``bench_fail``; the deployable callers report + continue.
            raise RuntimeError(f"bench worker error: {resp.get('error', '?')}")
        return resp


class _BenchTimeout(Exception):
    """Internal sentinel — converted to RuntimeError at the API boundary."""


class _BenchWorkerEOF(Exception):
    """Internal sentinel — worker died without writing a response."""


# Module-level singleton. Reused across ``benchmark_program_isolated`` calls
# in the same process so we pay the worker startup cost once.
_bench_worker_singleton: _BenchWorker | None = None


def _bench_worker() -> _BenchWorker:
    global _bench_worker_singleton
    if _bench_worker_singleton is None:
        _bench_worker_singleton = _BenchWorker()
    return _bench_worker_singleton


def benchmark_program_isolated(
    graph: Graph,
    *,
    wall_timeout_s: float,
    warmup: int = 5,
    num_iters: int | str = 20,
    compile_timeout_s: float | None = None,
    run_timeout_s: float | None = None,
    nvcc_flags: str | None = None,
) -> BenchmarkResult:
    """Wall-time-bounded ``benchmark_program``. Runs the bench in a
    persistent subprocess; on ``wall_timeout_s`` overrun the worker is
    SIGKILLed and the next call respawns it on a clean device.

    The in-process ``compile_timeout_s`` / ``run_timeout_s`` budgets are
    still enforced inside the worker (they're cheaper and give better
    error messages than a SIGKILL). ``wall_timeout_s`` is the backstop
    for the failure mode they don't cover: a kernel that keeps the GPU
    busy past any per-launch / per-iter budget, which on cupy makes the
    next ``deviceSynchronize`` block indefinitely.

    ``on_iter`` is not supported here — interleaved benchmarking (the
    ``deplodock run --bench`` path that alternates torch eager / compile
    / deplodock) needs the bench to share a Python process with torch
    and must use ``benchmark_program`` directly instead.
    """
    resp = _bench_worker().run_job(
        {
            "graph": graph,
            "nvcc_flags": nvcc_flags,
            "torch_spec": None,  # no torch comparison — pure deplodock bench
            "kwargs": {
                "warmup": warmup,
                "num_iters": num_iters,
                "compile_timeout_s": compile_timeout_s,
                "run_timeout_s": run_timeout_s,
            },
        },
        wall_timeout_s=wall_timeout_s,
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
    """Run the deployable eager / torch.compile / deplodock comparison in the SIGKILL-able worker.

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

    Returns ``(results, bench, torch_available)`` — the shape ``bench_lowered_vs_torch`` returns."""
    resp = _bench_worker().run_job(
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
    return resp["results"], resp["result"], resp["torch_available"]
