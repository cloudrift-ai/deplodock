"""Kernel IR → CUDA source.

Builds a CUDA ``RenderCtx`` (intrinsic + GPU-builtin spelling tables,
per-buf shapes), emits the ``extern "C" __global__ __launch_bounds__(N)
void`` signature, then walks the body — every Stmt's own ``render``
method does the per-line emission.
"""

from __future__ import annotations

from deplodock.compiler.backend.cuda.dtype import cuda_includes, cuda_name
from deplodock.compiler.backend.cuda.dtype import nbytes_of as _nbytes_of
from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.kernel.ir import KernelOp, Smem, TmaDescriptor
from deplodock.compiler.ir.stmt import RenderCtx, render_body
from deplodock.compiler.tensor import Tensor

# Per-CTA static-smem hard cap on every CUDA arch we target. Above this,
# PTXAS rejects ``__shared__ T arr[N]`` decls (``uses too much shared
# data``); the only way to use more is dynamic smem (``extern __shared__``
# + ``cudaFuncAttributeMaxDynamicSharedMemorySize``). When the kernel's
# total Smem footprint exceeds this, ``render_kernelop`` switches to a
# single dynamic pool with per-buffer offsets.
STATIC_SMEM_CAP = 48 * 1024

# TMA / mbarrier prelude. NVRTC doesn't ship ``<cuda.h>`` /
# ``<cuda/ptx>`` / ``<cuda/barrier>``, so we can't ``#include`` the
# stock ``cuda::ptx::*`` intrinsics; pulling those headers in via a
# CUDA-toolkit-version-locked include path also drags in libcu++.
# Instead we forward-declare ``CUtensorMap`` as an opaque 128-byte
# aligned struct (the kernel only takes its address) and define small
# ``__forceinline__`` wrappers around the inline-PTX so the body reads
# like ``mbarrier_init(&mbar[s], 1)`` rather than 5 lines of asm. Same
# generated SASS as raw asm — these helpers fold away at compile time.
_TMA_PRELUDE = """\
struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };

static __device__ __forceinline__ void mbarrier_init(unsigned long long* mbar, int count) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\\n" :: "r"(addr), "r"(count) : "memory");
}

static __device__ __forceinline__ void mbarrier_arrive_expect_tx(unsigned long long* mbar, int bytes) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    unsigned long long state;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;\\n"
                 : "=l"(state) : "r"(addr), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void mbarrier_arrive(unsigned long long* mbar) {
    // Simple arrive — no transaction-byte count. Used by warp-specialized
    // consumer warps to signal "slot empty" after the producer's
    // expect-tx round has been consumed.
    unsigned int addr = __cvta_generic_to_shared(mbar);
    unsigned long long state;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\\n"
                 : "=l"(state) : "r"(addr) : "memory");
}

static __device__ __forceinline__ bool mbarrier_try_wait_parity(unsigned long long* mbar, int phase) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    int ready;
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.b32 %0, 1, 0, P;}\\n"
                 : "=r"(ready) : "r"(addr), "r"(phase) : "memory");
    return ready != 0;
}

static __device__ __forceinline__ void mbarrier_wait_parity(unsigned long long* mbar, int phase) {
    // Issue one ``mbarrier.try_wait`` first — its hint timeout makes the
    // warp suspend rather than spin while the TMA tx drains, freeing the
    // scheduler to run other warps. The PTX-level ``while !try_wait`` loop
    // is required (try_wait can return early); the suspend hint prevents
    // hot-spinning across all 256 CTA threads (~3-4× kernel speedup on
    // small matmuls where the wait-vs-compute ratio is high).
    //
    // The ``"memory"`` clobber prevents the compiler from reordering
    // smem loads across this asm. The primary correctness anchor is
    // the trailing ``__syncthreads()`` materialize emits after each
    // MbarrierWait (see ``100_materialize_tile.py``); the clobber is
    // defensive belt-and-braces so the asm itself reads as a fence
    // even if a future caller forgets the surrounding Sync.
    unsigned int addr = __cvta_generic_to_shared(mbar);
    asm volatile("{.reg .pred P; bw: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra bw;}\\n"
                 :: "r"(addr), "r"(phase) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* smem, const CUtensorMap* desc, int c0, int c1, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_4d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, int c3, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(maddr) : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_5d(
    void* smem, const CUtensorMap* desc, int c0, int c1, int c2, int c3, int c4, unsigned long long* mbar) {
    unsigned int saddr = __cvta_generic_to_shared(smem);
    unsigned int maddr = __cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];\\n"
                 :: "r"(saddr), "l"(desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(maddr) : "memory");
}

"""

# Warp-level MMA prelude (the ``s16816`` path) — the sole tensor-core path.
# Pure inline PTX (``ldmatrix`` + ``mma.sync.aligned``), so NVRTC needs no
# ``<mma.h>``. ``__forceinline__`` wrappers keep the kernel body reading as
# ``dpl_mma_m16n8k16_{f16,bf16}(c, a, b, c)`` rather than raw asm; same SASS.
# The ``a``/``b`` operands are ``unsigned`` 32-bit register arrays (two packed
# 16-bit elems each); ``c``/``d`` are ``float`` (f32 accumulate). Lane→element
# layout is the PTX-fixed mma.m16n8k16 fragment map (see the ``LdmatrixLoad`` /
# ``RegStore`` address arithmetic in ``ir/kernel/ir.py``).
_MMA_SYNC_PRELUDE = """\
static __device__ __forceinline__ void dpl_ldmatrix_x4(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ void dpl_ldmatrix_x2_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];\\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

static __device__ __forceinline__ void dpl_mma_m16n8k16_f16(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void dpl_mma_m16n8k16_bf16(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

"""

_INTRINSIC_TO_CUDA: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
    "erf": "erff",
}

_BUILTIN_TO_CUDA: dict[str, str] = {
    "thread_idx.x": "threadIdx.x",
    "thread_idx.y": "threadIdx.y",
    "thread_idx.z": "threadIdx.z",
    "block_idx.x": "blockIdx.x",
    "block_idx.y": "blockIdx.y",
    "block_idx.z": "blockIdx.z",
    "block_dim.x": "blockDim.x",
    "block_dim.y": "blockDim.y",
    "block_dim.z": "blockDim.z",
    "grid_dim.x": "gridDim.x",
    "grid_dim.y": "gridDim.y",
    "grid_dim.z": "gridDim.z",
    "warp_size": "warpSize",
}

# Block size for the linear thread-flattening path (``Tile.block_axes``
# empty); the host-side launcher rounds the grid up to cover all threads.
_BLOCK_SIZE = 256


def render_kernelop(
    kernel_op: KernelOp,
    tensors: dict[str, Tensor] | None = None,
    shapes: dict[str, tuple[int, ...]] | None = None,
    literal_constants: dict[str, float] | None = None,
    runtime_args: tuple[str, ...] = (),
) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function for a ``KernelOp``.

    ``tensors`` maps each global-buffer name (anything appearing on a
    ``Load.input`` or ``Write.output``) to a :class:`Tensor` describing
    its shape + dtype. The renderer uses the shape to row-major-flatten
    multi-dim indices and the dtype for kernel-signature param types.
    Production callers typically build this from the surrounding graph
    (``{nid: graph.nodes[nid].output for nid in ...}``); tests pass it
    as a literal dict.

    ``shapes`` is the legacy form (shape-only, no dtype) — kept for
    back-compat with tests that pre-date the dtype migration; dtypes
    default to F32 in that path. New callers should prefer ``tensors``.

    ``literal_constants`` maps input-buffer names to scalar values that
    should be embedded in the kernel body as ``float`` literals instead
    of passed as kernel parameters. Loads of those bufs render as
    ``float name = <value>;`` (see ``Load.render``) and the buf is
    excluded from the kernel signature.

    Kernel signature is derived from the body: ``kernel_op.inputs``
    (distinct ``Load.input`` names) become input params,
    ``kernel_op.outputs`` (distinct ``Write.output`` names) become
    writeable output params, ordered by first appearance. Parameter
    types come from the per-buffer :class:`Tensor.dtype` (passed in via
    ``tensors=`` / ``shapes=`` — the caller supplies them); literal-constant
    inputs are skipped. Unknown buffers fall back to ``F32``.
    """
    literals = dict(literal_constants or {})
    tmap: dict[str, Tensor] = dict(tensors) if tensors else {}
    if shapes:
        for n, s in shapes.items():
            tmap.setdefault(n, Tensor(n, tuple(s)))
    smem_offsets, smem_total = _compute_dynamic_smem_offsets(kernel_op)
    # Body.coordination populates atomic_writes / broadcast_writes so
    # ``Write.render`` can decide ``atomicAdd`` and broadcast-guard
    # emission from structural body analysis (block axes / cooperative
    # thread axes vs. Write.index).
    escape = kernel_op.body.coordination
    ctx = RenderCtx(
        target=CudaRenderTarget(),
        shapes={n: tuple(t.shape) for n, t in tmap.items()},
        indent=1,
        intrinsics=_INTRINSIC_TO_CUDA,
        builtins=_BUILTIN_TO_CUDA,
        literal_constants=literals,
        smem_dynamic_offsets=smem_offsets,
        buffer_dtypes={n: t.dtype.name for n, t in tmap.items()},
        atomic_writes=dict(escape._write_atomic_axes),
        broadcast_writes=dict(escape._write_broadcast_axes),
    )

    def _dtype_for(name: str) -> object:
        return tmap[name].dtype if name in tmap else F32

    sig_parts = [f"const {cuda_name(_dtype_for(n))}* {n}" for n in kernel_op.inputs if n not in literals]
    sig_parts.extend(f"{cuda_name(_dtype_for(n))}* {n}" for n in kernel_op.outputs)
    # TMA descriptors are passed as ``__grid_constant__`` value parameters.
    # The kernel only takes their address (``&desc``) for inline asm, so
    # the opaque ``CUtensorMap`` forward decl above suffices.
    desc_names = tuple(dict.fromkeys(s.name for s in kernel_op.body.iter_of_type(TmaDescriptor)))
    # Descriptors are passed by pointer (placed in global memory by the
    # host) rather than ``__grid_constant__`` value parameters: cupy's
    # arg-packing path doesn't preserve the 64-byte alignment that
    # by-value ``CUtensorMap`` parameters require, so this avoids a
    # CUDA_ERROR_MISALIGNED_ADDRESS at launch.
    sig_parts.extend(f"const CUtensorMap* __restrict__ {n}" for n in desc_names)
    # Runtime int args for symbolic axis extents (Dim("seq_len") etc.) come
    # after buffers and TMA descriptors so the launcher can simply tail-append
    # the resolved int values to the kernel-arg pack.
    sig_parts.extend(f"int {n}" for n in runtime_args)
    params_text = ", ".join(sig_parts)
    bounds = _launch_bounds_for(kernel_op)
    launch_bounds = f"\n__launch_bounds__({bounds})"

    body_text = "\n".join(render_body(kernel_op.body, ctx))
    if smem_offsets:
        # All Smem decls were rewritten to pointer aliases into a single
        # dynamic pool; declare the pool at function entry. ``__align__(16)``
        # satisfies TMA's 16-byte requirement and any FP32/FP64 alignment.
        # ``extern __shared__`` arrays must be declared without an explicit
        # size — NVCC otherwise treats the decl as a *static* definition
        # (with the static cap), defeating the whole point of the switch.
        # The runtime size is supplied at launch via ``shared_mem=``.
        pool_decl = f"    extern __shared__ __align__(16) unsigned char _smem_pool[];  // {smem_total} bytes\n"
        body_text = pool_decl + body_text
    prelude = _TMA_PRELUDE if desc_names else ""
    sig_dtypes = [_dtype_for(n) for n in kernel_op.inputs if n not in literals]
    sig_dtypes.extend(_dtype_for(n) for n in kernel_op.outputs)
    includes = "".join(f"#include {h}\n" for h in cuda_includes(sig_dtypes))
    # The mma.sync (s16816) tensor-core path is pure inline PTX — its
    # ldmatrix / mma.sync wrappers are emitted in ``_MMA_SYNC_PRELUDE``, so
    # NVRTC needs no ``<mma.h>`` (the legacy ``nvcuda::wmma`` family is gone).
    from deplodock.compiler.ir.kernel.ir import MmaSyncPtx  # noqa: PLC0415

    uses_mma_sync = any(isinstance(s, MmaSyncPtx) for s in kernel_op.body.iter())
    mma_sync_prelude = _MMA_SYNC_PRELUDE if uses_mma_sync else ""
    header = f'{includes}{mma_sync_prelude}{prelude}extern "C" __global__{launch_bounds} void {kernel_op.name}({params_text})'
    return f"{header} {{\n{body_text}\n}}\n"


def _compute_dynamic_smem_offsets(kernel_op: KernelOp) -> tuple[dict[str, int], int]:
    """Walk the body collecting ``Smem`` decls. If their summed footprint
    exceeds the static cap, return ``({name: byte_offset}, total_bytes)``
    so ``Smem.render`` can emit pool aliases. Otherwise return an empty
    map — the kernel keeps using ``__shared__ T arr[N]``.

    Offsets are aligned on the larger of the dtype size and the Smem's
    explicit ``align`` field (TMA slabs request 16-byte). The final pool
    size is the offset right after the last buffer (no trailing padding
    required — dynamic smem is sized at launch time)."""
    smems: list[Smem] = list(kernel_op.smem_buffers.values())
    if not smems:
        return {}, 0

    total = 0
    sizes: list[tuple[Smem, int]] = []
    for s in smems:
        elements = 1
        for e in s.extents:
            elements *= int(e)
        n_bytes = elements * _nbytes_of(s.dtype)
        total += n_bytes
        sizes.append((s, n_bytes))
    if total <= STATIC_SMEM_CAP:
        return {}, 0

    offsets: dict[str, int] = {}
    cursor = 0
    for s, n_bytes in sizes:
        natural = _nbytes_of(s.dtype)
        align = max(natural, int(s.align) if s.align else 0)
        cursor = (cursor + align - 1) // align * align
        offsets[s.name] = cursor
        cursor += n_bytes
    return offsets, cursor


def _launch_bounds_for(kernel_op: KernelOp) -> int:
    """Derive ``__launch_bounds__`` from the outermost tile flavor.

    - ``GridTile`` (cooperative): launch bounds = product of inner
      ``ThreadTile`` axis extents (per-CTA thread count), or — for a
      ``WarpTile`` inner — ``prod(warp_extents) * 32`` (32 lanes per warp).
    - Standalone ``ThreadTile`` (pointwise): use the default ``_BLOCK_SIZE``
      since launch is flattened across blockIdx + threadIdx.
    """
    from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile, WarpTile  # noqa: PLC0415

    for s in kernel_op.body:
        if isinstance(s, GridTile):
            for child in s.body:
                if isinstance(child, ThreadTile):
                    bsize = 1
                    for ax in child.axes:
                        bsize *= ax.extent.as_static()
                    return max(bsize, 1)
                if isinstance(child, WarpTile):
                    bsize = 32
                    for ax in child.axes:
                        bsize *= ax.extent.as_static()
                    return max(bsize, 32)
            return _BLOCK_SIZE
        if isinstance(s, ThreadTile):
            return _BLOCK_SIZE
    return _BLOCK_SIZE


__all__ = ["render_kernelop"]
