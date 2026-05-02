"""Kernel IR → CUDA source.

Builds a CUDA ``RenderCtx`` (intrinsic + GPU-builtin spelling tables,
per-buf shapes), emits the ``extern "C" __global__ __launch_bounds__(N)
void`` signature, then walks the body — every Stmt's own ``render``
method does the per-line emission.
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel.ir import KernelOp, TmaDescriptor
from deplodock.compiler.ir.stmt import RenderCtx, Tile, render_body

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

static __device__ __forceinline__ bool mbarrier_try_wait_parity(unsigned long long* mbar, int phase) {
    unsigned int addr = __cvta_generic_to_shared(mbar);
    int ready;
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.b32 %0, 1, 0, P;}\\n"
                 : "=r"(ready) : "r"(addr), "r"(phase));
    return ready != 0;
}

static __device__ __forceinline__ void mbarrier_wait_parity(unsigned long long* mbar, int phase) {
    // Issue one ``mbarrier.try_wait`` first — its hint timeout makes the
    // warp suspend rather than spin while the TMA tx drains, freeing the
    // scheduler to run other warps. The PTX-level ``while !try_wait`` loop
    // is required (try_wait can return early); the suspend hint prevents
    // hot-spinning across all 256 CTA threads (~3-4× kernel speedup on
    // small matmuls where the wait-vs-compute ratio is high).
    unsigned int addr = __cvta_generic_to_shared(mbar);
    asm volatile("{.reg .pred P; bw: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra bw;}\\n"
                 :: "r"(addr), "r"(phase));
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
    shapes: dict[str, tuple[int, ...]] | None = None,
    literal_constants: dict[str, float] | None = None,
) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function for a ``KernelOp``.

    ``shapes`` maps each global-buffer name (anything appearing on a
    ``Load.input`` or ``Write.output``) to its declared shape; the
    renderer uses it to row-major-flatten multi-dim indices. Production
    callers typically build ``shapes`` from the surrounding graph
    (``{nid: graph.nodes[nid].output.shape for nid in ...}``); tests pass
    it as a literal dict.

    ``literal_constants`` maps input-buffer names to scalar values that
    should be embedded in the kernel body as float literals instead of
    passed as ``float*`` parameters. Loads of those bufs render as
    ``float name = <value>;`` (see ``Load.render``) and the buf is
    excluded from the kernel signature.

    Kernel signature is derived from the body: ``kernel_op.inputs`` (distinct
    ``Load.input`` names) become ``const float*`` params, ``kernel_op.outputs``
    (distinct ``Write.output`` names) become ``float*`` params, ordered
    by first appearance. Literal-constant inputs are skipped.
    """
    literals = dict(literal_constants or {})
    ctx = RenderCtx(
        shapes=dict(shapes or {}),
        indent=1,
        intrinsics=_INTRINSIC_TO_CUDA,
        builtins=_BUILTIN_TO_CUDA,
        literal_constants=literals,
    )

    sig_parts = [f"const float* {n}" for n in kernel_op.inputs if n not in literals]
    sig_parts.extend(f"float* {n}" for n in kernel_op.outputs)
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
    params_text = ", ".join(sig_parts)
    bounds = _launch_bounds_for(kernel_op)
    launch_bounds = f"\n__launch_bounds__({bounds})"

    body_text = "\n".join(render_body(kernel_op.body, ctx))
    prelude = _TMA_PRELUDE if desc_names else ""
    return f'{prelude}extern "C" __global__{launch_bounds} void {kernel_op.name}({params_text}) {{\n{body_text}\n}}\n'


def _launch_bounds_for(kernel_op: KernelOp) -> int:
    """Derive ``__launch_bounds__`` from the first ``Tile``'s thread axes
    when ``block_axes`` is populated; otherwise ``_BLOCK_SIZE``."""
    for s in kernel_op.body:
        if isinstance(s, Tile):
            if s.block_axes:
                bsize = 1
                for ax in s.thread_axes:
                    bsize *= int(ax.extent)
                return max(bsize, 1)
            return _BLOCK_SIZE
    return _BLOCK_SIZE


__all__ = ["render_kernelop"]
