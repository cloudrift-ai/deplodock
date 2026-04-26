"""Kernel IR → CUDA source.

Builds a CUDA ``RenderCtx`` (intrinsic + GPU-builtin spelling tables,
per-buf shapes), emits the ``extern "C" __global__ __launch_bounds__(N)
void`` signature, then walks the body — every Stmt's own ``render``
method does the per-line emission.
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.stmt import RenderCtx, Tile, render_body

_INTRINSIC_TO_CUDA: dict[str, str] = {
    "exp": "expf",
    "rsqrt": "rsqrtf",
    "tanh": "tanhf",
    "fabs": "fabsf",
    "fmax": "fmaxf",
    "fmin": "fminf",
    "pow": "powf",
    "sqrt": "sqrtf",
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


def render_kernelop(kernel_op: KernelOp, shapes: dict[str, tuple[int, ...]] | None = None) -> str:
    """Render a complete ``extern "C" __global__`` CUDA function for a ``KernelOp``.

    ``shapes`` maps each global-buffer name (anything appearing on a
    ``Load.input`` or ``Write.output``) to its declared shape; the
    renderer uses it to row-major-flatten multi-dim indices. Production
    callers typically build ``shapes`` from the surrounding graph
    (``{nid: graph.nodes[nid].output.shape for nid in ...}``); tests pass
    it as a literal dict.

    Kernel signature is derived from the body: ``kernel_op.inputs`` (distinct
    ``Load.input`` names) become ``const float*`` params, ``kernel_op.outputs``
    (distinct ``Write.output`` names) become ``float*`` params, ordered
    by first appearance.
    """
    ctx = RenderCtx(
        shapes=dict(shapes or {}),
        indent=1,
        intrinsics=_INTRINSIC_TO_CUDA,
        builtins=_BUILTIN_TO_CUDA,
    )

    sig_parts = [f"const float* {n}" for n in kernel_op.inputs]
    sig_parts.extend(f"float* {n}" for n in kernel_op.outputs)
    params_text = ", ".join(sig_parts)
    bounds = _launch_bounds_for(kernel_op)
    launch_bounds = f"\n__launch_bounds__({bounds})"

    body_text = "\n".join(render_body(kernel_op.body, ctx))
    return f'extern "C" __global__{launch_bounds} void {kernel_op.name}({params_text}) {{\n{body_text}\n}}\n'


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
