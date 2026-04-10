"""Activation kernels: SiLU, SiLU+Mul."""

from __future__ import annotations


def fused_silu_mul_source(name: str = "fused_silu_mul") -> str:
    """Generate CUDA source for fused SiLU(gate) * up kernel.

    One thread per element. out[i] = silu(gate[i]) * up[i].
    """
    return f"""
__global__ void {name}(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }}
}}
"""
