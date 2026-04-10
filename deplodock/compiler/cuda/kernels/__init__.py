"""CUDA kernel templates organized by operation."""

from deplodock.compiler.cuda.kernels.activation import fused_silu_mul_source
from deplodock.compiler.cuda.kernels.attention import (
    naive_attention_qk_source,
    naive_attention_softmax_source,
    naive_attention_sv_source,
)
from deplodock.compiler.cuda.kernels.matmul import (
    dual_matmul_silu_mul_source,
    matmul_residual_add_source,
    naive_matmul_source,
    triple_matmul_source,
)
from deplodock.compiler.cuda.kernels.rmsnorm import fused_rmsnorm_source
from deplodock.compiler.cuda.kernels.rope import fused_rope_source

__all__ = [
    "dual_matmul_silu_mul_source",
    "fused_rmsnorm_source",
    "fused_rope_source",
    "fused_silu_mul_source",
    "matmul_residual_add_source",
    "naive_attention_qk_source",
    "naive_attention_softmax_source",
    "naive_attention_sv_source",
    "naive_matmul_source",
    "triple_matmul_source",
]
