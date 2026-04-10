"""CUDA kernel templates loaded from .cu files.

Each kernel is a .cu file with __KERNEL_NAME__ placeholder.
load_kernel() reads the file and substitutes placeholders.
"""

from pathlib import Path

_KERNELS_DIR = Path(__file__).parent


def load_kernel(template: str, **substitutions: str) -> str:
    """Load a .cu template and apply __KEY__ substitutions.

    Args:
        template: Base name of the .cu file (without extension).
        **substitutions: key=value pairs. Each __KEY__ in the template
            is replaced with the value (key is uppercased automatically).

    Returns:
        CUDA source string with substitutions applied.
    """
    path = _KERNELS_DIR / f"{template}.cu"
    source = path.read_text()
    for key, value in substitutions.items():
        source = source.replace(f"__{key.upper()}__", str(value))
    return source


# --- Backwards-compatible API ---
# These match the old function signatures so existing imports keep working.


def fused_rmsnorm_source(name: str = "fused_rmsnorm") -> str:
    return load_kernel("rmsnorm", kernel_name=name)


def fused_silu_mul_source(name: str = "fused_silu_mul") -> str:
    return load_kernel("activation", kernel_name=name)


def fused_rope_source(name: str = "fused_rope") -> str:
    return load_kernel("rope", kernel_name=name)


def naive_attention_qk_source(name: str = "attention_qk") -> str:
    return load_kernel("attention_qk", kernel_name=name)


def naive_attention_softmax_source(name: str = "attention_softmax") -> str:
    return load_kernel("attention_softmax", kernel_name=name)


def naive_attention_sv_source(name: str = "attention_sv") -> str:
    return load_kernel("attention_sv", kernel_name=name)


def naive_matmul_source(name: str = "naive_matmul") -> str:
    return load_kernel("matmul_naive", kernel_name=name)


def matmul_residual_add_source(name: str = "matmul_residual_add") -> str:
    return load_kernel("matmul_residual_add", kernel_name=name)


def triple_matmul_source(name: str = "triple_matmul") -> str:
    return load_kernel("matmul_triple", kernel_name=name)


def dual_matmul_silu_mul_source(name: str = "dual_matmul_silu_mul") -> str:
    return load_kernel("matmul_dual_silu_mul", kernel_name=name)


__all__ = [
    "load_kernel",
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
