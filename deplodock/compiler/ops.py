"""Tensor operation types for the minimal IR."""

from dataclasses import dataclass


@dataclass
class Op:
    """Base class for all operations."""


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""


@dataclass
class ElementwiseOp(Op):
    """Apply a scalar function independently to each element."""

    fn: str  # "mul", "add", "exp", "sub", "div", ...


@dataclass
class ReduceOp(Op):
    """Collapse one or more dimensions via an associative binary op."""

    fn: str  # "sum", "max", "prod"
    axis: int | str  # concrete or symbolic


@dataclass
class ScanOp(Op):
    """Cumulative application of an associative binary op along an axis."""

    fn: str  # "sum", "max", "prod"
    axis: int | str


@dataclass
class GatherOp(Op):
    """Read elements from arbitrary positions along an axis."""

    axis: int | str


@dataclass
class ScatterOp(Op):
    """Write (or reduce) values into arbitrary positions along an axis."""

    axis: int | str
    reduce_fn: str | None = None  # None = overwrite, "sum" = scatter-add


@dataclass
class FusedReduceElementwiseOp(Op):
    """Fused reduce + elementwise — accumulate without materializing intermediate."""

    reduce_fn: str  # "sum", "max", "prod"
    elementwise_fn: str  # "mul", "add", ...
    axis: int | str


# ---------------------------------------------------------------------------
# Structural ops (for lowering from PyTorch)
# ---------------------------------------------------------------------------


@dataclass
class ConstantOp(Op):
    """Fixed tensor: weights, RoPE tables, scalars. Not an activation."""

    name: str


@dataclass
class TransposeOp(Op):
    """Permute dimensions."""

    axes: tuple[int, ...]


@dataclass
class ReshapeOp(Op):
    """Reshape tensor without changing data."""

    shape: tuple[int | str, ...]


# ---------------------------------------------------------------------------
# Fused ops (assembly targets)
# ---------------------------------------------------------------------------


@dataclass
class MatmulOp(Op):
    """Matrix multiply — fused Reduce{sum}(Elementwise{mul})."""


@dataclass
class FusedRMSNormOp(Op):
    """Fused RMS normalization: rsqrt(mean(x^2) + eps) * x * weight."""

    eps: float


@dataclass
class FusedSoftmaxOp(Op):
    """Fused online softmax along an axis."""

    axis: int | str


@dataclass
class FusedSiLUMulOp(Op):
    """Fused SiLU activation with elementwise multiply: silu(gate) * up."""


@dataclass
class FusedAttentionOp(Op):
    """Flash attention: Q @ K^T -> scale -> softmax -> @ V."""

    num_heads: int
    head_dim: int
    scale: float
