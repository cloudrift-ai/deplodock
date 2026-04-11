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
    """Matrix multiply — fused Reduce{sum}(Elementwise{mul}).

    Produced by the matmul recognition rule (rules/fusion/001).
    Has specialized SGEMM lowering in backend/cuda/lower.py.
    """


@dataclass
class FusedRegionOp(Op):
    """A fused region of primitive ops with a generated kernel.

    Produced by auto_fuse(). Contains the original subgraph of primitive
    ops (for the kernel generator to walk) and the generated CUDA source
    (filled in after kernel generation).
    """

    region_ops: list  # [(node_id, op, input_ids), ...] — primitive ops in topo order
    input_names: list  # external inputs to this region
    output_names: list  # external outputs from this region
    kernel_source: str = ""  # generated CUDA source (filled by kernel_gen)
