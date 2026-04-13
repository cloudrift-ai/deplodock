"""Tensor operation types for the minimal IR."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Op:
    """Base class for all operations."""


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""


# ---------------------------------------------------------------------------
# Op metadata registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpInfo:
    """Declarative metadata for an elementwise op.

    Backend-agnostic: describes semantic properties only, not how the op
    maps to C code (that's the codegen's job).
    """

    arity: int  # 1 (unary) or 2 (binary)
    commutative: bool = False  # True for add, mul; False for sub, div


OP_REGISTRY: dict[str, OpInfo] = {
    "add": OpInfo(2, commutative=True),
    "sub": OpInfo(2),
    "mul": OpInfo(2, commutative=True),
    "div": OpInfo(2),
    "mod": OpInfo(2),
    "neg": OpInfo(1),
    "exp": OpInfo(1),
    "rsqrt": OpInfo(1),
    "recip": OpInfo(1),
    "relu": OpInfo(1),
    "tanh": OpInfo(1),
    "sigmoid": OpInfo(1),
}

# Default for unknown ops: assume unary, non-commutative.
_DEFAULT_OP_INFO = OpInfo(1)


@dataclass
class ElementwiseOp(Op):
    """Apply a scalar function independently to each element."""

    fn: str  # "mul", "add", "exp", "sub", "div", ...

    @property
    def info(self) -> OpInfo:
        return OP_REGISTRY.get(self.fn, _DEFAULT_OP_INFO)


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
class FusedRegionOp(Op):
    """A fused region of primitive ops with a generated kernel.

    Produced by auto_fuse(). Contains the original subgraph of primitive
    ops (for the kernel generator to walk) and the generated CUDA source
    (filled in after kernel generation).
    """

    region_ops: list  # [(node_id, op, input_ids), ...] — primitive ops in topo order
    input_names: list  # external inputs to this region
    output_names: list  # external outputs from this region
    kernel_source: str = ""  # generated CUDA source (filled by cuda/kernel_gen)
    shapes: dict = field(default_factory=dict)  # node_id → shape for all region nodes + inputs
