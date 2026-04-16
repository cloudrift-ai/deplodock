"""Tensor operation types and structural kernel IR.

This module defines two layers:

1. **Primitive ops** (``ElementwiseOp``, ``ReduceOp``, ``IndexMapOp``,
   ``ConstantOp``, ``TransposeOp``, ``ReshapeOp``, ``SliceOp``, ``CatOp``,
   ``LinearOp``, ``MatmulOp``, ``SdpaOp``, ``UnsqueezeOp``, ``MeanOp``, ...)
   — one ``Op`` subclass per primitive tensor operation. A ``Node[T_Op]``
   in ``ir.py`` wraps one of these.

2. **Structural kernel IR** (``KernelOp`` and its tree types below).
   One ``KernelOp`` is one GPU kernel; its input-assembly tree and body
   together describe *what* the kernel computes. Analogies for readers:

   - **Dataflow / signal-flow graph** — leaves are buffer-backed sources,
     internal nodes transform values, edges carry per-coord values. This
     is the framing for the ``KernelInput`` tree as a whole.
   - **Hardware multiplexer** (FPGA N-to-1 mux / 1-to-N demux) — for
     coord-predicated selection, inputs and outputs both.
   - **Operad / expression tree** — N-ary ops composed with operadic
     identity collapse, for ``Combine``.
   - **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store;
     MLIR ``linalg`` structured ops) — for ``KernelOp`` as a whole.
   - **Systolic core** — fused multiply-accumulate over a reduction axis,
     for ``ContractionCore``.

Every elementwise chain in the structural IR is a
``tuple[Node[ElementwiseOp], ...]`` (alias ``ElementwiseChain``); every
reduction slot is a ``Node[ReduceOp]``. Invariants are enforced at
construction time by ``__post_init__`` hooks on the structural
dataclasses (see the ``_assert_*`` helpers at the bottom of the file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.backend.ir.expr import Expr
    from deplodock.compiler.ir import Node


@dataclass
class Op:
    """Base class for all operations."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Derive the output shape from input shapes.

        Override in subclasses with op-specific logic. Used by graph rewrites
        (e.g. ``shape_utils.propagate_shapes``) to re-derive shapes after an
        upstream rewrite changes its inputs.
        """
        raise NotImplementedError(f"{type(self).__name__}.infer_output_shape not implemented")


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("InputOp has no inputs; use node.output.shape directly")


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
    "pow": OpInfo(2),
    "abs": OpInfo(1),
}

# Default for unknown ops: assume unary, non-commutative.
_DEFAULT_OP_INFO = OpInfo(1)


def _drop_axis(shape: tuple, axis: int | str) -> tuple:
    """Return shape with the given axis removed (handles negative axes)."""
    if not isinstance(axis, int):
        return tuple(shape)  # symbolic axis — leave shape as-is
    a = axis if axis >= 0 else len(shape) + axis
    if a < 0 or a >= len(shape):
        return tuple(shape)
    return tuple(shape[:a]) + tuple(shape[a + 1 :])


@dataclass
class ElementwiseOp(Op):
    """Apply a scalar function independently to each element."""

    fn: str  # "mul", "add", "exp", "sub", "div", ...

    @property
    def info(self) -> OpInfo:
        return OP_REGISTRY.get(self.fn, _DEFAULT_OP_INFO)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        from deplodock.compiler.shape_utils import broadcast_shapes

        return broadcast_shapes(*input_shapes)


@dataclass(frozen=True)
class ReduceInfo:
    """Declarative metadata for a reduction op."""

    identity: float  # identity element: 0 for sum, -inf for max, 1 for prod


REDUCE_REGISTRY: dict[str, ReduceInfo] = {
    "sum": ReduceInfo(0.0),
    "max": ReduceInfo(-1e30),
    "prod": ReduceInfo(1.0),
}

_DEFAULT_REDUCE_INFO = ReduceInfo(0.0)


@dataclass
class ReduceOp(Op):
    """Collapse one or more dimensions via an associative binary op."""

    fn: str  # "sum", "max", "prod"
    axis: int | str  # concrete or symbolic

    @property
    def info(self) -> ReduceInfo:
        return REDUCE_REGISTRY.get(self.fn, _DEFAULT_REDUCE_INFO)

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _drop_axis(input_shapes[0], self.axis)


@dataclass
class ScanOp(Op):
    """Cumulative application of an associative binary op along an axis."""

    fn: str  # "sum", "max", "prod"
    axis: int | str

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scan preserves shape


@dataclass
class GatherOp(Op):
    """Read elements from arbitrary positions along an axis."""

    axis: int | str

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Output shape = input shape with the gather axis sized by the index input.
        # Conservative fallback: keep input shape (callers should pre-size if needed).
        return tuple(input_shapes[0])


@dataclass
class ScatterOp(Op):
    """Write (or reduce) values into arbitrary positions along an axis."""

    axis: int | str
    reduce_fn: str | None = None  # None = overwrite, "sum" = scatter-add

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scatter preserves the destination shape


# ---------------------------------------------------------------------------
# Structural ops (for lowering from PyTorch)
# ---------------------------------------------------------------------------


@dataclass
class ConstantOp(Op):
    """Fixed tensor: weights, RoPE tables, scalars. Not an activation."""

    name: str
    value: float | None = None  # scalar value captured at trace time

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("ConstantOp has no inputs; use node.output.shape directly")


@dataclass
class TransposeOp(Op):
    """Permute dimensions.

    ``axes`` either lists a full permutation (``len(axes) == ndim``) or
    names two axes to swap (``len(axes) == 2``), matching torch's
    ``permute``/``transpose`` overloads.
    """

    axes: tuple[int, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = input_shapes[0]
        ndim = len(in_shape)
        if len(self.axes) == ndim:
            return tuple(in_shape[a] for a in self.axes)
        a, b = self.axes[0] % ndim, self.axes[1] % ndim
        out = list(in_shape)
        out[a], out[b] = out[b], out[a]
        return tuple(out)


@dataclass
class ReshapeOp(Op):
    """Reshape tensor without changing data."""

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.shape)


@dataclass
class SliceOp(Op):
    """Extract a sub-tensor along a dimension.

    Inputs: [tensor, dim_const, start_const, end_const] where the
    constants are scalar ConstantOps from the tracer.
    """

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.shape)


@dataclass
class CatOp(Op):
    """Concatenate tensors along a dimension.

    Inputs: [dim_const, tensor_1, tensor_2, ...] where dim_const
    is a scalar ConstantOp indicating the concat axis.
    """

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Tensor inputs are all but the trailing scalar dim-constant.
        # Find them by skipping shape-(1,) inputs at the tail.
        tensor_shapes = [s for s in input_shapes if len(s) > 1 or (len(s) == 1 and isinstance(s[0], int) and s[0] != 1)]
        if not tensor_shapes:
            return tuple(input_shapes[0])
        # Cat along the last dim by default (matches CatOp tracer convention).
        ndim = len(tensor_shapes[0])
        out = list(tensor_shapes[0])
        last = ndim - 1
        total = 0
        for s in tensor_shapes:
            d = s[last]
            if not isinstance(d, int):
                return tuple(out)  # symbolic; bail out
            total += d
        out[last] = total
        return tuple(out)


# ---------------------------------------------------------------------------
# Torch IR ops (captured from PyTorch, decomposed by rewriter passes)
# ---------------------------------------------------------------------------


@dataclass
class LinearOp(Op):
    """PyTorch aten.linear: output = x @ weight.T [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]  # (out_features, in_features)
        return tuple(x_shape[:-1]) + (w_shape[-2],)


@dataclass
class MatmulOp(Op):
    """PyTorch aten.mm/matmul/addmm: output = A @ B [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        # Standard matmul: A(..., M, K) @ B(..., K, N) → (..., M, N)
        return tuple(a_shape[:-1]) + (b_shape[-1],)


@dataclass
class SdpaOp(Op):
    """PyTorch scaled_dot_product_attention(Q, K, V, ...)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # SDPA output mirrors Q's batch+heads+seq dims, with V's last (head_dim).
        q_shape = input_shapes[0]
        v_shape = input_shapes[2]
        return tuple(q_shape[:-1]) + (v_shape[-1],)


@dataclass
class UnsqueezeOp(Op):
    """PyTorch aten.unsqueeze: add a size-1 dimension."""

    dim: int = 0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = list(input_shapes[0])
        d = self.dim if self.dim >= 0 else len(in_shape) + 1 + self.dim
        in_shape.insert(d, 1)
        return tuple(in_shape)


@dataclass
class MeanOp(Op):
    """PyTorch aten.mean.dim: reduction that averages along an axis.

    Kept as its own op so the tracer does a faithful 1:1 capture; a
    decomposition rule rewrites it into sum + div.
    """

    axis: int | str = -1

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _drop_axis(input_shapes[0], self.axis)


# ---------------------------------------------------------------------------
# Unified view op (subsumes Slice/Cat/Transpose/Reshape/Unsqueeze)
# ---------------------------------------------------------------------------


@dataclass
class IndexSource:
    """One input source for an IndexMapOp.

    ``coord_map[i]`` is a ``LoopExpr`` producing the input's i-th index from
    placeholder vars ``Var("out_coord_0")``, ``Var("out_coord_1")``, ...
    See ``deplodock.compiler.coord_expr`` for the substitution helpers.

    ``select`` is None for single-source ops; for multi-source IndexMaps
    (cat) it's a boolean ``LoopExpr`` selecting which output positions
    read this source.
    """

    input_idx: int  # position in IndexMapOp's input list
    coord_map: tuple  # tuple[LoopExpr, ...] — kept untyped to avoid backend import at import time
    select: object | None = None  # LoopExpr | None


@dataclass
class IndexMapOp(Op):
    """Compute output by reindexing inputs via affine coord arithmetic.

    Subsumes Slice, Cat, Transpose, Reshape, Unsqueeze — every layout-only
    op is a function from output coordinates to input coordinates.
    Multi-source forms (cat) use ``select`` on each source to pick which
    output positions read which input.
    """

    out_shape: tuple[int, ...]
    sources: tuple[IndexSource, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.out_shape)

    def is_identity(self, input_shape: tuple) -> bool:
        """True when this IndexMap is a pure pointer alias of its single input."""
        from deplodock.compiler.backend.ir.expr import Var
        from deplodock.compiler.coord_expr import PLACEHOLDER_PREFIX

        if len(self.sources) != 1:
            return False
        src = self.sources[0]
        if src.select is not None:
            return False
        if tuple(self.out_shape) != tuple(input_shape):
            return False
        for i, c in enumerate(src.coord_map):
            if not isinstance(c, Var) or c.name != f"{PLACEHOLDER_PREFIX}{i}":
                return False
        return True


# ---------------------------------------------------------------------------
# Structural kernel IR
# ---------------------------------------------------------------------------
#
# One KernelOp = one GPU kernel. Its shape is a tiled dataflow pipeline:
#
#     inputs (KernelInput tree) ──► [contraction] ──► [reduce_stages] ──►
#                                        [epilogue] ──► outputs (Port | Mux)
#
# Every stage except inputs/outputs is optional; omitting all three mid
# stages yields a pointwise / copy kernel (body lives inside inputs[0]).
#
# KernelInput is a recursive tagged union (``Port | Mux | Combine``):
#   - Port    : signal-flow leaf; one external buffer read + optional indexmap.
#   - Mux     : hardware-mux; coord-predicated dispatch among branches.
#   - Combine : operadic composition; elementwise-chain over N sub-inputs.
#
# KernelOutput is a narrower union (``Port | Mux``): outputs don't
# assemble values, they just dispatch writes (Mux on outputs = scatter).
# Post-body elementwise work (bias, activation, residual) lives in the
# kernel-level ``epilogue`` chain.


@dataclass
class Port:
    """Signal-flow leaf: one external buffer read/write with optional layout.

    ``buffer_id`` names the external graph node; ``indexmap`` (when set)
    describes the per-output coord access pattern (transpose, slice,
    broadcast). ``None`` = identity load/store at the natural shape.
    Used on both the input and output sides of a kernel.
    """

    buffer_id: str
    indexmap: IndexMapOp | None = None


@dataclass
class MuxBranch:
    """One branch of a Mux: an input tree + a coord-predicate selector."""

    input: KernelInput
    select: Expr


@dataclass
class Mux:
    """Hardware multiplexer: coord-predicated dispatch among branches.

    On inputs: at each output coord, exactly one branch's ``select`` is
    True and its ``input`` supplies the value. Branches must be disjoint;
    invariants expect them to be exhaustive or to carry a catch-all in
    the last position (compiler-side convention, not structurally encoded).

    On outputs: same shape, inverted semantics — each branch describes
    where to write when its predicate is True. Unmatched coords produce
    no write (masked scatter).
    """

    branches: tuple[MuxBranch, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Mux.branches must be non-empty")


@dataclass
class Combine:
    """Operadic composition: N sub-inputs combined by an elementwise chain.

    ``sources`` are the operadic inputs (each another ``KernelInput``);
    ``ops`` is an elementwise chain applied to produce one value per
    output coord. Nesting is operadic composition — Combines compose into
    Combines.

    Canonicalization: a no-op wrapper (single source, empty ops) is
    illegal; the tree should already have been collapsed to the source.
    """

    sources: tuple[KernelInput, ...]
    ops: tuple[Node[ElementwiseOp], ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("Combine.sources must be non-empty")
        if len(self.sources) == 1 and not self.ops:
            raise ValueError("Combine(sources=(x,), ops=()) is a no-op wrapper; use the inner input directly")
        _assert_elementwise_chain(self.ops, "Combine.ops")


# A kernel input slot is a signal-flow tree; the leaves read external
# buffers (``Port``), internal nodes transform values (``Combine``) or
# dispatch between sources (``Mux``).
type KernelInput = Port | Mux | Combine

# A kernel output slot is simpler: either a plain write target (``Port``)
# or a scatter/masked writeout (``Mux``). Post-body elementwise work lives
# in ``KernelOp.epilogue``.
type KernelOutput = Port | Mux


@dataclass
class ContractionCore:
    """Systolic core: sum (or associative reduce) over a per-K operand.

    For matmul, ``operand`` is a ``Combine(sources=(a, b), ops=(mul,))``:
    the per-K element product whose reduction over ``k_axis`` yields one
    output element. Generalizes to other associative contractions by
    choosing different ``operand.ops`` chains (e.g. sub+abs for
    sum-of-abs-diff) and different reduce functions.
    """

    operand: KernelInput
    k_axis: int
    reduce: Node[ReduceOp]

    def __post_init__(self) -> None:
        _assert_reduce_node(self.reduce, "ContractionCore.reduce")


@dataclass
class ReduceStage:
    """One reduction in a multi-reduce chain.

    ``pre_ops``: elementwise chain between the previous stage's output
    (or the pipeline's pre-reduce value, for the first stage) and this
    reduce. Empty when the reduce runs directly on the prior output.

    ``reduce``: the ``ReduceOp`` Node. The previous stage's reduced axis
    is gone from the iteration space by the time ``pre_ops`` runs —
    ``pre_ops`` consume in-register accumulators, not external loads.
    """

    pre_ops: tuple[Node[ElementwiseOp], ...]
    reduce: Node[ReduceOp]

    def __post_init__(self) -> None:
        _assert_elementwise_chain(self.pre_ops, "ReduceStage.pre_ops")
        _assert_reduce_node(self.reduce, "ReduceStage.reduce")


@dataclass
class KernelOp(Op):
    """One kernel's worth of computation: a tiled dataflow pipeline.

    The pipeline is linear:

        inputs → [contraction] → [reduce_stages] → [epilogue] → outputs

    - ``inputs``: a tuple of ``KernelInput`` trees, one per logical body
      input. Recursive (Port | Mux | Combine); every elementwise op used
      to assemble a value lives inside the tree.
    - ``contraction``: optional systolic core (matmul and its
      generalizations). When present, its ``operand`` is itself a
      ``KernelInput`` (typically a Combine of two inputs with a mul).
    - ``reduce_stages``: optional sequence of post-contraction (or
      post-input if no contraction) reductions. Each stage's ``pre_ops``
      consume the prior stage's (or contraction's) output.
    - ``epilogue``: optional post-body elementwise chain at the output
      iteration space (bias, activation, residual).
    - ``outputs``: write targets (Port | Mux) — Mux supports scatter /
      masked writeout.

    ``external_shapes`` keeps buffer-id → shape for every leaf ``Port``
    the codegen needs to allocate launch args for.
    """

    inputs: tuple[KernelInput, ...]
    outputs: tuple[KernelOutput, ...]
    contraction: ContractionCore | None = None
    reduce_stages: tuple[ReduceStage, ...] = ()
    epilogue: tuple[Node[ElementwiseOp], ...] = ()
    external_shapes: dict = field(default_factory=dict)
    kernel_source: str = ""  # backend-set after emission

    def __post_init__(self) -> None:
        _assert_elementwise_chain(self.epilogue, "KernelOp.epilogue")


# ---------------------------------------------------------------------------
# Runtime invariant helpers — back the type annotations on chains / slots.
# Python's type system can't statically enforce that a Node's ``op`` is a
# specific subclass; these run at construction time to fail loudly when a
# rule (or a test) mis-files an op into the wrong slot.
# ---------------------------------------------------------------------------


type ElementwiseChain = tuple[Node[ElementwiseOp], ...]


def _assert_elementwise_chain(chain: ElementwiseChain, where: str) -> None:
    for i, node in enumerate(chain):
        _assert_elementwise_node(node, f"{where}[{i}]")


def _assert_elementwise_node(node: Node[ElementwiseOp], where: str) -> None:
    if not isinstance(node.op, ElementwiseOp):
        raise TypeError(f"{where} has op {type(node.op).__name__}, expected ElementwiseOp")


def _assert_reduce_node(node: Node[ReduceOp], where: str) -> None:
    if not isinstance(node.op, ReduceOp):
        raise TypeError(f"{where} has op {type(node.op).__name__}, expected ReduceOp")
