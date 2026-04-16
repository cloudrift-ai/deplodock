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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.backend.ir.expr import Expr


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

    def forward(self, *inputs):
        """Compute the operation using numpy arrays. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.forward not implemented")


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("InputOp has no inputs; use node.output.shape directly")

    def forward(self, *inputs):
        raise NotImplementedError("InputOp is a sentinel; value is supplied by the executor")


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
        return tuple(shape)
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

    def forward(self, *inputs):
        import numpy as np

        _EW_FN = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "mod": lambda a, b: a % b,
            "neg": lambda a: -a,
            "exp": lambda a: np.exp(a),
            "rsqrt": lambda a: 1.0 / np.sqrt(a),
            "recip": lambda a: 1.0 / a,
            "relu": lambda a: np.maximum(a, 0),
            "tanh": lambda a: np.tanh(a),
            "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
            "pow": lambda a, b: np.power(a, b),
            "abs": lambda a: np.abs(a),
            "silu": lambda a: a / (1.0 + np.exp(-a)),
        }
        fn = _EW_FN.get(self.fn)
        if fn is None:
            raise NotImplementedError(f"ElementwiseOp.forward: unknown fn {self.fn!r}")
        return fn(*inputs)


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

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        _RED_FN = {
            "sum": lambda x, ax: np.sum(x, axis=ax),
            "max": lambda x, ax: np.max(x, axis=ax),
            "prod": lambda x, ax: np.prod(x, axis=ax),
        }
        fn = _RED_FN.get(self.fn)
        if fn is None:
            raise NotImplementedError(f"ReduceOp.forward: unknown fn {self.fn!r}")
        return fn(a, self.axis)


@dataclass
class ScanOp(Op):
    """Cumulative application of an associative binary op along an axis."""

    fn: str  # "sum", "max", "prod"
    axis: int | str

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scan preserves shape

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        _SCAN_FN = {
            "sum": lambda x, ax: np.cumsum(x, axis=ax),
            "max": lambda x, ax: np.maximum.accumulate(x, axis=ax),
            "prod": lambda x, ax: np.cumprod(x, axis=ax),
        }
        fn = _SCAN_FN.get(self.fn)
        if fn is None:
            raise NotImplementedError(f"ScanOp.forward: unknown fn {self.fn!r}")
        return fn(a, self.axis)


@dataclass
class GatherOp(Op):
    """Read elements from arbitrary positions along an axis."""

    axis: int | str

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Output shape = input shape with the gather axis sized by the index input.
        # Conservative fallback: keep input shape (callers should pre-size if needed).
        return tuple(input_shapes[0])

    def forward(self, *inputs):
        import numpy as np

        data, indices = inputs[0], inputs[1].astype(np.intp)
        return np.take_along_axis(data, indices, axis=self.axis)


@dataclass
class ScatterOp(Op):
    """Write (or reduce) values into arbitrary positions along an axis."""

    axis: int | str
    reduce_fn: str | None = None  # None = overwrite, "sum" = scatter-add

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scatter preserves the destination shape

    def forward(self, *inputs):
        import numpy as np

        dest, indices, values = inputs[0].copy(), inputs[1].astype(np.intp), inputs[2]
        if self.reduce_fn == "sum":
            np.add.at(dest, (np.arange(dest.shape[0])[:, None], indices), values)
        else:
            np.put_along_axis(dest, indices, values, axis=self.axis)
        return dest


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

    def forward(self, *inputs):
        import numpy as np

        if self.value is not None:
            return np.array([self.value], dtype=np.float32)
        raise NotImplementedError("ConstantOp with value=None must be supplied by the executor")


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

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        ndim = a.ndim
        if len(self.axes) == ndim:
            return np.transpose(a, self.axes)
        ax0, ax1 = self.axes[0] % ndim, self.axes[1] % ndim
        return np.swapaxes(a, ax0, ax1)


@dataclass
class ReshapeOp(Op):
    """Reshape tensor without changing data."""

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        if -1 not in self.shape:
            return tuple(self.shape)
        in_numel = 1
        for d in input_shapes[0]:
            in_numel *= int(d)
        known = 1
        for d in self.shape:
            if d != -1:
                known *= int(d)
        resolved = list(self.shape)
        resolved[resolved.index(-1)] = in_numel // known if known else 1
        return tuple(resolved)

    def forward(self, *inputs):
        import numpy as np

        return np.reshape(inputs[0], self.shape)


@dataclass
class SliceOp(Op):
    """Extract a sub-tensor along a dimension.

    Inputs: [tensor, dim_const, start_const, end_const] where the
    constants are scalar ConstantOps from the tracer.
    """

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.shape)

    def forward(self, *inputs):
        tensor = inputs[0]
        dim = int(inputs[1].flat[0]) if len(inputs) > 1 else 0
        start = int(inputs[2].flat[0]) if len(inputs) > 2 else 0
        end = int(inputs[3].flat[0]) if len(inputs) > 3 else tensor.shape[dim]
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, end)
        return tensor[tuple(slices)]


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

    def forward(self, *inputs):
        import numpy as np

        arrays = []
        dim = -1
        for inp in inputs:
            if inp.ndim == 0 or (inp.ndim == 1 and inp.size == 1):
                dim = int(inp.flat[0])
            else:
                arrays.append(inp)
        return np.concatenate(arrays, axis=dim)


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

    def forward(self, *inputs):
        x, w = inputs[0], inputs[1]
        result = x @ w.T
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class MatmulOp(Op):
    """PyTorch aten.mm/matmul/addmm: output = A @ B [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        # Standard matmul: A(..., M, K) @ B(..., K, N) → (..., M, N)
        return tuple(a_shape[:-1]) + (b_shape[-1],)

    def forward(self, *inputs):
        a, b = inputs[0], inputs[1]
        result = a @ b
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class SdpaOp(Op):
    """PyTorch scaled_dot_product_attention(Q, K, V, ...)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # SDPA output mirrors Q's batch+heads+seq dims, with V's last (head_dim).
        q_shape = input_shapes[0]
        v_shape = input_shapes[2]
        return tuple(q_shape[:-1]) + (v_shape[-1],)

    def forward(self, *inputs):
        import numpy as np

        q, k, v = inputs[0], inputs[1], inputs[2]
        d_k = q.shape[-1]
        scores = q @ np.swapaxes(k, -2, -1) / np.sqrt(d_k)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return attn @ v


@dataclass
class UnsqueezeOp(Op):
    """PyTorch aten.unsqueeze: add a size-1 dimension."""

    dim: int = 0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = list(input_shapes[0])
        d = self.dim if self.dim >= 0 else len(in_shape) + 1 + self.dim
        in_shape.insert(d, 1)
        return tuple(in_shape)

    def forward(self, *inputs):
        import numpy as np

        return np.expand_dims(inputs[0], axis=self.dim)


@dataclass
class MeanOp(Op):
    """PyTorch aten.mean.dim: reduction that averages along an axis.

    Kept as its own op so the tracer does a faithful 1:1 capture; a
    decomposition rule rewrites it into sum + div.
    """

    axis: int | str = -1

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _drop_axis(input_shapes[0], self.axis)

    def forward(self, *inputs):
        import numpy as np

        return np.mean(inputs[0], axis=self.axis)


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
    ops: tuple[ElementwiseOp, ...]

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
class Assign:
    """One named value in the kernel's SSA body: ``name = op(args)``.

    SSA invariants (enforced by ``KernelOp.__post_init__``):
      - Each ``name`` is defined exactly once across all Assigns.
      - Every ``arg`` references an input ``Port.buffer_id`` or a prior
        ``Assign.name``.
      - No forward references.
    """

    name: str
    op: ElementwiseOp | ReduceOp
    args: tuple[str, ...]


@dataclass
class KernelOp(Op):
    """One kernel's worth of computation as an SSA program.

    The kernel reads external buffers via ``inputs`` (Port | Mux | Combine),
    computes through a flat sequence of named ``Assign`` statements, and
    writes the result via ``outputs`` (Port | Mux).

    Every kernel reads as a program::

        mul = mul(a, b)
        dot = reduce_sum(mul)
        out = add(dot, bias)

    The codegen walks the body sequentially, maintaining a ``values`` dict
    mapping Assign names to C expressions. Contraction (matmul K-loop) is
    detected by pattern-matching the SSA graph, not by a separate field.
    """

    inputs: tuple[KernelInput, ...]
    body: tuple[Assign, ...] = ()
    outputs: tuple[KernelOutput, ...] = ()

    def __post_init__(self) -> None:
        _validate_ssa(self)

    def infer_shapes(self, input_shapes: dict[str, tuple] | None = None) -> dict[str, tuple]:
        """Derive the shape of every named value (inputs + Assigns).

        ``input_shapes`` maps Port ``buffer_id`` → shape for all external
        buffers. When a Port carries an ``indexmap``, its effective shape
        is ``indexmap.out_shape`` regardless of the provided shape.

        Walks the SSA body, calling ``op.infer_output_shape`` at each
        Assign. Returns a dict mapping value names to shapes.
        """
        ext = input_shapes or {}
        shapes: dict[str, tuple] = {}
        for inp in self.inputs:
            if isinstance(inp, Port):
                if inp.indexmap is not None:
                    shapes[inp.buffer_id] = tuple(inp.indexmap.out_shape)
                else:
                    shapes[inp.buffer_id] = tuple(ext.get(inp.buffer_id, ()))
            elif isinstance(inp, Combine):
                for src in inp.sources:
                    if isinstance(src, Port):
                        if src.indexmap is not None:
                            shapes[src.buffer_id] = tuple(src.indexmap.out_shape)
                        else:
                            shapes[src.buffer_id] = tuple(ext.get(src.buffer_id, ()))
        for assign in self.body:
            # Prefer graph-provided shape (from ext dict) if available.
            if assign.name in ext:
                shapes[assign.name] = tuple(ext[assign.name])
                continue
            arg_shapes = [shapes[a] for a in assign.args if a in shapes]
            if arg_shapes:
                try:
                    shapes[assign.name] = assign.op.infer_output_shape(arg_shapes)
                except (ValueError, TypeError):
                    shapes[assign.name] = max(arg_shapes, key=len)
        return shapes

    def infer_output_shape(self, input_shapes: dict[str, tuple] | list[tuple] | None = None) -> tuple:
        """Derive the kernel's output shape from the SSA body.

        ``input_shapes`` is a dict mapping Port buffer_id → shape.
        """
        ext = input_shapes if isinstance(input_shapes, dict) else None
        shapes = self.infer_shapes(ext)
        if self.body:
            return shapes.get(self.body[-1].name, ())
        if shapes:
            return next(iter(shapes.values()))
        return ()


type ElementwiseChain = tuple[ElementwiseOp, ...]


# ---------------------------------------------------------------------------
# Runtime invariant helpers
# ---------------------------------------------------------------------------


def _assert_elementwise_chain(chain: ElementwiseChain, where: str) -> None:
    for i, op in enumerate(chain):
        if not isinstance(op, ElementwiseOp):
            raise TypeError(f"{where}[{i}] is {type(op).__name__}, expected ElementwiseOp")


def _validate_ssa(kernel: KernelOp) -> None:
    """Enforce SSA invariants: unique names, defined-before-use."""
    defined: set[str] = set()
    for inp in kernel.inputs:
        if isinstance(inp, Port):
            defined.add(inp.buffer_id)
        elif isinstance(inp, Combine):
            for src in inp.sources:
                if isinstance(src, Port):
                    defined.add(src.buffer_id)
    for assign in kernel.body:
        for arg in assign.args:
            if arg not in defined:
                raise ValueError(f"Assign {assign.name!r}: arg {arg!r} not defined")
        if assign.name in defined:
            raise ValueError(f"Assign {assign.name!r}: name already defined")
        defined.add(assign.name)
