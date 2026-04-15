"""Tensor operation types for the minimal IR."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Region's primary output shape is precomputed during fusion.
        if self.output_names and self.shapes:
            return tuple(self.shapes.get(self.output_names[0], ()))
        return ()


# ---------------------------------------------------------------------------
# KernelOp: structured fused op (prologue → core → epilogue)
# ---------------------------------------------------------------------------


@dataclass
class Port:
    """How a KernelOp reads/writes one external buffer.

    ``buffer_id`` names the external graph node; ``indexmap`` (when set)
    describes the per-output coord access pattern (transpose, slice,
    broadcast, cat). ``None`` = identity load/store at the natural shape.
    """

    buffer_id: str
    indexmap: IndexMapOp | None = None


@dataclass
class ReduceStage:
    """One reduction in a multi-reduce chain.

    ``pre_ops``: elementwise chain between the previous stage's output (or
    the prologue, for the first stage) and this reduce. Empty for a single
    reduce immediately after the prologue.

    ``reduce``: the ReduceOp Node (kept untyped to avoid circular import).
    """

    pre_ops: tuple  # tuple[Node, ...]
    reduce: object  # Node


@dataclass
class ContractionCore:
    """Matmul-shaped sum reduction: out[..., m, n] = sum_k a[..., m, k] * b[..., k, n].

    The IndexMaps on a/b absorb transpositions and broadcasts that used to
    be separate ops upstream. The mul + sum are implicit — this is THE
    matmul template.
    """

    a: Port
    b: Port
    k_axis: int  # which dim of a's post-IndexMap shape is K


@dataclass
class KernelOp(Op):
    """One kernel's worth of computation: prologue → core → epilogue.

    The four codegen templates encode in the union, not in a tag string:
      - core is None                       ⇒ pointwise (prologue only)
      - isinstance(core, ContractionCore)  ⇒ matmul / batched matmul
      - isinstance(core, tuple)            ⇒ reduce chain (1+ ReduceStages)

    Each chain stage is a topologically-sorted tuple of primitive Nodes.
    Cross-stage data flow is implicit: the last node of ``prologue`` feeds
    the core; the core's output feeds the first node of ``epilogue``. A
    KernelOp has one external output (Port) by construction; multi-output
    fusion is a future extension.

    During Stage 2 of the refactor, ``prologue`` may hold the full flat
    node list (core=None); Stage 3 rules populate ``core`` structurally.
    Compat properties emulate the old ``FusedRegionOp`` field names so
    backend readers work unchanged until their migration.
    """

    inputs: list  # list[Port] — external reads
    outputs: list  # list[Port] — external writes (today: always 1)
    prologue: tuple = ()  # tuple[Node, ...] — elementwise chain
    # core: ContractionCore | tuple[ReduceStage, ...] | None
    core: object = None
    epilogue: tuple = ()  # tuple[Node, ...] — elementwise chain
    kernel_source: str = ""  # backend-set after emission
    external_shapes: dict = field(default_factory=dict)  # buffer_id → shape for external buffers

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Walk stages in reverse priority to find the output shape.
        if self.epilogue:
            return tuple(self.epilogue[-1].output.shape)
        if isinstance(self.core, ContractionCore):
            return self._contraction_out_shape(input_shapes)
        if isinstance(self.core, tuple) and self.core:
            return tuple(self.core[-1].reduce.output.shape)
        if self.prologue:
            return tuple(self.prologue[-1].output.shape)
        return ()

    def _contraction_out_shape(self, input_shapes: list[tuple]) -> tuple:
        core = self.core
        assert isinstance(core, ContractionCore)
        a_shape = self._port_shape(core.a, input_shapes)
        b_shape = self._port_shape(core.b, input_shapes)
        return tuple(a_shape[:-1]) + (b_shape[-1],)

    def _port_shape(self, port: Port, input_shapes: list[tuple]) -> tuple:
        if port.indexmap is not None:
            return tuple(port.indexmap.out_shape)
        for i, p in enumerate(self.inputs):
            if p.buffer_id == port.buffer_id:
                return tuple(input_shapes[i])
        return ()

    # ------------------------------------------------------------------
    # Backward-compat shims: emulate FusedRegionOp API during migration.
    # Each flattens prologue + core nodes + epilogue into (id, op, inputs)
    # tuples, derives name lists from Port lists, and merges internal
    # node shapes with external buffer shapes.
    # ------------------------------------------------------------------

    @property
    def region_ops(self) -> list:
        result: list = []
        for node in self.prologue:
            result.append((node.id, node.op, list(node.inputs)))
        if isinstance(self.core, tuple):
            for stage in self.core:
                for node in stage.pre_ops:
                    result.append((node.id, node.op, list(node.inputs)))
                r = stage.reduce
                result.append((r.id, r.op, list(r.inputs)))
        for node in self.epilogue:
            result.append((node.id, node.op, list(node.inputs)))
        return result

    @property
    def input_names(self) -> list:
        return [p.buffer_id for p in self.inputs]

    @property
    def output_names(self) -> list:
        return [p.buffer_id for p in self.outputs]

    @property
    def shapes(self) -> dict:
        result: dict = dict(self.external_shapes)
        for node in self.prologue:
            result[node.id] = tuple(node.output.shape)
        if isinstance(self.core, tuple):
            for stage in self.core:
                for node in stage.pre_ops:
                    result[node.id] = tuple(node.output.shape)
                result[stage.reduce.id] = tuple(stage.reduce.output.shape)
        for node in self.epilogue:
            result[node.id] = tuple(node.output.shape)
        return result
