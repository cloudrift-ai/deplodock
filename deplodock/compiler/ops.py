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
# KernelOp: structured fused op (prologue → core → epilogue)
# ---------------------------------------------------------------------------


def _needed_by_ids(ops: list) -> set:
    """Set of node ids referenced as inputs by the given (id, op, inputs) tuples."""
    needed: set = set()
    for _id, _op, input_ids in ops:
        needed.update(input_ids)
    return needed


@dataclass
class ContractionInfo:
    """Resolved matmul metadata for a ContractionCore kernel.

    Built by ``KernelOp.contraction_info(shapes)``. Codegen reads M/N/K
    + batch info to size the K-loop, tile dims, and grid.
    """

    a_id: str  # external buffer id for A operand
    b_id: str  # external buffer id for B operand
    m: int
    n: int
    k: int
    batch_dims: tuple[int, ...] = ()
    batch_size: int = 1
    # GQA / broadcast batch: when one operand has fewer batch elements,
    # its batch index is divided by this factor (28 Q heads / 4 KV heads = 7).
    a_batch_group: int = 1
    b_batch_group: int = 1


@dataclass
class AccessPattern:
    """How a single input tensor is accessed within the kernel.

    Derived per-input from the input shape and the kernel's output shape.
    Used by load-path emitters to pick the right indexing form.
    """

    shape: tuple[int, ...]
    size: int  # total int-dim element count
    is_scalar: bool  # size == 1
    is_row_vector: bool  # 1D, indexed by column only
    is_2d: bool  # indexed by both row and column
    is_per_row: bool = False  # last dim == 1 with >1 total elts
    is_broadcast: bool = False  # smaller than output, broadcast via modulo


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
    be separate ops upstream. The ``mul`` and ``reduce`` Node fields hold
    the elementwise-multiply and sum-reduction operations that the
    structural rule moved out of the KernelOp's prologue — they're
    "implicit" in the template but explicit in the IR so the backend can
    reach them for codegen.

    ``post_stages`` lets a contraction carry a downstream row-reduce chain
    in the same kernel (matmul → softmax fusion): the contraction reduce
    runs first, then each ReduceStage applies its pre_ops chain and
    reduce to the per-row accumulator.
    """

    a: Port
    b: Port
    k_axis: int  # which dim of a's post-IndexMap shape is K
    mul: object = None  # Node holding the elementwise multiply
    reduce: object = None  # Node holding the sum reduction
    post_stages: tuple = ()  # tuple[ReduceStage, ...] — downstream row reduces


@dataclass
class KernelOp(Op):
    """One kernel's worth of computation: prologue → core → epilogue.

    The four codegen templates encode in the union, not in a tag string:
      - core is None                       ⇒ pointwise (prologue only)
      - isinstance(core, ContractionCore)  ⇒ matmul / batched matmul
      - isinstance(core, tuple)            ⇒ reduce chain (1+ ReduceStages)

    Each chain stage is a topologically-sorted tuple of primitive Nodes.
    Cross-stage data flow is implicit: ``prologue`` holds every body node
    in topo order (flat-prologue convention); ``core`` is an annotation
    pointing at specific nodes (ReduceStage.reduce, ContractionCore.mul/
    reduce/post_stages) already in prologue. Backends read the body via
    ``analysis.flat_region_ops(kernel)`` which dedups across slots.

    A KernelOp has one external output (Port) by construction;
    multi-output fusion is a future extension.
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
    # Body / phase views — pure derivations of the structured fields.
    # These are the canonical source for backend readers; analysis.py
    # delegates to them. Keep them free of shape / codegen concerns so
    # they stay Layer-1 appropriate.
    # ------------------------------------------------------------------

    def body_ops(self) -> list:
        """Flat ``(id, op, inputs)`` view of body nodes in topo order.

        Walks prologue + core (ContractionCore.mul/reduce + post_stages,
        or tuple[ReduceStage] pre_ops/reduce) + epilogue, deduped by id.
        """
        seen: set[str] = set()
        out: list = []

        def emit(node) -> None:
            if node is None or node.id in seen:
                return
            seen.add(node.id)
            out.append((node.id, node.op, list(node.inputs)))

        for n in self.prologue:
            emit(n)
        if isinstance(self.core, ContractionCore):
            emit(self.core.mul)
            emit(self.core.reduce)
            for stage in self.core.post_stages:
                if not isinstance(stage, ReduceStage):
                    continue
                for pre in stage.pre_ops:
                    emit(pre)
                emit(stage.reduce)
        elif isinstance(self.core, tuple):
            for stage in self.core:
                if not isinstance(stage, ReduceStage):
                    continue
                for pre in stage.pre_ops:
                    emit(pre)
                emit(stage.reduce)
        for n in self.epilogue:
            emit(n)
        return out

    def phases(self) -> tuple:
        """Return ``(prologue, reduces, inter_reduce, epilogue)`` as
        RegionEntry lists. Each list contains ``(id, op, inputs)`` tuples.

        - ``prologue``: pre-reduce ops; for ContractionCore this includes
          the mul appended after user prologue (mul feeds the K-loop).
        - ``reduces``: one entry per reduce Node in core order.
        - ``inter_reduce[i]``: pre_ops chain that feeds ``reduces[i+1]``.
        - ``epilogue``: post-last-reduce ops.
        """

        def entry(n):
            return (n.id, n.op, list(n.inputs))

        prologue = [entry(n) for n in self.prologue]
        epilogue = [entry(n) for n in self.epilogue]

        if isinstance(self.core, ContractionCore):
            reduces: list = []
            inter_reduce: list = []
            if self.core.mul is not None:
                prologue.append(entry(self.core.mul))
            if self.core.reduce is not None:
                reduces.append(entry(self.core.reduce))
            for stage in self.core.post_stages:
                if not isinstance(stage, ReduceStage):
                    continue
                inter_reduce.append([entry(pn) for pn in stage.pre_ops])
                if stage.reduce is not None:
                    reduces.append(entry(stage.reduce))
            return prologue, reduces, inter_reduce, epilogue

        if isinstance(self.core, tuple) and self.core:
            stages = [s for s in self.core if isinstance(s, ReduceStage)]
            reduces = [entry(s.reduce) for s in stages if s.reduce is not None]
            inter_reduce = [[entry(pn) for pn in s.pre_ops] for s in stages[1:]]
            return prologue, reduces, inter_reduce, epilogue

        return prologue, [], [], epilogue

    def reduce_fn_names(self) -> list:
        """Return the ``.fn`` string of each reduce Node in core order."""
        if isinstance(self.core, ContractionCore):
            names = []
            if self.core.reduce is not None:
                names.append(self.core.reduce.op.fn)
            for stage in self.core.post_stages:
                if isinstance(stage, ReduceStage) and stage.reduce is not None:
                    names.append(stage.reduce.op.fn)
            return names
        if isinstance(self.core, tuple):
            return [s.reduce.op.fn for s in self.core if isinstance(s, ReduceStage) and s.reduce is not None]
        return []

    def port_indexmaps(self) -> dict:
        """Per-input Port.indexmap dict (only inputs with an indexmap set)."""
        return {p.buffer_id: p.indexmap for p in self.inputs if p.indexmap is not None}

    def contraction_info(self, shapes: dict) -> ContractionInfo | None:
        """Resolve M/N/K + batch metadata for a ContractionCore kernel.

        Returns ``None`` if this isn't a contraction or the operand shapes
        are incompatible (K mismatch, sub-2D, batch dims that don't match
        and don't divide cleanly for GQA-style broadcast).
        """
        import math as _math

        if not isinstance(self.core, ContractionCore):
            return None
        a_id = self.core.a.buffer_id
        b_id = self.core.b.buffer_id
        a_shape = shapes.get(a_id)
        b_shape = shapes.get(b_id)
        if a_shape is None or b_shape is None or len(a_shape) < 2 or len(b_shape) < 2:
            return None

        batch_dims: tuple = ()
        batch_size = 1
        a_batch_group = 1
        b_batch_group = 1
        if len(a_shape) > 2 and len(b_shape) > 2:
            a_batch = a_shape[:-2]
            b_batch = b_shape[:-2]
            if a_batch == b_batch:
                batch_dims = a_batch
            else:
                # GQA-style broadcast: pad shorter batch with 1s; each dim
                # must match or one must divide the other.
                max_len = max(len(a_batch), len(b_batch))
                a_padded = (1,) * (max_len - len(a_batch)) + a_batch
                b_padded = (1,) * (max_len - len(b_batch)) + b_batch
                merged_batch: list = []
                for ad, bd in zip(a_padded, b_padded, strict=True):
                    if not isinstance(ad, int) or not isinstance(bd, int):
                        return None
                    if ad == bd:
                        merged_batch.append(ad)
                    elif ad > bd and bd > 0 and ad % bd == 0:
                        merged_batch.append(ad)
                    elif bd > ad and ad > 0 and bd % ad == 0:
                        merged_batch.append(bd)
                    else:
                        return None
                batch_dims = tuple(merged_batch)
                a_bs = _math.prod(d for d in a_padded if isinstance(d, int))
                b_bs = _math.prod(d for d in b_padded if isinstance(d, int))
                if a_bs >= b_bs and b_bs > 0:
                    b_batch_group = a_bs // b_bs
                elif a_bs > 0:
                    a_batch_group = b_bs // a_bs
            batch_size = _math.prod(d for d in batch_dims if isinstance(d, int))
            a_k = a_shape[-1]
            b_k = b_shape[-2]
        else:
            a_k = a_shape[-1]
            b_k = b_shape[0]

        if a_k != b_k:
            return None

        if batch_dims:
            m = a_shape[-2] if isinstance(a_shape[-2], int) else 1
        else:
            m = _math.prod(d for d in a_shape[:-1] if isinstance(d, int)) if any(isinstance(d, int) for d in a_shape[:-1]) else 1
        k = a_k if isinstance(a_k, int) else 1
        n = b_shape[-1] if isinstance(b_shape[-1], int) else 1

        return ContractionInfo(
            a_id=a_id,
            b_id=b_id,
            m=m,
            n=n,
            k=k,
            batch_dims=batch_dims,
            batch_size=batch_size,
            a_batch_group=a_batch_group,
            b_batch_group=b_batch_group,
        )

    def tile_pattern(self, shapes: dict, output_shape: tuple) -> str:
        """Classify the kernel as pointwise / row_reduce / reduce_broadcast / contraction.

        Pure derivation from the structured fields + shapes:
          - no reduces → pointwise.
          - ContractionCore with both operands 2D → contraction.
          - any other reduce shape → row_reduce, or reduce_broadcast when
            the epilogue requires a second per-element pass.
        """
        _prologue, reduces, _inter, _epilogue = self.phases()
        if not reduces:
            return "pointwise"
        cinfo = self.contraction_info(shapes)
        if cinfo is not None:
            access = self.input_accesses(shapes, output_shape)
            a_acc = access.get(cinfo.a_id)
            b_acc = access.get(cinfo.b_id)
            if a_acc and b_acc and a_acc.is_2d and b_acc.is_2d:
                return "contraction"
        return "reduce_broadcast" if self.epilogue_needs_per_element(shapes, output_shape) else "row_reduce"

    def tile_dims(self, shapes: dict, output_shape: tuple) -> tuple:
        """Return ``(rows, cols, k_dim)`` for the kernel's tile schedule.

        - pointwise: rows=1, cols=total output elements, k_dim=0.
        - contraction: M, N, K from ``contraction_info``.
        - row reduce / reduce_broadcast: rows = product of leading dims of
          the pre-reduction tensor, cols = trailing dim, k_dim=cols.
        """
        import math as _math

        _prologue, reduces, _inter, _epilogue = self.phases()
        if not reduces:
            total = _math.prod(d for d in output_shape if isinstance(d, int))
            return 1, total, 0

        cinfo = self.contraction_info(shapes)
        if cinfo is not None:
            access = self.input_accesses(shapes, output_shape)
            a_acc = access.get(cinfo.a_id)
            b_acc = access.get(cinfo.b_id)
            if a_acc and b_acc and a_acc.is_2d and b_acc.is_2d:
                return cinfo.m, cinfo.n, cinfo.k

        first_reduce_input = reduces[0][2][0]
        pre_shape = shapes.get(first_reduce_input, output_shape)
        if len(pre_shape) >= 2:
            rows = _math.prod(d for d in pre_shape[:-1] if isinstance(d, int))
            cols = pre_shape[-1] if isinstance(pre_shape[-1], int) else 1
        else:
            rows = 1
            cols = _math.prod(d for d in pre_shape if isinstance(d, int))
        return rows, cols, cols

    def epilogue_needs_per_element(self, shapes: dict, output_shape: tuple) -> bool:
        """True when the epilogue needs a second per-element pass.

        Holds when any epilogue op (or a prologue op it depends on)
        reads per-element values from a 2D input — e.g. RMSNorm's
        ``mul(x, rsqrt)`` epilogue needs the original ``x`` per element.
        """
        prologue, _reduces, _inter, epilogue = self.phases()
        if not epilogue:
            return False

        epilogue_needs_set = _needed_by_ids(epilogue)
        if any(nid in epilogue_needs_set for nid, _, _ in prologue):
            epilogue_needs_set = epilogue_needs_set | _needed_by_ids(prologue)

        access = self.input_accesses(shapes, output_shape)
        for p in self.inputs:
            if p.buffer_id in epilogue_needs_set:
                acc = access.get(p.buffer_id)
                if acc and acc.is_2d:
                    return True
        for nid, _, _ in prologue:
            if nid in _needed_by_ids(epilogue):
                return True
        return False

    def input_accesses(self, shapes: dict, output_shape: tuple) -> dict:
        """Build an AccessPattern per external input.

        Classification is shape-driven: scalar, row-vector, per-row,
        broadcast (smaller-than-output, NumPy-broadcast-compatible), or 2D.
        """
        import math as _math

        from deplodock.compiler.shape_utils import is_broadcast_compatible

        out_size = _math.prod(d for d in output_shape if isinstance(d, int))
        result: dict = {}
        for p in self.inputs:
            inp = p.buffer_id
            inp_shape = shapes.get(inp, (1,))
            inp_size = _math.prod(d for d in inp_shape if isinstance(d, int))
            has_symbolic = any(isinstance(d, str) for d in inp_shape)
            last_dim = inp_shape[-1] if inp_shape else 1
            last_dim_is_one = isinstance(last_dim, int) and last_dim == 1
            is_per_row = last_dim_is_one and inp_size > 1 and len(inp_shape) >= 2
            is_broadcast = (
                inp_size > 1
                and inp_size < out_size
                and not is_per_row
                and len(inp_shape) >= 2
                and is_broadcast_compatible(inp_shape, output_shape)
            )
            result[inp] = AccessPattern(
                shape=inp_shape,
                size=inp_size,
                is_scalar=(inp_size == 1 and not has_symbolic),
                is_row_vector=(len(inp_shape) == 1 and (inp_size > 1 or has_symbolic)),
                is_2d=(len(inp_shape) >= 2 and (inp_size > 1 or has_symbolic) and not is_per_row and not is_broadcast),
                is_per_row=is_per_row,
                is_broadcast=is_broadcast,
            )
        return result
