"""Minimal tensor IR — the dialect that survives decomposition.

After decomposition rewrites the frontend ops (``LinearOp``, ``MatmulOp``,
``SdpaOp``, ``MeanOp``, ``UnsqueezeOp``, ``TransposeOp``, ``ReshapeOp``,
``SliceOp``, ``CatOp``) into their primitives, only this set of ops should
remain in the graph:

- ``ElementwiseOp`` — scalar function per element (add, mul, exp, silu, ...).
- ``ReduceOp`` — collapse one axis via an associative binary op.
- ``ScanOp`` — cumulative variant of ``ReduceOp``.
- ``GatherOp`` / ``ScatterOp`` — data-dependent reads/writes along an axis.
- ``IndexMapOp`` — unified layout-only op (subsumes slice / cat / transpose
  / reshape / unsqueeze) described by affine coord arithmetic over
  placeholder vars from ``ir.expr``.

Plus the boundary sentinels ``InputOp`` and ``ConstantOp`` from ``ir.base``.
The ``lifting/`` pass wraps each tensor op in a trivial ``ir.loop.LoopOp``
and the ``fusion/`` pass splices adjacent LoopOp pairs via the
tree-splicer in ``ir/loop/splicer.py``.

Op metadata (arity / commutative / reducer identity) lives on
``ir.expr.ExprOp`` — the single source of truth shared across
elementwise, reduce, scan, and accumulator use sites. The old
``OpInfo`` / ``ReduceInfo`` registries are gone; read straight from
``op.op.arity`` / ``op.op.identity`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op, _keepdim_axis
from deplodock.compiler.ir.expr import ExprOp, coerce_expr_op

# ---------------------------------------------------------------------------
# Elementwise / reduce / scan
# ---------------------------------------------------------------------------


@dataclass
class ElementwiseOp(Op):
    """Apply a scalar function independently to each element.

    The ``op`` field is an ``ExprOp`` carrying the function's name +
    arity + commutativity + (for reducer use) identity. A plain string
    is accepted by ``__post_init__`` and resolved to the canonical
    ``ExprOp`` via the registry in ``ir.expr``.
    """

    op: ExprOp = field(default_factory=lambda: coerce_expr_op("copy"))

    def __post_init__(self) -> None:
        # Accept a string name for back-compat with ``ElementwiseOp("add")``
        # / ``ElementwiseOp(op="add")``-style construction; resolve to the
        # canonical ExprOp so the rest of the pipeline sees structured
        # metadata.
        object.__setattr__(self, "op", coerce_expr_op(self.op))

    @property
    def name(self) -> str:
        """String name of the inner ExprOp — convenient for readers + tests."""
        return self.op.name

    @property
    def fn(self) -> str:
        """Alias for ``name`` — kept for pattern-matcher ``constraints={"fn": ...}``."""
        return self.op.name

    @property
    def arity(self) -> int:
        return self.op.arity

    @property
    def commutative(self) -> bool:
        return self.op.commutative

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Elementwise is rank-preserving with no implicit broadcasting:
        every input must have shape equal to the output. Broadcasts must be
        expressed as explicit ``IndexMapOp`` wrappers upstream (the
        decomposition rules use the ``broadcast_to`` helper for this).
        """
        if not input_shapes:
            return ()
        head = tuple(input_shapes[0])
        for s in input_shapes[1:]:
            if tuple(s) != head:
                shapes_fmt = [tuple(s) for s in input_shapes]
                raise ValueError(
                    f"ElementwiseOp({self.op.name!r}) input shapes must all match output; "
                    f"got {shapes_fmt}. Wrap in IndexMapOp (pipeline/passes/frontend/decomposition/_broadcast.broadcast_to)."
                )
        return head

    def forward(self, *inputs):
        # No shape check here — inside a LoopOp body, forward is called
        # per-iteration on scalar values, so a tensor-level match assert
        # doesn't apply. infer_output_shape enforces it at the graph level.
        return self.op.fn(*inputs)


@dataclass
class ReduceOp(Op):
    """Collapse one or more dimensions via an associative binary op.

    ``op`` is the combine (``sum`` / ``max`` / ``prod`` / …); ``axis`` is
    the reduced dimension (concrete int or symbolic name).
    """

    op: ExprOp = field(default_factory=lambda: coerce_expr_op("sum"))
    axis: int | str = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", coerce_expr_op(self.op))

    @property
    def name(self) -> str:
        return self.op.name

    @property
    def fn(self) -> str:
        return self.op.name

    @property
    def identity(self) -> float:
        return self.op.identity if self.op.identity is not None else 0.0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _keepdim_axis(input_shapes[0], self.axis)

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        name = self.op.name
        if name in ("sum", "add"):
            return np.sum(a, axis=self.axis, keepdims=True)
        if name in ("prod", "multiply"):
            return np.prod(a, axis=self.axis, keepdims=True)
        if name in ("maximum", "amax", "fmax"):
            return np.max(a, axis=self.axis, keepdims=True)
        if name in ("minimum", "fmin"):
            return np.min(a, axis=self.axis, keepdims=True)
        raise NotImplementedError(f"ReduceOp.forward: unknown fn {name!r}")


@dataclass
class ScanOp(Op):
    """Cumulative application of an associative binary op along an axis."""

    op: ExprOp = field(default_factory=lambda: coerce_expr_op("sum"))
    axis: int | str = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", coerce_expr_op(self.op))

    @property
    def name(self) -> str:
        return self.op.name

    @property
    def fn(self) -> str:
        return self.op.name

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(input_shapes[0])  # scan preserves shape

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        name = self.op.name
        if name in ("sum", "add"):
            return np.cumsum(a, axis=self.axis)
        if name in ("maximum", "fmax"):
            return np.maximum.accumulate(a, axis=self.axis)
        if name in ("prod", "multiply"):
            return np.cumprod(a, axis=self.axis)
        raise NotImplementedError(f"ScanOp.forward: unknown fn {name!r}")


# ---------------------------------------------------------------------------
# Gather / scatter
# ---------------------------------------------------------------------------


@dataclass
class GatherOp(Op):
    """Read elements from arbitrary positions along an axis."""

    axis: int | str = 0

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

    axis: int | str = 0
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
# Unified layout op (subsumes Slice/Cat/Transpose/Reshape/Unsqueeze)
# ---------------------------------------------------------------------------


@dataclass
class IndexSource:
    """One input source for an IndexMapOp.

    ``coord_map[i]`` is an ``Expr`` producing the input's i-th index from
    placeholder vars ``Var("out_coord_0")``, ``Var("out_coord_1")``, ...
    See ``deplodock.compiler.ir.expr`` for the placeholder convention and
    substitution helpers.

    ``select`` is None for single-source ops; for multi-source IndexMaps
    (cat) it's a boolean ``Expr`` selecting which output positions read
    this source.
    """

    input_idx: int  # position in IndexMapOp's input list
    coord_map: tuple  # tuple[Expr, ...] — kept untyped to avoid forward-reference clutter
    select: object | None = None  # Expr | None


@dataclass
class IndexMapOp(Op):
    """Compute output by reindexing inputs via affine coord arithmetic.

    Subsumes Slice, Cat, Transpose, Reshape, Unsqueeze — every layout-only
    op is a function from output coordinates to input coordinates.
    Multi-source forms (cat) use ``select`` on each source to pick which
    output positions read which input.
    """

    out_shape: tuple[int, ...] = ()
    sources: tuple[IndexSource, ...] = ()

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.out_shape)

    def forward(self, *inputs):
        import numpy as np

        shape = tuple(int(d) for d in self.out_shape)
        output = np.empty(shape, dtype=inputs[0].dtype if inputs else np.float32)
        for out_idx in np.ndindex(shape):
            env = {f"out_coord_{i}": out_idx[i] for i in range(len(out_idx))}
            for source in self.sources:
                if source.select is not None and not source.select.eval(env):
                    continue
                in_coords = tuple(int(c.eval(env)) for c in source.coord_map)
                input_tensor = inputs[source.input_idx]
                # Clip coords to valid range. After fusion, a Load's IndexMap
                # may produce out-of-bounds coords when the consuming Select
                # masks the range to another branch — reading garbage is
                # safe because the value is never used. CUDA emits direct
                # reads without bounds-checking for the same reason.
                clipped = tuple(max(0, min(c, input_tensor.shape[i] - 1)) for i, c in enumerate(in_coords))
                output[out_idx] = input_tensor[clipped]
                break
        return output

    def is_identity(self, input_shape: tuple) -> bool:
        """True when this IndexMap is a pure pointer alias of its single input."""
        from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Var

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
