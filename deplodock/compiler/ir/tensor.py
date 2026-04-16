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
Fusion / ``assemble_kernels`` then folds this IR into ``ir.block.KernelOp``
nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.base import Op, _drop_axis

# ---------------------------------------------------------------------------
# Op metadata registries
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


# ---------------------------------------------------------------------------
# Elementwise / reduce / scan
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gather / scatter
# ---------------------------------------------------------------------------


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

    out_shape: tuple[int, ...]
    sources: tuple[IndexSource, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.out_shape)

    def forward(self, *inputs):
        import numpy as np

        from deplodock.compiler.ir.expr import BinOp, Literal, Ternary, Var

        def _eval(expr, env):
            if isinstance(expr, Var):
                return env[expr.name]
            if isinstance(expr, Literal):
                return expr.value
            if isinstance(expr, BinOp):
                lv, r = _eval(expr.left, env), _eval(expr.right, env)
                if expr.op == "+":
                    return lv + r
                if expr.op == "-":
                    return lv - r
                if expr.op == "*":
                    return lv * r
                if expr.op in ("/", "//"):
                    return int(lv) // int(r)
                if expr.op == "%":
                    return int(lv) % int(r)
                if expr.op == "<":
                    return lv < r
                if expr.op == ">=":
                    return lv >= r
                if expr.op == "==":
                    return lv == r
                if expr.op == "&&":
                    return bool(lv) and bool(r)
                if expr.op == "||":
                    return bool(lv) or bool(r)
                raise ValueError(f"Unknown BinOp: {expr.op}")
            if isinstance(expr, Ternary):
                return _eval(expr.if_true, env) if _eval(expr.cond, env) else _eval(expr.if_false, env)
            raise TypeError(f"Unsupported expr type in IndexMapOp.forward: {type(expr).__name__}")

        shape = tuple(int(d) for d in self.out_shape)
        output = np.empty(shape, dtype=inputs[0].dtype if inputs else np.float32)
        for out_idx in np.ndindex(shape):
            env = {f"out_coord_{i}": out_idx[i] for i in range(len(out_idx))}
            for source in self.sources:
                if source.select is not None and not _eval(source.select, env):
                    continue
                in_coords = tuple(int(_eval(c, env)) for c in source.coord_map)
                output[out_idx] = inputs[source.input_idx][in_coords]
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
