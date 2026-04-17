"""Base Op class and boundary sentinels shared across every IR level.

``Op`` is the root of all tensor operations. ``InputOp`` and ``ConstantOp``
are boundary sentinels that carry tensors into the graph without computing
anything — they appear unchanged from the frontend stage all the way through
to fusion, and the numpy backend supplies their values at runtime.

``_drop_axis`` is a small shape helper used by both ``ReduceOp`` (minimal IR,
``tensor.py``) and ``MeanOp`` (frontend IR, ``frontend.py``). Keeping it here
avoids a frontend→tensor dependency for a single function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Op:
    """Base class for all operations."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Derive the output shape from input shapes. Override in subclasses."""
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


def _drop_axis(shape: tuple, axis: int | str) -> tuple:
    """Return shape with the given axis set to 1 (keepdim=True semantics).

    Keeping the reduced dim as size-1 ensures that post-reduce values
    broadcast correctly with pre-reduce values (e.g. RMSNorm's
    ``mul(X:(M,K), rsqrt:(M,1))`` broadcasts to ``(M,K)``).
    """
    if not isinstance(axis, int):
        return tuple(shape)
    a = axis if axis >= 0 else len(shape) + axis
    if a < 0 or a >= len(shape):
        return tuple(shape)
    return tuple(shape[:a]) + (1,) + tuple(shape[a + 1 :])
