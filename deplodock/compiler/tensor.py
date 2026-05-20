"""``Tensor`` — symbolic descriptor of a tensor-shaped buffer.

Holds the three things every consumer asks for: a name, a shape, and a
:class:`DataType`. Reused as the per-node ``Node.output`` value in the
graph, as the per-buffer descriptor on ``KernelOp`` (kernel signature),
and as the render-time ``tensors`` map for index flattening.

``dtype`` accepts a :class:`DataType` directly or any string spelling
that :func:`deplodock.compiler.dtype.get` resolves (canonical name,
PyTorch alias, etc.); ``__post_init__`` coerces to the canonical
:class:`DataType` so downstream code never sees a bare string.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.dtype import get as _get_dtype


@dataclass
class Tensor:
    """Multidimensional array descriptor.

    ``constant`` marks tensors whose value is fixed at compile time
    (``ConstantOp.output`` — weights, RoPE tables, scalar literals).
    ``value`` carries the captured scalar when the constant is a
    0-D float (``ConstantOp.value is not None``); otherwise ``None``.
    Together they let downstream consumers (cuda lowering, the load
    vectorizer) recognize a scalar-literal buffer without re-querying
    the graph for ``ConstantOp`` predecessors.
    """

    name: str
    shape: tuple[int | str, ...]  # concrete ints or symbolic dim names
    dtype: DataType = F32
    constant: bool = False
    value: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DataType):
            self.dtype = _get_dtype(self.dtype)
