"""Base Op class and boundary sentinels shared across every IR level.

``Op`` is the root of all tensor operations. ``InputOp`` and ``ConstantOp``
are boundary sentinels that carry tensors into the graph without computing
anything — they appear unchanged from the frontend stage all the way through
to fusion, and the numpy backend supplies their values at runtime.

The shared base for body-carrying ops (``LoopOp`` / ``KernelOp`` /
``TileOp``) lives in :mod:`deplodock.compiler.ir.body_op` — split out to
avoid a circular import with the ``ir.stmt`` package (which imports
``Stage`` from ``ir.tile.ir`` for normalization).

``_keepdim_axis`` is a small shape helper used by both ``ReduceOp`` (minimal IR,
``tensor.py``) and ``MeanOp`` (frontend IR, ``frontend.py``). Keeping it here
avoids a frontend→tensor dependency for a single function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph, Node
    from deplodock.compiler.tensor import Tensor


@dataclass
class Op:
    """Base class for all operations.

    ``source`` is the predecessor op in any rewrite chain — the engine
    stamps it automatically on every 1:1 in-place rebind
    (``_apply_one`` Op branch), so a fully-lowered ``CudaOp`` keeps the
    full path back through ``KernelOp → TileOp → LoopOp`` (or any other
    chain a future pipeline introduces) via repeated ``.source``
    traversals. Keyword-only so positional construction of subclass
    fields keeps working; excluded from :meth:`Graph.structural_key`
    via ``_STRUCTURAL_SKIP_FIELDS`` — pure attribution metadata, not
    part of dataflow identity.

    ``inputs`` / ``outputs`` are per-op-instance buffer maps populated by
    the matcher (``_match_at``) just after a match is built — the matcher
    calls :meth:`populate_io` on every matched node, snapping in real
    :class:`Tensor` values from the surrounding graph so rule rewrites can
    read shapes / dtypes off ``op.inputs[name]`` without re-querying.
    Default: predecessor outputs by predecessor id, this node's output by
    its own id. :class:`BodyOp` overrides to walk body-derived buf names
    instead. Excluded from equality / repr — pure derived view of the
    surrounding graph at match time.
    """

    source: Op | None = field(default=None, kw_only=True, repr=False, compare=False)
    # Free-form metadata dict for rules to stamp the knobs they used
    # (e.g. ``{"BN": 64, "BM": 64}`` from ``005_blockify_launch``). The
    # engine's ``_apply_one`` merges the predecessor's knobs forward on
    # every 1:1 rebind, so a fully-lowered ``CudaOp`` carries every
    # autotune knob picked along the chain. Excluded from structural
    # identity and equality — pure attribution metadata.
    knobs: dict = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    inputs: dict[str, Tensor] = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    outputs: dict[str, Tensor] = field(default_factory=dict, kw_only=True, repr=False, compare=False)

    def populate_io(self, graph: Graph, node: Node) -> None:
        """Refresh ``inputs`` / ``outputs`` from the surrounding graph.
        Called by the matcher after a match is built. Default mapping
        works for non-body ops (one Tensor per predecessor / this node).
        :class:`BodyOp` overrides to walk body-derived buf names."""
        self.inputs = {pid: graph.nodes[pid].output for pid in node.inputs if pid in graph.nodes}
        self.outputs = {node.id: node.output}

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Derive the output shape from input shapes. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.infer_output_shape not implemented")

    def forward(self, *inputs):
        """Compute the operation using numpy arrays. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.forward not implemented")

    def pretty_body(self) -> str | None:
        """Render this op's body for kernel dumps. Default: None (skip)."""
        return None

    def validate(self, ctx) -> bool:  # noqa: ARG002 — ``ctx`` consumed by subclass overrides
        """Sanity-check this op against the compilation context. Return
        ``False`` to signal that the engine should *drop* this op (e.g.
        skip a fork variant that would produce an unrunnable kernel).
        Default: always valid. Override in subclasses to enforce per-op
        invariants — ``TileOp.validate`` for instance checks that the
        post-register-tile thread count fits the hardware launch budget.
        """
        return True

    def score(self, ctx) -> float | None:  # noqa: ARG002 — ``ctx`` consumed by subclass overrides
        """Heuristic "promisingness" prior for autotune candidate
        ordering. Higher = explore first. Returned without bounds; the
        search uses it only as a tiebreaker among unvisited siblings.
        Default: ``None`` (no prior available — the op does not
        contribute to the candidate's mean score). Override in
        subclasses with domain knowledge (``TileOp.score`` prefers CTAs
        near the target thread count that stage their inputs and
        register-tile, for example).
        """
        return None


@dataclass
class InputOp(Op):
    """Sentinel for graph input tensors (no computation)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("InputOp has no inputs; use node.output.shape directly")

    def forward(self, *inputs):
        raise NotImplementedError("InputOp is a sentinel; value is supplied by the executor")


@dataclass
class ConstantOp(Op):
    """Fixed tensor: weights, RoPE tables, scalars. Not an activation.

    ``load_ops`` is an ordered tuple of frontend ``Op`` instances applied
    to the source tensor at bind time, in order. Const-folding passes
    absorb a foldable op (``TransposeOp``, ``ReshapeOp``, ...) into this
    chain, so a chain of layout ops over a constant collapses to a single
    ``ConstantOp`` whose loader executes the chain via the reference
    NumPy backend. ``output.shape`` already reflects the post-chain shape.

    ``source_path`` / ``source_shape`` / ``source_dtype`` are the source
    tensor's address (HF parameter/buffer attribute path) and pre-chain
    layout — what the loader needs to read from safetensors before
    running ``load_ops``. Empty for scalar constants and for synthetic
    constants emitted by passes (which never reach the loader).
    """

    name: str
    value: float | None = None  # scalar value captured at trace time
    load_ops: tuple[Op, ...] = ()
    source_path: str | None = None
    source_shape: tuple[int, ...] | None = None
    source_dtype: str | None = None

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        raise NotImplementedError("ConstantOp has no inputs; use node.output.shape directly")

    def forward(self, *inputs):
        if self.value is not None:
            return np.array([self.value], dtype=np.float32)
        raise NotImplementedError("ConstantOp with value=None must be supplied by the executor")


def _keepdim_axis(shape: tuple, axis: int | str) -> tuple:
    """Return shape with the given axis set to 1 (keepdim=True semantics).

    This is the reduction's output shape: the reduced dim collapses to 1,
    preserving rank. Rank-preservation is a Tensor IR invariant — it lets
    elementwise ops consume reduction outputs without any implicit reshape
    (see ``pipeline/passes/frontend/decomposition/_broadcast.py``'s ``broadcast_to`` for the explicit
    broadcast wrapper that expands (…, 1, …) back up).
    """
    if not isinstance(axis, int):
        return tuple(shape)
    a = axis if axis >= 0 else len(shape) + axis
    if a < 0 or a >= len(shape):
        return tuple(shape)
    return tuple(shape[:a]) + (1,) + tuple(shape[a + 1 :])
