"""Post-trace rewrite that swaps a concrete shape dim for a symbolic ``Dim``.

The PyTorch tracer only sees concrete shapes (``torch.export`` consumes
example inputs of fixed size). The dynamic-shapes path in
``plans/dynamic-shapes.md`` uses a two-step recipe:

1. Trace the model at one canonical seq_len with ``trace_module``.
2. Walk the resulting graph and rewrite every ``Dim(<canonical>)`` at
   the marked axis position to ``Dim(<symbolic_name>)`` via
   :func:`make_dynamic`.

The output graph compiles once (CudaOp source carries
``int <symbolic_name>`` runtime arg) and runs at any seq_len whose
value flows in via ``input_data``.
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.frontend.ir import ReshapeOp, SliceOp


def make_dynamic(graph: Graph, symbolic_name: str, concrete_value: int) -> Graph:
    """Rewrite every ``Dim(concrete_value)`` in ``graph`` to ``Dim(symbolic_name)``.

    Mutates the graph in place AND returns it for chaining. Covers:

    - ``node.output.shape`` on every node — the workhorse, drives every
      downstream pass.
    - ``ReshapeOp.shape`` and ``SliceOp.shape`` op-level fields — these
      duplicate the target shape from the tracer for decomposition's
      coord-map math.

    The match is value-equality on ``Dim``: every concrete-value
    occurrence (across any axis position) is rewritten. Callers
    constructing graphs with multiple distinct dims of the same value
    should pick a unique sentinel value for the dim being made dynamic.
    """
    target = Dim(concrete_value)
    sym = Dim(symbolic_name)

    def _rewrite_shape(shape: tuple) -> tuple:
        return tuple(sym if d == target else d for d in shape)

    for node in graph.nodes.values():
        node.output.shape = _rewrite_shape(node.output.shape)
        op = node.op
        if isinstance(op, (ReshapeOp, SliceOp)):
            # Op-level shape mirrors the post-trace target shape; rewrite the
            # int / str entries directly since these are not yet ``Dim``-wrapped.
            op.shape = tuple(symbolic_name if d == concrete_value else d for d in op.shape)
    return graph


__all__ = ["make_dynamic"]
