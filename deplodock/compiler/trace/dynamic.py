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


def parse_specs(specs: list[str] | None, *, default_value: int) -> list[tuple[str, int]]:
    """Parse ``--dynamic NAME[=VALUE]`` CLI strings to ``(name, value)`` pairs.

    Bare ``NAME`` (no ``=``) uses ``default_value`` — typically
    ``args.seq_len`` for the most common ``--dynamic seq_len`` shorthand.
    Explicit ``NAME=VALUE`` requires ``VALUE`` to be a positive int.

    Raises ``ValueError`` with a CLI-friendly message on a bad spec; the
    caller is expected to ``sys.exit(2)`` so the failure surfaces as a
    usage error rather than a stack trace.
    """
    out: list[tuple[str, int]] = []
    if not specs:
        return out
    seen: set[str] = set()
    for raw in specs:
        if "=" in raw:
            name, _, value_str = raw.partition("=")
            name = name.strip()
            try:
                value = int(value_str)
            except ValueError as e:
                raise ValueError(f"--dynamic {raw!r}: VALUE must be a positive int, got {value_str!r}") from e
        else:
            name = raw.strip()
            value = default_value
        if not name:
            raise ValueError(f"--dynamic {raw!r}: NAME is empty")
        if value <= 0:
            raise ValueError(f"--dynamic {raw!r}: VALUE must be > 0, got {value}")
        if name in seen:
            raise ValueError(f"--dynamic {raw!r}: NAME {name!r} appears more than once")
        seen.add(name)
        out.append((name, value))
    return out


def apply_specs(graph: Graph, specs: list[tuple[str, int]]) -> Graph:
    """Apply a list of ``(name, value)`` rewrites in order — convenience
    wrapper so callers don't need to loop over :func:`make_dynamic`."""
    for name, value in specs:
        make_dynamic(graph, name, value)
    return graph


__all__ = ["make_dynamic", "parse_specs", "apply_specs"]
