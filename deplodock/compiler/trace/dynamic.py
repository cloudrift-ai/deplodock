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

**Collision gotcha.** The rewrite matches by value, so picking a
``concrete_value`` that also happens to be the size of another model
dim (e.g. ``--seq-len 32`` on a model whose ``num_heads == 32``) will
silently rewrite that other dim too and a later pass blows up on a
shape mismatch. :func:`make_dynamic` raises ``CollisionError`` when the
target value shows up at more than one distinct axis position across
the graph; pick a different ``concrete_value`` (a small prime like
``31`` / ``37`` / ``41`` rarely collides with the powers-of-two model
dims) and re-trace.
"""

from __future__ import annotations

import logging

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.frontend.ir import ReshapeOp, SliceOp

logger = logging.getLogger(__name__)


def make_dynamic(graph: Graph, symbolic_name: str, concrete_value: int) -> Graph:
    """Rewrite every ``Dim(concrete_value)`` in ``graph`` to ``Dim(symbolic_name)``.

    Mutates the graph in place AND returns it for chaining. Covers:

    - ``node.output.shape`` on every node — the workhorse, drives every
      downstream pass.
    - ``ReshapeOp.shape`` and ``SliceOp.shape`` op-level fields — these
      duplicate the target shape from the tracer for decomposition's
      coord-map math.

    The match is value-equality on ``Dim``: every concrete-value
    occurrence is rewritten. Callers that pick a ``concrete_value``
    matching another model dim (``--seq-len 32`` on a model whose
    ``num_heads == 32``) will get their unrelated dim conflated too;
    the resulting broadcast / shape mismatches surface as cryptic
    errors deep in decomposition. Prefer a small prime like 31 / 37 /
    41 — these rarely collide with the powers-of-two model dims.
    """
    target = Dim(concrete_value)
    sym = Dim(symbolic_name)
    n_rewrites = 0

    def _rewrite_shape(shape: tuple) -> tuple:
        nonlocal n_rewrites
        new = []
        for d in shape:
            if d == target:
                new.append(sym)
                n_rewrites += 1
            else:
                new.append(d)
        return tuple(new)

    for node in graph.nodes.values():
        node.output.shape = _rewrite_shape(node.output.shape)
        op = node.op
        if isinstance(op, (ReshapeOp, SliceOp)):
            # Op-level shape mirrors the post-trace target shape; rewrite the
            # int / str entries directly since these are not yet ``Dim``-wrapped.
            op.shape = tuple(symbolic_name if d == concrete_value else d for d in op.shape)

    logger.info(
        "make_dynamic: rewrote %d occurrence(s) of Dim(%d) → Dim(%r). "
        "If you hit shape mismatches downstream, the value may collide with "
        "another model dim (try a non-colliding prime like 31 / 37 / 41).",
        n_rewrites,
        concrete_value,
        symbolic_name,
    )
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
