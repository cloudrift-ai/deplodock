"""Shared base class for body-carrying IR ops (``LoopOp`` / ``TileOp`` /
``KernelOp``).

Lives inside the ``ir.stmt`` package — and imports its dependencies
(``Body``, ``Load``, ``Write``, ``Stmt``, ``pretty_body``) from their leaf
modules directly rather than via the package ``__init__`` — so that
``ir.stmt.normalize`` (which loads ``Stage`` from ``ir.tile.ir``) doesn't
re-enter ``ir.stmt.__init__`` through the back door when ``tile.ir``
imports ``BodyOp``. The leaf-module imports load cleanly because none of
them touches ``ir.tile``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.base import pretty_body as _pretty_body_stmts
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Load, Write
from deplodock.compiler.tensor import Tensor


@dataclass
class BodyOp(Op):
    """Shared base for IR ops that carry a structured :class:`Body` of
    stmts plus a kernel name — ``LoopOp`` (loop IR), ``TileOp`` (tile IR),
    ``KernelOp`` (kernel IR). Provides:

    - ``body`` / ``name`` fields, plus a ``__post_init__`` that coerces
      tuple-bodies to :class:`Body`,
    - ``__iter__`` over the body's deep stmt iterator,
    - ``loads`` / ``writes`` typed iters,
    - ``body_inputs`` / ``body_outputs`` — distinct global-buffer names
      referenced by body Loads / Writes, in first-use order. Subclasses
      override (``KernelOp`` to include ``CpAsyncCopy.src`` /
      ``TmaDescriptor.src_buf`` and exclude smem buffers; ``TileOp`` to
      include ``Stage.buf`` and exclude staged names),
    - a unified ``pretty_body`` returning the indented body listing
      (the surrounding dump emits the kernel name / I/O label).

    ``__post_init__`` pre-populates ``inputs`` / ``outputs`` (inherited
    from :class:`Op`) with body-derived names keyed to placeholder
    ``Tensor(name, (), F32)`` values, so pre-match callers (lifting /
    fusion / tests) already see the right keys + ordering without a
    matcher run. The matcher's :meth:`populate_io` hook then overrides
    each entry with the real graph-sourced ``Tensor`` and sanity-checks
    that the body has no Load / Write naming a buffer the dict doesn't
    cover."""

    body: Body = field(default_factory=Body)
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body.coerce(self.body)
        self._seed_io_placeholders()

    def _seed_io_placeholders(self) -> None:
        """Populate empty ``inputs`` / ``outputs`` with body-derived names
        keyed to placeholder ``Tensor(name, (), F32)`` values. Skips
        either dict if the caller already supplied entries."""
        if not self.inputs:
            self.inputs = {n: Tensor(n, (), F32) for n in self.body_inputs}
        if not self.outputs:
            self.outputs = {n: Tensor(n, (), F32) for n in self.body_outputs}

    def __iter__(self) -> Iterator[Stmt]:
        return self.body.iter()

    @property
    def loads(self) -> tuple[Load, ...]:
        return self.body.iter_of_type(Load)

    @property
    def writes(self) -> tuple[Write, ...]:
        return self.body.iter_of_type(Write)

    @property
    def body_inputs(self) -> tuple[str, ...]:
        """Distinct ``Load.input`` buf names in body first-use order.
        Override when the op's body has extra global-buffer reference
        stmts (Stage, CpAsyncCopy, TmaDescriptor) or excludes certain
        local names (Stage names, Smem-declared bufs)."""
        return tuple(dict.fromkeys(s.input for s in self.loads))

    @property
    def body_outputs(self) -> tuple[str, ...]:
        """Distinct ``Write.output`` buf names in body first-use order."""
        return tuple(dict.fromkeys(s.output for s in self.writes))

    def populate_io(self, graph, node) -> None:  # noqa: ANN001, ARG002 — matches Op.populate_io signature
        """Override :meth:`Op.populate_io` to key off body-derived buf
        names (which may differ from ``node.inputs`` ordering and may
        include Stage / CpAsyncCopy / TmaDescriptor refs that aren't
        separate graph predecessors). Sanity-checks that every body
        Load / Write names a buffer already covered by ``inputs`` /
        ``outputs`` (caught early — a rule that mutated the body without
        re-seeding I/O surfaces here, not on a later mystery KeyError),
        then replaces each entry's placeholder Tensor with the real
        graph-sourced one when the buffer is a graph node."""
        body_in = self.body_inputs
        stray_in = [n for n in body_in if n not in self.inputs]
        if stray_in:
            raise ValueError(f"{type(self).__name__}: body has Load buffers {stray_in} not covered by inputs={list(self.inputs)}")
        body_out = self.body_outputs
        stray_out = [n for n in body_out if n not in self.outputs]
        if stray_out:
            raise ValueError(f"{type(self).__name__}: body has Write buffers {stray_out} not covered by outputs={list(self.outputs)}")
        for name in self.inputs:
            gnode = graph.nodes.get(name)
            if gnode is not None:
                self.inputs[name] = gnode.output
        for name in self.outputs:
            gnode = graph.nodes.get(name)
            if gnode is not None:
                self.outputs[name] = gnode.output

    def pretty_body(self) -> str:
        """Indented body listing — one stmt per line via per-stmt
        ``pretty``. Callers (``format_kernels`` /
        ``Candidate._format_nodes``) emit the surrounding name / I/O
        label themselves; duplicating it here would just rot."""
        return "\n".join(_pretty_body_stmts(self.body, "    "))
