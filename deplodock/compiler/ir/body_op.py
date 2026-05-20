"""Shared base class for body-carrying IR ops (``LoopOp`` / ``TileOp`` /
``KernelOp``).

Lives in its own module to dodge a circular import: the ``ir.stmt``
package's ``normalize`` submodule imports ``Stage`` from ``ir.tile.ir``,
so ``ir.tile.ir`` can't import ``BodyOp`` from any module that triggers
``ir.stmt.__init__`` at module-load time. The Body / Load / Write / Stmt
/ pretty_body symbols ``BodyOp`` needs are imported lazily inside methods
(and via :data:`TYPE_CHECKING` for annotations) to avoid that cycle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.ir.base import Op

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.base import Stmt
    from deplodock.compiler.ir.stmt.body import Body
    from deplodock.compiler.ir.stmt.leaves import Load, Write


def _empty_body() -> Body:
    from deplodock.compiler.ir.stmt.body import Body as _Body

    return _Body()


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
    - a unified ``pretty_body`` printing a ``kernel <name>  inputs: ...
      outputs: ...`` header above the indented body listing.
    """

    body: Body = field(default_factory=_empty_body)
    name: str = ""

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.stmt.body import Body as _Body  # noqa: PLC0415

        if not isinstance(self.body, _Body):
            self.body = _Body.coerce(self.body)

    def __iter__(self) -> Iterator[Stmt]:
        return self.body.iter()

    @property
    def loads(self) -> tuple[Load, ...]:
        from deplodock.compiler.ir.stmt.leaves import Load as _Load  # noqa: PLC0415

        return self.body.iter_of_type(_Load)

    @property
    def writes(self) -> tuple[Write, ...]:
        from deplodock.compiler.ir.stmt.leaves import Write as _Write  # noqa: PLC0415

        return self.body.iter_of_type(_Write)

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

    def pretty_body(self) -> str:
        """``kernel <name>  inputs: ...  outputs: ...`` header above the
        indented body listing."""
        from deplodock.compiler.ir.stmt.base import pretty_body as _pretty_body_stmts  # noqa: PLC0415

        sig_in = ", ".join(self.body_inputs) or "-"
        sig_out = ", ".join(self.body_outputs) or "-"
        head = f"kernel {self.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}"
        return "\n".join([head, *_pretty_body_stmts(self.body, "    ")])
