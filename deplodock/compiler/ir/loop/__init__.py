"""Loop IR — post-fusion kernel representation plus its analysis/normalization.

Submodules:
- :mod:`.ir` — ``LoopOp`` and its body node types (``Loop``, ``Load``,
  ``Write``, ``Assign``, ``Accum``, ``Select``, ``SelectBranch``, ``Axis``,
  ``Stmt``). Construction runs structural normalization (see
  :mod:`.normalize`) then validation via ``__post_init__``.
- :mod:`.normalize` — pure ``body → body`` passes applied at construction
  (drop size-1 free axes, canonical free-axis order, pointwise
  linearization).

The public surface below re-exports the common types so callers use
``from deplodock.compiler.ir.loop import LoopOp, ...``.
"""

from deplodock.compiler.ir.loop.builder import LoopBuilder
from deplodock.compiler.ir.loop.ir import (
    ACCUM_IDENTITY,
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopMeta,
    LoopOp,
    Scope,
    Select,
    SelectBranch,
    Stmt,
    Write,
    iter_body,
    map_body,
    pretty_print,
)
from deplodock.compiler.ir.loop.splicer import splice_graph, splice_loop_ops, splice_loops

__all__ = [
    "ACCUM_IDENTITY",
    "Accum",
    "Assign",
    "Axis",
    "Load",
    "Loop",
    "LoopBuilder",
    "LoopMeta",
    "LoopOp",
    "Scope",
    "Select",
    "SelectBranch",
    "Stmt",
    "Write",
    "iter_body",
    "map_body",
    "pretty_print",
    "splice_graph",
    "splice_loop_ops",
    "splice_loops",
]
