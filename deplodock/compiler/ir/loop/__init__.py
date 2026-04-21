"""Loop IR — post-fusion kernel representation plus its analysis/normalization.

Submodules:
- :mod:`.ir` — ``LoopOp`` and its body node types (``Loop``, ``Load``,
  ``Write``, ``Assign``, ``Accum``, ``Select``, ``SelectBranch``, ``Axis``,
  ``Stmt``). Construction runs structural normalization (see
  :mod:`.normalize`) then validation via ``__post_init__``.
- :mod:`.plan` — ``analyze_kernel`` lowers a ``LoopOp`` to a ``KernelPlan``
  (iteration spaces, reduce segments, rematerialization) for codegen.
- :mod:`.normalize` — pure ``body → body`` passes applied at construction
  (drop size-1 free axes, canonical free-axis order, pointwise
  linearization).

The public surface below re-exports the common types so callers use
``from deplodock.compiler.ir.loop import LoopOp, ...``. For the
plan-module's ``Loop`` and ``Inline`` (which collide with ``ir.Loop``),
import from :mod:`.plan` explicitly.
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
    flat_body_to_nested,
    flatten_body,
    iter_loops,
    pretty_print,
)
from deplodock.compiler.ir.loop.plan import KernelPlan, analyze_kernel

__all__ = [
    "ACCUM_IDENTITY",
    "Accum",
    "Assign",
    "Axis",
    "KernelPlan",
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
    "analyze_kernel",
    "flat_body_to_nested",
    "flatten_body",
    "iter_loops",
    "pretty_print",
]
