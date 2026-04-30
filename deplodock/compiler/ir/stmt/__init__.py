"""Shared statement primitives — the leaves and control flow used across
every IR layer.

Defined here rather than under any one IR package because all three IRs
(Loop, Tile, Kernel) consume the same leaf vocabulary:

- ``Stmt`` — abstract base for every body statement.
- Leaves: ``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``, ``Select``,
  ``SelectBranch`` — pure compute primitives that read/write SSA names
  and external buffers (in :mod:`.leaves`).
- Block stmts: ``Loop``, ``Tile``, ``StridedLoop``, ``Cond`` — carry
  child bodies (in :mod:`.blocks`).
- Tree walks: ``iter_body``, ``map_body`` (in :mod:`.visit`).
- Body normalization: ``normalize_body`` driver + 8 passes
  (drop-size-one, canonicalize-axis-order, copy-alias-elim,
  reduce-axis-unify, hoist, simplify, dedup-loads, rename-ssa) in
  :mod:`.normalize`.
- Pretty printing + render context: ``RenderCtx``, ``op_to_expr``,
  ``select_to_ternary``, ``render_index`` (in :mod:`.base`).

Each IR layer adds its own scheduling-specific Stmts on top:

- Loop IR: nothing extra — its bodies are exactly Loop / leaves.
- Tile IR: ``Stage``, ``Combine``, plus the shared ``Tile`` /
  ``Loop`` / ``StridedLoop`` constructs from this module.
- Kernel IR: ``Smem``, ``Sync``, ``TreeHalve``, plus the shared
  constructs.

Loop-IR's ``LoopOp``, ``LoopMeta``, validation, and Loop-IR-specific
canonicalization stay in ``ir/loop/`` because they're Loop-IR-internal —
they enforce Loop-IR's invariants (SSA scoping rules, axis uniqueness)
and produce Loop-IR's canonical form.
"""

from deplodock.compiler.ir.stmt.base import (
    INDENT,
    RenderCtx,
    Stmt,
    op_to_expr,
    pretty_body,
    render_body,
    render_index,
    select_to_ternary,
)
from deplodock.compiler.ir.stmt.base import (
    _axis_identity as _axis_identity,  # re-export for ir.tile.ir / ir.kernel.ir
)
from deplodock.compiler.ir.stmt.base import (
    _pad as _pad,  # re-export for ir.kernel.ir
)
from deplodock.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop, Tile
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Select, SelectBranch, Write
from deplodock.compiler.ir.stmt.normalize import (
    canonicalize_free_axis_order,
    dedup_loads,
    drop_size_one_free_axes,
    eliminate_copy_aliases,
    hoist_loop_invariants,
    normalize_body,
    rename_ssa_sequential,
    simplify_body,
    unify_sibling_reduce_axes,
)
from deplodock.compiler.ir.stmt.visit import iter_body, map_body

__all__ = [
    "INDENT",
    "Accum",
    "Assign",
    "Cond",
    "Init",
    "Load",
    "Loop",
    "RenderCtx",
    "Select",
    "SelectBranch",
    "Stmt",
    "StridedLoop",
    "Tile",
    "Write",
    "canonicalize_free_axis_order",
    "dedup_loads",
    "drop_size_one_free_axes",
    "eliminate_copy_aliases",
    "hoist_loop_invariants",
    "iter_body",
    "map_body",
    "normalize_body",
    "op_to_expr",
    "pretty_body",
    "rename_ssa_sequential",
    "render_body",
    "render_index",
    "select_to_ternary",
    "simplify_body",
    "unify_sibling_reduce_axes",
]
