"""Shared statement primitives — the leaves and control flow used across
every IR layer.

Defined here rather than under any one IR package because all three IRs
(Loop, Tile, Kernel) consume the same leaf vocabulary:

- ``Stmt`` — abstract base for every body statement.
- Leaves: ``Load``, ``Assign``, ``Accum``, ``Mma``, ``Init``, ``Write``,
  ``Select``, ``SelectBranch`` — pure compute primitives that read/write SSA
  names and external buffers (in :mod:`.leaves`). ``Mma`` is the tensor-core
  fused multiply-accumulate (``c += a @ b``); a fragment ``Load``'s ``dtype``
  is a ``FragmentType`` (carrying the atom kind + role) — both lowered by
  ``kernel/005_lower_atom_tile``.
- Block stmts: ``Loop``, ``StridedLoop``, ``Cond`` — carry child bodies
  (in :mod:`.blocks`).
- Tree walks: :meth:`Body.iter` (pre-order recursive) and
  :meth:`Body.map` (flat 1:N transformer) — methods on
  :class:`Body` (in :mod:`.body`).
- Body normalization: ``normalize_body`` driver + 8 passes
  (drop-size-one, canonicalize-axis-order, copy-alias-elim,
  reduce-axis-unify, hoist, simplify, dedup-loads, rename-ssa) in
  :mod:`.normalize`.
- Pretty printing + render context: ``RenderCtx``, ``op_to_expr``,
  ``select_to_ternary``, ``render_index`` (in :mod:`.base`).

Each IR layer adds its own scheduling-specific Stmts on top:

- Loop IR: nothing extra — its bodies are exactly Loop / leaves.
- Tile IR: ``Stage``, ``Combine``, plus the shared ``Loop`` / ``StridedLoop``
  constructs from this module and the typed tile flavors (``GridTile``,
  ``ThreadTile``, ``RegisterTile``, ``SerialTile``, ``StridedTile``) from
  :mod:`deplodock.compiler.ir.tile.ir`.
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
from deplodock.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Mma, Pack, Select, SelectBranch, Unpack, Write
from deplodock.compiler.ir.stmt.normalize import (
    canonicalize_buffer_names,
    canonicalize_free_axis_order,
    dedup_loads,
    drop_size_one_free_axes,
    eliminate_copy_aliases,
    hoist_loop_invariants,
    normalize_body,
    rename_ssa_sequential,
    simplify_body,
    sort_commutative_args,
    unify_sibling_reduce_axes,
)

__all__ = [
    "INDENT",
    "Accum",
    "Assign",
    "Body",
    "Cond",
    "Init",
    "Load",
    "Loop",
    "Mma",
    "Pack",
    "RenderCtx",
    "Select",
    "SelectBranch",
    "Stmt",
    "StridedLoop",
    "Unpack",
    "Write",
    "canonicalize_buffer_names",
    "canonicalize_free_axis_order",
    "dedup_loads",
    "drop_size_one_free_axes",
    "eliminate_copy_aliases",
    "hoist_loop_invariants",
    "normalize_body",
    "op_to_expr",
    "pretty_body",
    "rename_ssa_sequential",
    "render_body",
    "render_index",
    "select_to_ternary",
    "simplify_body",
    "sort_commutative_args",
    "unify_sibling_reduce_axes",
]
