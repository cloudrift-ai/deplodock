"""Masked-tile index arithmetic — the single home for the edge-clamp, the
zero-fill predicate, and the symbolic-K locator.

A masked tile (a non-divisor / symbolic output axis, or a symbolic reduce axis)
tiles past its real extent, so the overhang reads must be made safe:

- **Edge-clamp** (output M/N mask): ``coord < bound ? coord : bound-1`` — a
  harmless in-bounds *duplicate*. The masked output cell it feeds is never
  written (the boundary ``Cond`` drops it), so the wrong value is dropped.
  :func:`mask_index` builds this; used by the gmem-direct read clamp
  (``kernel/009``) and the staged cooperative-load clamp (``kernel/_stage_expand``).
- **Zero-fill** (symbolic-K reduce mask): the read must STILL be in-bounds (so
  the address is edge-clamped too), but the loaded *value* is then zeroed where
  the K coord runs past the runtime extent — a clamped duplicate would corrupt
  the reduction. The value-zeroing ``Select`` is the caller's (it needs a typed
  zero); this module owns the in-bounds predicate (:func:`in_bounds`) that gates
  it and the :func:`locate_symbolic_k` locator that finds which staged source
  dim carries the symbolic contraction extent.

The warp/MMA tier (``kernel/005_lower_atom_tile``) renders its own M/N clamp and
K zero-fill directly in CUDA C from ``(coord, bound)`` tuples on the ``Mma`` /
``LdmatrixLoad`` / ``RegStore`` — it builds no ``Expr`` clamps, so it does not
route through here; it shares only the *concept*, not the construction.
"""

from __future__ import annotations

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr


def ext_expr(ext: int | Expr) -> Expr:
    """An extent as an ``Expr``: static ints become ``Literal``s, symbolic
    extents (e.g. ``Var('seq_len')``) pass through and render against the
    runtime kernel arg."""
    return Literal(ext, "int") if isinstance(ext, int) else ext


def ext_minus_one(ext: int | Expr) -> Expr:
    """``ext - 1``, folded to a single ``Literal`` for static extents."""
    return Literal(ext - 1, "int") if isinstance(ext, int) else BinaryExpr("-", ext, Literal(1, "int"))


def is_symbolic_extent(ext: int | Expr) -> bool:
    """True iff ``ext`` is a runtime-sized (symbolic) extent rather than a static int."""
    return not isinstance(ext, int)


def in_bounds(coord: Expr, bound: int | Expr) -> BinaryExpr:
    """The boundary predicate ``coord < bound`` — the in-bounds test a masked
    tile gates its store / zero-fill on."""
    return BinaryExpr("<", coord, ext_expr(bound))


def mask_index(coord: Expr, bound: int | Expr, mode: str = "clamp") -> Expr:
    """Edge-clamp a tile coord to a safe in-bounds read: ``coord < bound ? coord
    : bound-1``.

    Both masking modes clamp the *address* identically — the read must land in
    the buffer either way. ``mode`` records the caller's intent:

    - ``"clamp"`` (output M/N mask): the duplicate value feeds a masked output
      cell the boundary ``Cond`` never writes, so it is simply dropped.
    - ``"zero"`` (symbolic-K reduce mask): the in-bounds read is correct, but the
      loaded value must additionally be zeroed past the extent (the caller wraps
      it in a ``Select`` gated on :func:`in_bounds`) so the reduction accumulates
      ``0``, not a duplicate.

    The returned index ternary is the same for both modes; the value handling is
    the caller's."""
    if mode not in ("clamp", "zero"):
        raise ValueError(f"mask_index mode must be 'clamp' or 'zero', got {mode!r}")
    return TernaryExpr(cond=in_bounds(coord, bound), if_true=coord, if_false=ext_minus_one(bound))


def locate_symbolic_k(
    cache_axes,  # noqa: ANN001 — tuple[Axis, ...]
    dims: tuple[int, ...],
    extents: tuple[int | Expr, ...],
    reduce_names: frozenset[str],
) -> tuple[int, int | Expr] | None:
    """Locate a staged source's symbolic contraction (K) gmem dim.

    Walks the source's ``cache_axes`` (aligned with ``dims``, the per-axis source
    dim): the first cache axis whose name is a reduce axis (``reduce_names``) and
    whose mapped source dim has a symbolic extent is the masked-K dim. Returns
    ``(k_dim, bound)`` — the source dim index and its runtime extent — or ``None``
    when no contraction dim is symbolic (the common static-K case)."""
    for i, ax in enumerate(cache_axes):
        if ax.name not in reduce_names:
            continue
        k_dim = dims[i]
        if k_dim < len(extents) and is_symbolic_extent(extents[k_dim]):
            return (k_dim, extents[k_dim])
    return None
