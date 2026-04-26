"""Structural normalization passes for ``LoopOp`` bodies.

Invoked from ``LoopOp.__post_init__`` so every constructed ``LoopOp`` —
including intermediate results produced by fusion rules — lands in a
canonical shape before validation. Pure ``body → body`` transforms,
applied in order by :func:`normalize_body`:

- :func:`drop_size_one_free_axes` — inline ``Loop(axis, extent=1)`` free
  loops, rewriting ``Var(axis.name)`` to ``Literal(0, "int")`` throughout
  the body. Reduce loops (bodies with ``Accum``) are left alone; dropping
  them would strip the accumulator.
- :func:`canonicalize_free_axis_order` — sort the outer chain of free
  ``Loop`` blocks alphabetically by axis name. Free loops commute, so
  reordering is safe; the chain terminates at a reduce Loop or a
  branching body.
- :func:`eliminate_copy_aliases` — treat ``y = copy(x)`` Assigns as
  aliases, rewire downstream references to the alias root, and drop the
  copies. Merge chains leave stacks of identity copies; this collapses
  them so the IR prints / compares cleanly.
- :func:`unify_sibling_reduce_axes` — at every scope, find sibling reduce
  ``Loop``s whose reduce axes index the same ``(source, dim)`` position and
  rename them to a single canonical axis name. Softmax's max-over-K /
  sum-over-K end up sharing one reduce axis; likewise
  ``sum(x, -1) + max(x, -1)`` after fusion.
- :func:`rename_ssa_sequential` — after the other passes settle, rename
  Assign / Select SSA names to ``v0``, ``v1``, ... in definition order.
  Accum and Load names are preserved (Accum names carry semantic roles;
  Load names are structural binding sites).
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.loop.ir import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    Select,
    Stmt,
    iter_body,
    map_body,
)
from deplodock.compiler.ir.sigma import Sigma

__all__ = [
    "normalize_body",
    "drop_size_one_free_axes",
    "canonicalize_free_axis_order",
    "eliminate_copy_aliases",
    "unify_sibling_reduce_axes",
    "hoist_loop_invariants",
    "rename_ssa_sequential",
]


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    """Closure that maps ``Axis(old, ...)`` to ``new``, leaving others alone."""
    return lambda a: new if a.name == old else a


# ---------------------------------------------------------------------------
# Top-level composition
# ---------------------------------------------------------------------------


def normalize_body(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Apply the structural and cosmetic normalization passes in order."""
    from deplodock.compiler.ir.loop.simplify import simplify_body

    stmts = drop_size_one_free_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = rename_ssa_sequential(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator."""

    def fn(s: Stmt) -> Stmt | tuple[Stmt, ...]:
        if not isinstance(s, Loop):
            return s
        candidate = Loop(axis=s.axis, body=map_body(s.body, fn))
        if int(candidate.axis.extent) == 1 and not candidate.is_reduce:
            sub = Sigma({candidate.axis.name: Literal(0, "int")})
            return tuple(c.rewrite(_identity_rename, sub) for c in candidate.body)
        return candidate

    return map_body(stmts, fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def canonicalize_free_axis_order(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Sort the outer chain of free ``Loop`` blocks alphabetically by axis
    name. The chain is the sequence of single-child free Loops at the top of
    ``stmts``; it terminates at a reduce Loop or a branching body. Recursion
    continues into that terminal body."""
    chain_axes: list[Axis] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain_axes.append(loop.axis)
        current = loop.body

    terminal = tuple(Loop(axis=s.axis, body=canonicalize_free_axis_order(s.body)) if isinstance(s, Loop) else s for s in current)

    chain_axes_sorted = sorted(chain_axes, key=lambda a: a.name)
    result: tuple[Stmt, ...] = terminal
    for axis in reversed(chain_axes_sorted):
        result = (Loop(axis=axis, body=result),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them (``y = copy(copy(copy(x)))``). Every such Assign is
    dropped and downstream references to ``y`` are rewired to the alias
    root. Semantically a no-op — pure IR hygiene."""
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def fn(s: Stmt) -> Stmt | None:
        if isinstance(s, Loop):
            return Loop(axis=s.axis, body=map_body(s.body, fn))
        if isinstance(s, Assign) and s.op.name == "copy" and len(s.args) == 1:
            alias[s.name] = s.args[0]
            return None
        return s.rewrite(resolve)

    return map_body(stmts, fn)


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes index
    the same ``(Load.source, dim)`` position and rename them to a single
    canonical axis name.

    Fusion commonly plants sibling reduces that sweep the same buffer at the
    same dim — softmax's max-over-K followed by sum-over-K, or
    ``sum(x, -1) + max(x, -1)`` after fusion. Sharing an axis name makes
    ``LoopOp.axes`` (which dedupes by name) report one reduce axis rather
    than N, matching the schedule the emitter would otherwise produce.

    Self-contained within a single body: two Loads with the same ``source``
    index are provably the same external buffer, so no caller-supplied
    input-name list is needed. Idempotent."""

    def walk(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        # Recurse into nested scopes first.
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, Loop):
                new_body.append(Loop(axis=s.axis, body=walk(s.body)))
            else:
                new_body.append(s)

        # Group sibling reduce Loops by their reduce axis's (source, dim) pattern.
        groups: dict[frozenset[tuple[str, int]], list[int]] = {}
        for i, s in enumerate(new_body):
            if isinstance(s, Loop) and s.is_reduce:
                positions = _reduce_axis_source_positions(s.body, s.axis.name)
                if positions:
                    groups.setdefault(frozenset(positions), []).append(i)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            first = new_body[indices[0]]
            assert isinstance(first, Loop)
            canonical = first.axis.name
            canonical_extent = int(first.axis.extent)
            for idx in indices[1:]:
                loop = new_body[idx]
                assert isinstance(loop, Loop)
                if int(loop.axis.extent) != canonical_extent or loop.axis.name == canonical:
                    continue
                new_axis = Axis(name=canonical, extent=canonical_extent)
                sub = Sigma({loop.axis.name: Var(canonical)})
                rename_axis = _make_axis_renamer(loop.axis.name, new_axis)
                renamed = tuple(s.rewrite(_identity_rename, sub, rename_axis) for s in loop.body)
                new_body[idx] = Loop(axis=new_axis, body=renamed)

        return tuple(new_body)

    return walk(stmts)


def _reduce_axis_source_positions(body: tuple[Stmt, ...], reduce_axis_name: str) -> set[tuple[str, int]]:
    """Collect ``(source, dim)`` positions where ``Var(reduce_axis_name)``
    appears bare in a Load index within ``body`` (recursing into nested
    Loops — important when the reduce body contains a nested free loop
    that does the actual load)."""
    return {
        (s.input, dim)
        for s in iter_body(body)
        if isinstance(s, Load)
        for dim, e in enumerate(s.index)
        if isinstance(e, Var) and e.name == reduce_axis_name
    }


# ---------------------------------------------------------------------------
# Pass 5: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Move ``Load`` / ``Assign`` / ``Select`` stmts out of ``Loop``s whose
    axis their value doesn't depend on.

    Applied bottom-up. For each ``Loop``, the body is first recursively
    hoisted; then stmts whose live axes don't include the loop's axis and
    whose SSA deps are available in the enclosing scope (either outer
    bindings or earlier-hoisted siblings) are emitted *before* the Loop
    in the parent body. ``Accum``, ``Write``, and ``Loop`` stmts stay in
    place — their position is semantically meaningful.

    Makes the IR's scope structure honest: every stmt lands at the
    shallowest scope that contains its dependencies, so downstream codegen
    sees a canonical form regardless of where fusion originally placed
    the stmt.
    """
    axes_of: dict[str, frozenset[str]] = {}
    ssa_names: set[str] = set()

    def _load_index_parts(index: tuple[Expr, ...]) -> tuple[set[str], set[str]]:
        """Split free vars in ``index`` into (ssa_deps, axis_names).

        A Load's index can reference other Load SSA names (gather pattern),
        not just axis Vars. SSA names are deps; axis names contribute to
        live_axes directly.
        """
        deps: set[str] = set()
        axes: set[str] = set()
        for e in index:
            for v in e.free_vars():
                (deps if v in ssa_names else axes).add(v)
        return deps, axes

    def _stmt_live(s: Stmt) -> frozenset[str]:
        if isinstance(s, Load):
            deps, axes = _load_index_parts(s.index)
            result = set(axes)
            for d in deps:
                result |= axes_of.get(d, frozenset())
            return frozenset(result)
        if isinstance(s, Assign):
            result = set()
            for name in s.args:
                result |= axes_of.get(name, frozenset())
            return frozenset(result)
        if isinstance(s, Select):
            result = set()
            for b in s.branches:
                result |= b.select.free_vars()
                result |= axes_of.get(b.value, frozenset())
            return frozenset(result)
        return frozenset()

    def _deps(s: Stmt) -> tuple[str, ...]:
        if isinstance(s, Load):
            deps, _ = _load_index_parts(s.index)
            return tuple(deps)
        if isinstance(s, Assign):
            return s.args
        if isinstance(s, Select):
            return tuple(b.value for b in s.branches)
        return ()

    def walk(body: tuple[Stmt, ...], outer_bindings: set[str]) -> list[Stmt]:
        new_body: list[Stmt] = []
        bindings = set(outer_bindings)

        for s in body:
            if isinstance(s, Loop):
                inner = walk(s.body, bindings)
                axis = s.axis.name
                hoisted_names: set[str] = set()
                hoisted: list[Stmt] = []
                stay: list[Stmt] = []
                for c in inner:
                    if isinstance(c, (Load, Assign, Select)):
                        live = axes_of.get(c.name, frozenset())
                        if axis not in live and all(d in bindings or d in hoisted_names for d in _deps(c)):
                            hoisted.append(c)
                            hoisted_names.add(c.name)
                            continue
                    stay.append(c)

                for h in hoisted:
                    new_body.append(h)
                    bindings.add(h.name)
                new_body.append(Loop(axis=s.axis, body=tuple(stay)))

                # Accum bindings leak to the enclosing scope after the Loop
                # closes; record their outside-the-loop live_axes.
                for c in inner:
                    if isinstance(c, Accum):
                        axes_of[c.name] = axes_of.get(c.value, frozenset()) - {axis}
                        ssa_names.add(c.name)
                        bindings.add(c.name)
            else:
                new_body.append(s)
                name = getattr(s, "name", None)
                if name is not None:
                    if isinstance(s, Accum):
                        # Shouldn't appear at root; record conservatively.
                        axes_of[name] = axes_of.get(s.value, frozenset())
                    else:
                        axes_of[name] = _stmt_live(s)
                    ssa_names.add(name)
                    bindings.add(name)

        return new_body

    return tuple(walk(stmts, set()))


# ---------------------------------------------------------------------------
# Pass 6: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Canonicalize names in a fused body:

    - Axes renamed to ``a0, a1, ...`` in pre-order of first declaration.
      Sibling ``Loop``s with the same axis name (post-unification reduce
      axes) share one renumbering entry.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    ssa_rename: dict[str, str] = {}
    axis_rename: dict[str, str] = {}
    # Expr-level substitution covers axis renames AND Load renames — a Load's
    # SSA name can appear as ``Var(load_name)`` in another Load's index (the
    # gather pattern: ``data[..., Var("idx"), ...]`` where ``idx`` is another
    # Load's name). Assign/Select names never surface in Exprs so they stay
    # out of this map.
    expr_sub: dict[str, Expr] = {}
    counters = {"v": 0, "in": 0, "acc": 0}

    def _rename(name: str, prefix: str) -> str:
        new = f"{prefix}{counters[prefix]}"
        ssa_rename[name] = new
        counters[prefix] += 1
        return new

    for stmt in iter_body(stmts):
        if isinstance(stmt, Load) and stmt.name not in ssa_rename:
            new = _rename(stmt.name, "in")
            if stmt.name != new:
                expr_sub[stmt.name] = Var(new)
        elif isinstance(stmt, Accum) and stmt.name not in ssa_rename:
            _rename(stmt.name, "acc")
        elif isinstance(stmt, (Assign, Select)) and stmt.name not in ssa_rename:
            _rename(stmt.name, "v")
        elif isinstance(stmt, Loop) and stmt.axis.name not in axis_rename:
            new = f"a{len(axis_rename)}"
            axis_rename[stmt.axis.name] = new
            if stmt.axis.name != new:
                expr_sub[stmt.axis.name] = Var(new)

    if all(o == n for o, n in ssa_rename.items()) and all(o == n for o, n in axis_rename.items()):
        return stmts

    sigma = Sigma(expr_sub)

    def rename_ssa(name: str) -> str:
        return ssa_rename.get(name, name)

    def axis_fn(a: Axis) -> Axis:
        new = axis_rename.get(a.name, a.name)
        return Axis(name=new, extent=a.extent) if new != a.name else a

    return tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in stmts)
