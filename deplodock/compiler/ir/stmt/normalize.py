"""Body-level normalization passes.

Pure ``body → body`` transforms applied via :func:`normalize_body` from
``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so every constructed
Op lands in canonical form. The passes operate on the shared Stmt
vocabulary (``Loop``, ``Load``, ``Assign``, ``Accum``, ``Select``,
``Write``) and recurse through every block-structured Stmt
(``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) so they apply uniformly
to Loop IR and Tile IR bodies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Pack, Select, Unpack, Write

# ---------------------------------------------------------------------------
# Visitor helpers shared by every pass below
# ---------------------------------------------------------------------------


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def normalize_body(
    stmts: Body,
    *,
    hoist: bool = True,
    canonical_buffers: bool = False,
    cluster_ops: bool = False,
) -> Body:
    """Apply the structural and cosmetic normalization passes in order.

    Used by both ``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so
    bodies — Loop-IR and Tile-IR — land in a canonical shape before
    validation.

    ``hoist=False`` skips :func:`hoist_loop_invariants`. TileOp bodies turn
    it off because a Stage binding is scoped to the Loop where it's
    declared — hoisting Loads that read from a staged buffer above the
    Stage decl would leave the read referencing an undeclared name.

    ``canonical_buffers=True`` runs :func:`canonicalize_buffer_names` after
    SSA renaming. Off by default — buffer names bind to graph inputs /
    outputs and are meaningful at the Op boundary. Turned on by
    :attr:`Body.structural_key()` so two bodies that read identical patterns
    from differently-named buffers hash and compare equal.

    ``cluster_ops=True`` runs :func:`canonicalize_op_clusters` after
    buffer renaming. Off by default — collapsing ``sub`` to ``add`` (or
    ``mod`` to ``divide``) destroys semantics, so this is only safe
    when the output is a hash key, never a runnable body. Turned on by
    :attr:`Body.structural_key()` so two bodies that differ only in the
    *kind* of FMA / compare / SFU op at the same position hash equal.
    """
    stmts = Body.coerce(stmts)
    stmts = topo_sort_siblings(stmts)
    stmts = drop_size_one_free_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = merge_sibling_reduce_loops(stmts)
    if hoist:
        stmts = split_invariant_divides(stmts)
        stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = rename_ssa_sequential(stmts)
    if canonical_buffers:
        stmts = canonicalize_buffer_names(stmts)
    if cluster_ops:
        stmts = canonicalize_op_clusters(stmts)
    # Sort runs last so the keys it sorts by are the post-rename canonical
    # SSA / buffer names — that way two bodies that differ only in original
    # argument order produce identical post-normalization arg tuples.
    stmts = sort_commutative_args(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: Body) -> Body:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator.
    Recurses through StridedLoop / Tile / Cond bodies without rewriting
    those wrappers (their iteration semantics aren't a free Loop).

    Size-1 BLOCK / SPLITK_BLOCK protection used to live here when the
    planner stamped ``Loop.role`` for downstream launch_geometry to
    consume. The planner now constructs ``GridTile`` / ``ThreadTile``
    directly and applies its own size-1 filter (see
    ``010_partition_loops::_wrap_tower``), so by the time
    ``drop_size_one_free_axes`` runs on a LoopOp body, no Loop has any
    binding role — every size-1 free Loop is safely inlinable.
    """
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt | Body:
        # Body.map post-order: ``s.body`` is already recursively mapped.
        if isinstance(s, Loop) and s.axis.extent.is_static and s.axis.extent.as_static() == 1 and not s.is_reduce:
            sub = Sigma({s.axis.name: Literal(0, "int")})
            return tuple(c.rewrite(_identity_rename, sub) for c in s.body)
        return s

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def _recurse_canonicalize(s: Stmt) -> Stmt:
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(canonicalize_free_axis_order(b) for b in nested))


def canonicalize_free_axis_order(stmts: Body) -> Body:
    """Sort the outer chain of free ``Loop`` blocks alphabetically by axis
    name. The chain is the sequence of single-child free Loops at the top of
    ``stmts``; it terminates at a reduce Loop or a branching body.
    Recursion continues into terminal block bodies (Loop / StridedLoop /
    Tile / Cond)."""
    stmts = Body.coerce(stmts)
    chain: list[Loop] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain.append(loop)
        current = loop.body

    terminal = tuple(_recurse_canonicalize(s) for s in current)

    chain_sorted = sorted(chain, key=lambda lp: lp.axis.name)
    result: Body = terminal
    for loop in reversed(chain_sorted):
        result = (Loop(axis=loop.axis, body=result, unroll=loop.unroll),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: Body) -> Body:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them. Every such Assign is dropped and downstream
    references to ``y`` are rewired to the alias root. Pure IR hygiene."""
    stmts = Body.coerce(stmts)
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def fn(s: Stmt) -> Stmt | None:
        # Body.map post-order: block bodies already recursed; only handle leaves.
        if isinstance(s, (Loop, StridedLoop, Cond)):
            return s
        if isinstance(s, Assign) and s.op.name == "copy" and len(s.args) == 1:
            alias[s.name] = s.args[0]
            return None
        return s.rewrite(resolve)

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: Body) -> Body:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes
    index overlapping ``(Load.source, dim)`` positions and rename them
    to a single canonical axis name. Recurses through every block-
    structured Stmt (Loop / StridedLoop / Tile / Cond) to find nested
    scopes."""
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        # Recurse into nested bodies first (post-order) via the canonical
        # nested() / with_bodies() descent, then group siblings at this
        # scope. Splitting the recursion from the sibling-grouping keeps
        # this pass's scope-level logic isolated in ``_unify_siblings``.
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _unify_siblings(Body(recursed))

    return walk(stmts)


def _unify_siblings(body: Body) -> Body:
    """Single-scope sibling grouping: rename reduce-axis vars across
    sibling reduce Loops whose bare-Var Load positions overlap on any
    ``(source, dim)`` pair so they share one canonical axis name.

    Two reduce Loops that bind different axis names but both index the
    same input slot (e.g. ``x[..., a2]`` and ``x[..., a3]`` for the
    same ``x``) are semantically the same reduction dimension. Union-
    find on the overlap relation merges all transitively-connected
    Loops into one group. Within a group, the first Loop's axis name
    wins; later Loops are rewritten to use it.

    Pairing on overlap rather than exact-set equality lets matmul-
    siblings that bring in distinct weight tensors (e.g.
    ``silu(x@Wg) * (x@Wu)`` — both reduce over K and index x, but only
    one indexes Wg and the other Wu) unify on the shared x position;
    the downstream :func:`merge_sibling_reduce_loops` pass then
    concatenates their bodies.
    """
    stmts = list(body)

    # Key on ``Dim.expr`` (the underlying ``Expr``) so structural equality on
    # extents matches both static and symbolic siblings: two ``Dim('seq_len')``
    # siblings unify (both back to ``Var('seq_len')``); two distinct symbolic
    # names don't. ``Expr`` is frozen + hashable so it slots into the tuple key.
    entries: list[tuple[int, str, object, frozenset[tuple[str, int]]]] = []
    for i, s in enumerate(stmts):
        if isinstance(s, Loop) and s.is_reduce:
            positions = _reduce_axis_source_positions(s.body, s.axis.name)
            if positions:
                entries.append((i, s.axis.name, s.axis.extent.expr, frozenset(positions)))

    if len(entries) < 2:
        return Body(stmts)

    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            if entries[a][2] != entries[b][2]:
                continue
            if entries[a][3] & entries[b][3]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    for k, (idx, axis_name, extent, _) in enumerate(entries):
        canonical = entries[find(k)][1]
        if canonical == axis_name:
            continue
        loop = stmts[idx]
        assert isinstance(loop, Loop)
        # Preserve source_axis / real_extent across the rename so masked-tile
        # axes don't lose their pre-ceil-div bound.
        new_axis = Axis(name=canonical, extent=extent, source_axis=loop.axis.source_axis, real_extent=loop.axis.real_extent)
        sub = Sigma({loop.axis.name: Var(canonical)})
        rename_axis = _make_axis_renamer(loop.axis.name, new_axis)
        renamed = tuple(s.rewrite(_identity_rename, sub, rename_axis) for s in loop.body)
        stmts[idx] = replace(loop, axis=new_axis, body=renamed)

    return Body(stmts)


def _reduce_axis_source_positions(body: Body, reduce_axis_name: str) -> set[tuple[str, int]]:
    """Collect ``(source, dim)`` positions where ``Var(reduce_axis_name)``
    appears bare in a Load index within ``body`` (recursing into nested
    blocks)."""
    return {
        (s.input, dim)
        for s in body.iter()
        if isinstance(s, Load)
        for dim, e in enumerate(s.index)
        if isinstance(e, Var) and e.name == reduce_axis_name
    }


# ---------------------------------------------------------------------------
# Pass 4b: merge sibling reduce Loops with matching axis into one Loop.
# ---------------------------------------------------------------------------
#
# After :func:`unify_sibling_reduce_axes` renames sibling reduce axes
# that index overlapping ``(source, dim)`` positions to one canonical
# name, adjacent reduce Loops with the same axis name/extent become
# structurally identical iteration scopes. Merging concatenates their
# bodies into one Loop so the reduce axis is traversed once instead of
# twice. Downstream ``dedup_loads`` then collapses the duplicate Loads
# both halves share — e.g. ``load x[0, a0, k]`` in the gated-MLP
# pattern ``silu(x@Wg) * (x@Wu)`` where both matmuls reduce over the
# same K and share x as a Load source. Symmetric staging follows: once
# wu lives in the same K-loop as wg, ``stage_inputs`` / ``use_tma`` /
# ``use_ring_buffers`` apply uniformly.
# ---------------------------------------------------------------------------


def merge_sibling_reduce_loops(stmts: Body) -> Body:
    """Merge sibling reduce ``Loop``s with matching ``axis.name`` and
    ``axis.extent`` into one Loop whose body is the concatenation.

    Gates a merge on three conditions:

    1. The two bodies have disjoint SSA defs — no name collision and
       no inner-scope shadowing once they share one Loop.
    2. The second Loop's body does not read any SSA name the first
       Loop's body defines (including ``Accum`` exports). When it
       does, the two reductions are sequentially dependent — e.g.
       softmax's sum-exp loop reads ``acc_max`` from the preceding
       max loop. Merging would replace that read of the *finalized*
       max with a read of the in-flight per-iter value, changing
       semantics.
    3. No statement that sits between the two Loops defines an SSA
       name the second Loop's body reads — otherwise the merge would
       move that read above its def.

    Statements that sit between the two original Loops stay in their
    original positions in the parent Body. References to the first
    Loop's ``Accum`` remain valid (Accum names cross the Loop
    boundary). References to the second Loop's ``Accum`` from
    statements that originally followed it now resolve to the merged
    Loop above them — still defs-before-uses.

    Recurses through every block-structured Stmt to find nested scopes.
    """
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _merge_sibling_reduce_loops(Body(recursed))

    return walk(stmts)


def _merge_sibling_reduce_loops(body: Body) -> Body:
    items = list(body)
    if len(items) < 2:
        return body

    out: list[Stmt] = []
    consumed: set[int] = set()
    for i, s in enumerate(items):
        if i in consumed:
            continue
        if not (isinstance(s, Loop) and s.is_reduce):
            out.append(s)
            continue
        merged = s
        for j in range(i + 1, len(items)):
            if j in consumed:
                continue
            t = items[j]
            if not (
                isinstance(t, Loop)
                and t.is_reduce
                and t.axis.name == merged.axis.name
                and t.axis.extent == merged.axis.extent
                and t.unroll == merged.unroll
            ):
                continue
            merged_defs = _all_ssa_defs(merged.body)
            if merged_defs & _all_ssa_defs(t.body):
                continue
            if merged_defs & _all_ssa_uses(t.body):
                continue
            between_defs: set[str] = set()
            for k in range(i + 1, j):
                if k in consumed:
                    continue
                between_defs |= _all_ssa_defs(Body((items[k],)))
            if between_defs & _all_ssa_uses(t.body):
                continue
            merged = Loop(
                axis=merged.axis,
                body=Body(tuple(merged.body) + tuple(t.body)),
                unroll=merged.unroll,
            )
            consumed.add(j)
        out.append(merged)

    return Body(out)


# ---------------------------------------------------------------------------
# Pass 5a: split loop-invariant divides into reciprocal + multiply.
# ---------------------------------------------------------------------------
#
# ``divide(x, y)`` lowers to a single-precision divide on the XU pipe (the
# same pipe ``exp`` uses). When ``y`` is loop-invariant w.r.t. some
# enclosing Loop and ``x`` is not, the divide can't hoist as-is — its live
# set is the union of x's and y's. Splitting into::
#
#     recip_y = reciprocal(y)        # live = axes_of(y)
#     result  = multiply(x, recip_y) # live = axes_of(x) ∪ {recip_y}
#
# lets the next pass (``hoist_loop_invariants``) move ``recip_y`` out of
# every Loop axis that doesn't appear in ``y``. Inside the loop the
# divide turns into a multiply (FMA pipe), which is typically the
# under-utilized pipe on transcendental-heavy kernels (softmax,
# RMSNorm, attention output). One XU op per outer-axis iteration
# instead of one per inner-axis iteration.
#
# Gate: split iff ``axes_of(y)`` is a strict subset of ``axes_of(x)``.
# That's the precise structural condition for "splitting unblocks at
# least one Loop's worth of hoisting." Skip when y has axes x doesn't
# (no hoisting wins) or when both have identical axes (rcp would stay
# in the same scope as the original divide, no win and slight
# precision drift). When y is a true scalar (axes_of empty), the rcp
# hoists all the way to body root.
# ---------------------------------------------------------------------------


def split_invariant_divides(stmts: Body) -> Body:
    """Rewrite ``divide(x, y)`` → ``reciprocal(y) + multiply(x, recip)``
    when ``y``'s axis-dependency set is a strict subset of ``x``'s.

    Invariance is queried via :meth:`Body.deps_closure` over the
    pre-rewrite body, filtered to axis names. The strict-subset check
    means there's at least one axis ``x`` depends on that ``y``
    doesn't — splitting moves the rcp out of that axis's Loop while
    the multiply stays. Generates fresh SSA names for the rcp; the
    trailing :func:`rename_ssa_sequential` pass renumbers them into
    ``vN`` form.
    """
    from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    stmts = Body.coerce(stmts)
    closure = stmts.deps_closure
    axes = stmts.axis_names
    ssa_names: set[str] = set(closure.keys())
    fresh_counter = [0]

    def _fresh(prefix: str) -> str:
        while True:
            fresh_counter[0] += 1
            n = f"{prefix}_{fresh_counter[0]}"
            if n not in ssa_names:
                ssa_names.add(n)
                return n

    def _axes_of(name: str) -> frozenset[str]:
        return closure.get(name, frozenset()) & axes

    def walk(body: Body) -> Body:
        out: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                # Generic descent — recurse into every nested body, rebuild
                # the wrapper via with_bodies. The closure was built once
                # over the whole body, so post-Loop Accum bookkeeping is
                # already baked in — no per-wrapper update needed here.
                out.append(s.with_bodies(tuple(walk(b) for b in nested)))
                continue
            if isinstance(s, Assign) and s.op == ElementwiseImpl("divide") and len(s.args) == 2:
                x_name, y_name = s.args
                if _axes_of(y_name) < _axes_of(x_name):  # strict subset → splitting unblocks at least one hoist
                    recip_name = _fresh(f"recip_{y_name}")
                    recip = Assign(name=recip_name, op=ElementwiseImpl("reciprocal"), args=(y_name,))
                    mult = Assign(name=s.name, op=ElementwiseImpl("multiply"), args=(x_name, recip_name))
                    # Patch closure for the freshly-introduced rcp so a
                    # later divide reading the same y in the same body
                    # still sees the correct axis set.
                    closure[recip_name] = closure.get(y_name, frozenset())
                    closure[mult.name] = closure.get(x_name, frozenset()) | closure[recip_name]
                    out.append(recip)
                    out.append(mult)
                    continue
            out.append(s)
        return Body(out)

    return walk(stmts)


# ---------------------------------------------------------------------------
# Pass 5b: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: Body) -> Body:
    """Move stmts out of ``Loop``s whose axis they don't depend on.

    Hoists ``Load`` / ``Assign`` / ``Select`` (SSA values) and entire
    ``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond`` blocks whose contents
    transitively avoid the outer axis — provided the block contains no
    ``Write`` (a Write hoist would change observable side effects).
    Block-level hoisting is what lets a Loop and its downstream consumer
    move together: hoisting just the consumer would leave it referencing
    an Accum still defined inside the outer Loop body.

    ``Accum`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Loop-invariance is queried via :meth:`Body.depends_on`
    against the body's transitive read closure, so the hoisted set is
    automatically closed under SSA dependencies — no separate ordering
    check is needed.
    """
    stmts = Body.coerce(stmts)

    def _hoistable(s: Stmt, axis: str) -> bool:
        # Accum / Init are scope-bound to their enclosing Loop's reduction — they can't
        # move alone, but the whole enclosing block can. Side-effecting stmts (Write, or
        # any block containing a Write) pin their iteration count and stay put.
        if isinstance(s, (Accum, Init)) or s.has_side_effects():
            return False
        return not stmts.depends_on(s, axis)

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = [c for c in inner if _hoistable(c, axis)]
                hoisted_ids = {id(c) for c in hoisted}
                stay = [c for c in inner if id(c) not in hoisted_ids]
                new_body.extend(hoisted)
                new_body.append(replace(s, body=tuple(stay)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=tuple(walk(s.body)), else_body=tuple(walk(s.else_body))))
            else:
                new_body.append(s)
        return new_body

    return tuple(walk(stmts))


# ---------------------------------------------------------------------------
# Pass 6: simplify Exprs inside body Stmts (constant folding, identity collapse,
# range-based comparison folding). The per-Expr rewrite logic lives on each
# ``Expr`` subclass as ``simplify(ctx)``; the walk over Stmts is dispatched
# in :mod:`.passes` (singledispatch + Stage introspection).
# ---------------------------------------------------------------------------


def simplify_body(body: Body) -> Body:
    """Simplify every Expr inside a body. Seeds ``SimplifyCtx`` from
    ``Loop`` / ``StridedLoop`` / ``Tile`` axis extents as the walker descends.
    Tile-IR Stmt registrations are loaded when ``tile.ir`` is imported."""
    from deplodock.compiler.ir.stmt.passes import simplify  # noqa: PLC0415

    body = Body.coerce(body)
    ctx = SimplifyCtx.empty()
    return tuple(simplify(s, ctx) for s in body)


# ---------------------------------------------------------------------------
# Pass: deduplicate Load stmts with identical (input, index)
# ---------------------------------------------------------------------------


def dedup_loads(stmts: Body) -> Body:
    """Drop duplicate ``Load`` stmts within nested scopes.

    Two ``Load`` stmts with the same ``(input, index)`` read the same
    value; keep the first and rewire downstream SSA references to its
    name. Operates per-scope: a Load at an outer scope is reused by
    inner siblings (their identical ``index`` doesn't reference any
    inner-axis Var, so the values are equal). Loads inside a nested
    scope are not visible to outer / sibling scopes."""
    stmts = Body.coerce(stmts)

    def walk(
        body: Body,
        env: dict[tuple[str, tuple[str, ...]], str],
        parent_alias: dict[str, str],
    ) -> Body:
        local = dict(env)
        alias = dict(parent_alias)

        def rename(n: str) -> str:
            return alias.get(n, n)

        out: list[Stmt] = []
        for s in body:
            if isinstance(s, Load):
                # Rewire any SSA names in this Load's *index* to their deduped
                # alias first — a gather ``weight[(int)in0, a]`` whose index
                # Load ``in0`` was itself deduped must follow ``in0`` to the
                # kept name, or the index dangles after the duplicate is
                # dropped. (No-op for plain axis indices: axes aren't aliased.)
                s = s.rewrite(rename)
                key = (s.input, tuple(e.pretty() for e in s.index))
                if key in local:
                    alias[s.name] = local[key]
                    continue
                local[key] = s.name
                out.append(s)
            elif isinstance(s, Loop):
                out.append(replace(s, body=walk(s.body, local, alias)))
            elif isinstance(s, StridedLoop):
                out.append(replace(s, body=walk(s.body, local, alias)))
            elif isinstance(s, Cond):
                out.append(Cond(cond=s.cond, body=walk(s.body, local, alias), else_body=walk(s.else_body, local, alias)))
            else:
                out.append(s.rewrite(rename))
        return tuple(out)

    return walk(stmts, {}, {})


# ---------------------------------------------------------------------------
# Pass: topologically sort siblings so SSA defs precede their uses.
# ---------------------------------------------------------------------------


def topo_sort_siblings(stmts: Body) -> Body:
    """Reorder stmts within each Body so SSA defs precede their uses.

    Recurses into every child body via the Stmt protocol
    (:meth:`Stmt.nested` / :meth:`Stmt.with_bodies`), then runs a stable
    Kahn ordering over the current sibling list. A block stmt
    (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) is opaque at the
    parent level: it ``defs`` any Accum names that escape its body
    (visible to siblings via Loop's cross-boundary Accum semantics) and
    ``uses`` its wrapper-level deps plus any free SSA names referenced
    inside (names referenced inside but not defined inside).

    Splicer worklists (and any future producer that emits stmts with
    sibling-dedup) can land a consumer above an already-emitted producer
    when the producer was reused from an earlier emission. Sorting at
    normalize time decouples final body order from producer subtleties
    and guarantees every constructed ``LoopOp`` / ``TileOp`` lands with
    defs above uses, which downstream passes (validator, renamer,
    codegen) rely on.

    Stable: when the dep edges leave a free choice, the original sibling
    order is preserved (heap-based Kahn with index tiebreak). Idempotent:
    bodies already in topo order round-trip unchanged.
    """
    return _topo(Body.coerce(stmts))


def _topo(body: Body) -> Body:
    import heapq

    items: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            items.append(s.with_bodies(tuple(_topo(b) for b in nested)))
        else:
            items.append(s)

    n = len(items)
    if n <= 1:
        return Body(tuple(items))

    defs_uses = [_sibling_defs_uses(s) for s in items]
    # First-writer wins: handles repeated Accum decls (idempotent at the
    # same name) and the rare aliasing edge case without crashing.
    def_idx: dict[str, int] = {}
    for i, (defs, _) in enumerate(defs_uses):
        for name in defs:
            def_idx.setdefault(name, i)

    incoming: list[set[int]] = [set() for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for i, (_, uses) in enumerate(defs_uses):
        for name in uses:
            j = def_idx.get(name)
            if j is not None and j != i and j not in incoming[i]:
                incoming[i].add(j)
                outgoing[j].append(i)

    ready: list[int] = [i for i in range(n) if not incoming[i]]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for k in outgoing[i]:
            incoming[k].discard(i)
            if not incoming[k]:
                heapq.heappush(ready, k)

    if len(order) != n:
        # Cycle through SSA names — leave order untouched so the validator
        # rejects it with a precise message instead of silently shuffling.
        return Body(tuple(items))
    return Body(tuple(items[i] for i in order))


#: Marker suffix for loop-carried register-pipeline operand buffers
#: (``kernel/013_pipeline_mma_regs``). Such a fragment is written by the
#: prefetch at the *bottom* of the K_o loop body and read by the ``mma`` at the
#: *top* of the NEXT iteration — a loop-carried dependency across the back-edge,
#: NOT an intra-body def→use. ``topo_sort_siblings`` must not bind the read to
#: the in-body write (it would sink the read below the prefetch and compute the
#: next tile instead of the current one); these names are excluded from the
#: def-use edge graph so their already-correct source order is preserved by the
#: stable tiebreak. Single-assignment names (every other body) are unaffected.
_LOOP_CARRIED_MARK = "__rp1"


def _loop_carried(name: str) -> bool:
    return _LOOP_CARRIED_MARK in name


def _sibling_defs_uses(stmt: Stmt) -> tuple[frozenset[str], frozenset[str]]:
    """Names ``stmt`` makes visible to siblings, and names it depends on
    from siblings.

    Leaves: ``defs = stmt.defines()``, ``uses = stmt.deps()``.
    Block stmts: ``defs`` = Accum names escaping the body (recursive);
    ``uses`` = wrapper's own deps ∪ ((all inner uses) − (all inner SSA
    defs)).

    Loop-carried register-pipeline buffers (``_loop_carried``) are dropped from
    both sets so the topo sort leaves their read-then-rewrite ordering alone.
    """
    nested = stmt.nested()
    if not nested:
        return (
            frozenset(n for n in stmt.defines() if not _loop_carried(n)),
            frozenset(n for n in stmt.deps() if not _loop_carried(n)),
        )
    defs: set[str] = set()
    all_uses: set[str] = set(stmt.deps())
    all_inner_defs: set[str] = set()
    for b in nested:
        defs |= _exported_accs(b)
        all_uses |= _all_ssa_uses(b)
        all_inner_defs |= _all_ssa_defs(b)
    return frozenset(n for n in defs if not _loop_carried(n)), frozenset(n for n in (all_uses - all_inner_defs) if not _loop_carried(n))


def _exported_accs(body: Body) -> frozenset[str]:
    out: set[str] = set()
    for s in body:
        if isinstance(s, Accum):
            out.add(s.name)
        for b in s.nested():
            out |= _exported_accs(b)
    return frozenset(out)


def _all_ssa_defs(body: Body) -> frozenset[str]:
    out: set[str] = set()
    for s in body:
        out.update(s.defines())
        for b in s.nested():
            out |= _all_ssa_defs(b)
    return frozenset(out)


def _all_ssa_uses(body: Body) -> frozenset[str]:
    out: set[str] = set()
    for s in body:
        out.update(s.deps())
        for b in s.nested():
            out |= _all_ssa_uses(b)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: Body) -> Body:
    """Canonicalize names in a fused body:

    - Axes from every axis-bearing scope (``Loop`` / ``StridedLoop`` /
      ``Tile.axes`` / new tile flavors' axes) renamed to ``a0, a1, ...``
      in pre-order of first declaration. All scopes share one numbering
      namespace so Tile.axes ``a0_o`` and a Loop axis ``a2_o`` don't
      collide on rename.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    from deplodock.compiler.ir.tile.ir import (  # noqa: PLC0415 — break stmt↔tile cycle
        ParallelTile,
        SerialTileBase,
        Stage,
    )

    stmts = Body.coerce(stmts)
    ssa_rename: dict[str, str] = {}
    axis_rename: dict[str, str] = {}
    expr_sub: dict[str, Expr] = {}
    counters = {"v": 0, "in": 0, "acc": 0}

    def _rename(name: str, prefix: str) -> str:
        new = f"{prefix}{counters[prefix]}"
        ssa_rename[name] = new
        counters[prefix] += 1
        return new

    def _record_axis(name: str) -> None:
        if name in axis_rename:
            return
        new = f"a{len(axis_rename)}"
        axis_rename[name] = new
        if name != new:
            expr_sub[name] = Var(new)

    for stmt in stmts.iter():
        if isinstance(stmt, Load):
            for old in stmt.names:
                if old in ssa_rename:
                    continue
                # Only record the SSA rename in ``ssa_rename`` — NOT in
                # ``expr_sub`` (sigma). The Load/Write rewriters apply
                # ``_rename_ssa_vars_in_expr(sigma.apply(e), rename)`` to index
                # exprs: ``sigma`` is the axis-substitution channel, ``rename``
                # the SSA channel. Putting an SSA rename in *both* renames an
                # indirect (gather) index Var twice. Sequential renumbering can
                # form a chain — e.g. cell-3's index ``in2_3 → in5`` while a
                # pre-existing ``in5`` (a layernorm-weight Load) → ``in26`` —
                # and the double application collapses it (``in2_3 → in5 →
                # in26``), wiring the gather to the wrong row. ``acc`` / ``v``
                # names are likewise kept out of ``expr_sub`` (they reach exprs
                # only via ``rename``), so this keeps Load names consistent.
                _rename(old, "in")
        elif isinstance(stmt, Accum) and stmt.name not in ssa_rename:
            _rename(stmt.name, "acc")
        elif isinstance(stmt, (Assign, Select)) and stmt.name not in ssa_rename:
            _rename(stmt.name, "v")
        elif isinstance(stmt, Unpack):
            # ``low_name`` and ``high_name`` are fresh SSA scalars
            # defined by Unpack — must get rename slots in the ``v`` pool.
            # Without this, they collided with their input's renamed name
            # (e.g. paired Accum ``acc0_acc1_p`` → ``acc0`` makes
            # ``Unpack(low_name="acc0", value="acc0_acc1_p")`` rewrite to
            # ``Unpack(low_name="acc0", value="acc0")`` — self-referential).
            for old in (stmt.low_name, stmt.high_name):
                if old not in ssa_rename:
                    _rename(old, "v")
        elif isinstance(stmt, Pack) and stmt.name not in ssa_rename:
            # ``Pack.name`` defines a fresh f16x2 SSA value consumed by the
            # next Accum. Same reasoning as Assign — give it a ``v`` slot.
            _rename(stmt.name, "v")
        elif isinstance(stmt, ParallelTile):
            # GridTile / ThreadTile / RegisterTile — record every axis in
            # the tuple before any nested Stage's cache axes so the
            # parallel coords keep their pre-order rename slots.
            for ax in stmt.axes:
                _record_axis(ax.name)
        elif isinstance(stmt, Stage):
            for src in stmt.sources:
                for ax in src.cache_axes:
                    _record_axis(ax.name)
        elif isinstance(stmt, (Loop, StridedLoop)):
            _record_axis(stmt.axis.name)
        elif isinstance(stmt, SerialTileBase):
            _record_axis(stmt.axis.name)

    if all(o == n for o, n in ssa_rename.items()) and all(o == n for o, n in axis_rename.items()):
        return stmts

    sigma = Sigma(expr_sub)

    def rename_ssa(name: str) -> str:
        return ssa_rename.get(name, name)

    def axis_fn(a: Axis) -> Axis:
        new = axis_rename.get(a.name, a.name)
        if new == a.name:
            return a
        # Preserve source_axis / real_extent so masked-tile metadata
        # survives the SSA rename pass.
        return Axis(name=new, extent=a.extent, source_axis=a.source_axis, real_extent=a.real_extent)

    return tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in stmts)


# ---------------------------------------------------------------------------
# Pass: sort args of commutative Assigns.
# ---------------------------------------------------------------------------


def sort_commutative_args(stmts: Body) -> Body:
    """Sort ``Assign.args`` for commutative ``op``s so two bodies that
    differ only by argument order land in the same canonical form.

    Acts on ``Assign`` only — Expr-level commutativity (e.g. ``a + b``
    inside a ``Load`` index or ``Cond.cond``) is handled by
    :func:`simplify_body` via the per-Expr ``simplify`` rules. Recurses
    through every block-structured Stmt (``Loop`` / ``StridedLoop`` /
    ``Tile`` / ``Cond``)."""
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Assign) and s.op.commutative and len(s.args) > 1:
            sorted_args = tuple(sorted(s.args))
            if sorted_args != s.args:
                return replace(s, args=sorted_args)
        return s

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass: canonicalize external-buffer names (opt-in via normalize_body flag).
# ---------------------------------------------------------------------------


def canonicalize_buffer_names(stmts: Body) -> Body:
    """Rename ``Load.input`` and ``Write.output`` buffer references to
    ``b0, b1, ...`` in encounter order via :meth:`Body.iter`.

    Off by default — buffer names bind to graph nodes (each ``Load.input``
    matches the producing op's id), so renaming them in a body that's
    still attached to an Op would break that wiring. Used by
    :attr:`Body.structural_key()` for dedup queries where buffer identity
    doesn't matter (two bodies with identical access patterns over
    differently-named inputs are structurally equal)."""
    stmts = Body.coerce(stmts)

    rename: dict[str, str] = {}
    for s in stmts.iter():
        if isinstance(s, Load) and s.input not in rename:
            rename[s.input] = f"b{len(rename)}"
        elif isinstance(s, Write) and s.output not in rename:
            rename[s.output] = f"b{len(rename)}"

    if all(o == n for o, n in rename.items()):
        return stmts

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.input in rename:
            return replace(s, input=rename[s.input])
        if isinstance(s, Write) and s.output in rename:
            return replace(s, output=rename[s.output])
        return s

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass: collapse ops to their compute-unit cluster representative
# (opt-in via normalize_body's ``cluster_ops`` flag).
# ---------------------------------------------------------------------------


def canonicalize_op_clusters(stmts: Body) -> Body:
    """Replace every ``ElementwiseImpl`` field on every stmt with its
    cluster representative from :func:`cluster_representative`.

    The pass walks ``stmts`` with :meth:`Body.map` and uses
    ``dataclasses.fields`` to locate any field currently holding an
    ``ElementwiseImpl`` (covers ``Init.op`` / ``Assign.op`` /
    ``Accum.op`` / Kernel-IR's ``TreeHalve.op`` /
    ``WarpShuffle.op`` without coupling this module to those IR
    dialects). The replacement is destructive — the resulting body is
    only safe to consume from :attr:`Body.structural_key()`.
    """
    from dataclasses import fields, is_dataclass  # noqa: PLC0415

    from deplodock.compiler.ir.elementwise import ElementwiseImpl, cluster_representative  # noqa: PLC0415

    def fn(s: Stmt) -> Stmt:
        if not is_dataclass(s):
            return s
        changes: dict[str, ElementwiseImpl] = {}
        for f in fields(s):
            val = getattr(s, f.name)
            if isinstance(val, ElementwiseImpl):
                rep = cluster_representative(val)
                if rep != val:
                    changes[f.name] = rep
        if not changes:
            return s
        return replace(s, **changes)

    return stmts.map(fn)
