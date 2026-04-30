"""Register-tile matmul-shaped reduce kernels (axis-aware).

Each thread in the post-blockify state owns one output element of the
CTA's M×N tile (PAT × PAT threads — default ``PAT=16`` so 256 threads,
1 output / thread). ``PAT`` (per-axis thread tile width) and ``F``
(per-thread output factor) are both supplied by ``tuning`` via
``detect_pat`` / ``register_tile_factor`` and are paired through the
``_PAT_TO_FACTOR`` table; the only PAT that fires under the default
thread budget of 256 is ``PAT=16``, but the rule's logic is parametric
in PAT.

This pass splits each of the two ``BIND_THREAD`` axes ``a:PAT`` into
outer ``a_o:PAT/F`` (still ``BIND_THREAD``) plus a serial ``a_i:F``
dimension, and replicates the matmul reduce body + epilogue per
``(a_i, a_j)`` cell. Each replicated cell carries its own SSA
accumulator (``acc0_<i>_<j>``), giving F² independent partial sums per
thread that nvcc can schedule in parallel registers. With ``PAT=16``
and ``F=2`` this is per-thread output 4 (4× more FMAs per smem-load
round-trip) at 64 threads per CTA.

**Axis-aware replication of pre-K-outer stmts.** The K-inner body
always replicates F² (one per output cell). Stmts upstream of the
K-outer loop (sibling reduces in fused SDPA, scalar-load preambles
in linear-bias kernels) replicate by the *thread-axis subset* their
output depends on:

==============================  ============  =============================
Stmt's thread-axis dependence   Replicas      Per-cell name suffix
==============================  ============  =============================
``∅`` (constant / batch only)   1             ``""``  (shared across cells)
``{m_axis}`` (per-row)          F             ``"_<i>"``
``{n_axis}`` (per-col)          F             ``"_<j>"``
``{m_axis, n_axis}``            F²            ``"_<i>_<j>"``
==============================  ============  =============================

Names defined locally inside the K-inner body always carry the F²
suffix (one accumulator per output cell). Names defined upstream are
suffixed by their dependence subset, so a K-inner cell at (i, j)
reads ``acc_max_<i>`` (per-row dep) instead of ``acc_max_<i>_<j>``.

This generalization is what *would* let the rule fire on fused SDPA:
the softmax pre-passes produce ``acc_max`` / ``acc_sum`` per-row, so
they need F replicas (along m), not F², and the matmul body's reads of
those accumulators get rewritten to the right per-cell name. SDPA is
held off in practice by the smem gate below; the analysis itself is
correct on that shape.

Idempotence: triggers only when exactly two ``BIND_THREAD`` axes have
an extent that's a key in ``_PAT_TO_FACTOR`` (currently
``{16: 2, 32: 4, 64: 8}``). After firing, the split THREAD axis has
extent ``pat/F`` — for every entry in the table that's exactly ``8``,
which is deliberately *not* a candidate. So ``detect_pat`` returns
``None`` on a second pass and the rule skips at its first gate. This
invariant is load-bearing: any future addition to ``_PAT_TO_FACTOR``
whose ``pat/F`` lands on another candidate's ``pat`` would re-trigger
the rule and double-replicate the K-inner body.

The replicated cells use distinct SSA names so the resulting kernel
needs no nvcc-side scalarization — the per-cell accumulator chains are
already independent at the IR level. ``place_inits`` (kernel pass) emits
one ``Init`` per cell at the right scope (Tile body head, since the
free K-outer loop is reduce-passthrough).

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile`` whose ``block_axes`` is empty.
- ``detect_pat`` matches a PAT, and ``register_tile_factor`` is
  ``F > 1`` with ``pat % F == 0``.
- The Tile has exactly two ``BIND_THREAD`` axes with extent equal to PAT.
- The Tile body has a top-level free ``Loop`` (the K-outer chunk loop)
  whose body contains a single reduce ``Loop`` (the K-inner reduce)
  with at least two distinct K-indexed buffer Loads + exactly one Accum,
  body composed only of ``Load`` / ``Assign`` / ``Accum`` (located by
  ``_find_matmul_k_outer``).
- ``pre_outer`` contains no reduce Loops (the conservative smem gate —
  replicating a pre_outer reduce's Stage F-fold blows past the 48 KB
  smem budget after pad + double-buffer downstream; this is what
  currently keeps SDPA on the baseline path).

External SSA reads inside the K-inner body whose definitions can't be
located in ``pre_outer`` are *not* a gate — the analysis treats them
as un-replicated passthroughs (axes_used = ∅), which is correct for
Tile-input buffer names, constants, and other read-only values.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, Select, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import find_matmul_k_outer, is_matmul_reduce, single_tile
from deplodock.compiler.tuning import detect_pat, register_tile_factor

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    # ``detect_pat`` reads whichever PAT 005_blockify_launch landed on
    # by inspecting THREAD axis extents; ``register_tile_factor`` pairs
    # F with that PAT through the centralized table in ``tuning``.
    pat = detect_pat(tile)
    if pat is None:
        raise RuleSkipped("detect_pat returned None — Tile axes don't match a known PAT")
    factor = register_tile_factor(tile)
    if factor <= 1 or pat % factor != 0:
        raise RuleSkipped(f"register-tile factor F={factor} disabled or doesn't divide PAT={pat}")

    target_axes = [ba.axis.name for ba in tile.axes if ba.bind == BIND_THREAD and int(ba.axis.extent) == pat]
    if len(target_axes) != 2:
        raise RuleSkipped(f"need exactly 2 THREAD axes with extent {pat}, found {len(target_axes)}")

    matmul_loc = _find_matmul_k_outer(tile.body)
    if matmul_loc is None:
        raise RuleSkipped("no pure matmul-shaped reduce Loop at Tile body top level")

    # Conservative smem-budget gate. Replicating a pre_outer reduce
    # Loop introduces an F-multiplied Stage in ``010_stage_inputs``; the
    # Stage gets a ``cell:F`` dim added to its slab so its smem grows
    # ~F× per replicated reduce. With double-buffering and bank padding
    # downstream, even small kernels (SDPA at seq=128, head_dim=64)
    # exceed the 48 KB consumer limit. Until 008 has hypothetical smem
    # accounting against ``010``'s would-be Stages, skip when pre_outer
    # has reduce-bearing Loops. Pure matmul (no pre_outer) and matmul +
    # scalar-Load preamble (e.g. ``silu(A@B)``) are unaffected.
    pre_outer = tile.body[:matmul_loc]
    for s in pre_outer:
        if _has_reduce_loop(s):
            raise RuleSkipped(
                "pre_outer contains a reduce Loop — F-replicating its Stage "
                "would exceed the 48 KB smem budget after pad+double-buffer "
                "(needs explicit smem accounting before relaxing)"
            )

    rewritten = _register_tile(tile, target_axes[0], target_axes[1], factor)
    if rewritten is None:
        raise RuleSkipped("_register_tile bailed (unsupported shape)")
    return body[:idx] + (rewritten,) + body[idx + 1 :]


def _has_reduce_loop(s: Stmt) -> bool:
    """True iff ``s`` is or contains a reduce Loop in its subtree."""
    if isinstance(s, Loop) and s.is_reduce:
        return True
    for c in Body((s,)).iter():
        if isinstance(c, Loop) and c.is_reduce and c is not s:
            return True
    return False


def _find_matmul_k_outer(body: Body) -> int | None:
    """Return the index of the top-level free Loop wrapping a matmul-
    shaped reduce, or ``None`` if no such Loop exists.

    Built on :func:`find_matmul_k_outer` from ``_helpers`` (which
    enforces the structural shape — single K-inner reduce + pure-
    compute body of Load/Assign/Accum) plus rule-specific gates layered
    via ``extra_gate``:

    - ≥2 K-indexed buffers (matmul operand-pair signature, via
      :func:`is_matmul_reduce`).
    - Exactly one Accum (the rewrite emits one accumulator chain per
      output cell; multi-Accum bodies aren't supported here).

    The body-purity check (no Select / Cond / nested Loops / staged-
    buffer rereads) lives in the shared helper. Allowing Select would
    fire on SDPA-with-rotary kernels — semantically safe since
    ``Select.rewrite`` handles per-cell σ substitution, but the
    surrounding Stages get cell-dim factors added to their slab
    layouts; with rotary's 10+ stages the post-pad smem usage blows
    past the 48 KB consumer limit. Gate when staging is bounded.

    Pre-K_outer external reads (e.g. SDPA's ``acc_max`` / ``acc_sum``
    from sibling reduces) are *not* a gate — the new axis-aware
    rewrite handles them via per-cell SSA-name suffixing. Smem-budget
    concerns for replicating pre_outer reduce Stages are gated
    separately in :func:`_maybe_rewrite`.
    """

    def gate(k_outer: Loop, k_inner: Loop) -> bool:
        if not is_matmul_reduce(k_inner):
            return False
        if sum(1 for c in k_inner.body if isinstance(c, Accum)) != 1:
            return False
        return True

    return find_matmul_k_outer(body, extra_gate=gate)


def _split_axis(axes: tuple[BoundAxis, ...], target: str, factor: int) -> tuple[tuple[BoundAxis, ...], Axis]:
    """Replace ``BoundAxis(target:E, THREAD)`` with ``BoundAxis(target_o:E/F, THREAD)``.
    Returns (new_axes, outer_axis)."""
    new_axes: list[BoundAxis] = []
    outer: Axis | None = None
    for ba in axes:
        if ba.axis.name == target:
            ext = int(ba.axis.extent)
            outer = Axis(f"{target}_o", ext // factor)
            new_axes.append(BoundAxis(axis=outer, bind=BIND_THREAD))
        else:
            new_axes.append(ba)
    assert outer is not None
    return tuple(new_axes), outer


def _register_tile(tile: Tile, m_axis: str, n_axis: str, factor: int) -> Tile | None:
    new_axes, m_o = _split_axis(tile.axes, m_axis, factor)
    new_axes, n_o = _split_axis(new_axes, n_axis, factor)

    k_outer_idx = _find_matmul_k_outer(tile.body)
    if k_outer_idx is None:
        return None
    k_outer = tile.body[k_outer_idx]

    # K-outer body: stages stay CTA-scoped; inner reduce body replicates per cell.
    new_outer_body: list[Stmt] = []
    k_inner: Loop | None = None
    for s in k_outer.body:
        if isinstance(s, Stage):
            new_outer_body.append(s)
        elif isinstance(s, Loop) and s.is_reduce:
            k_inner = s
        else:
            return None  # unsupported shape — bail rather than corrupt
    assert k_inner is not None

    pre_outer = tile.body[:k_outer_idx]
    post_outer = tile.body[k_outer_idx + 1 :]

    # Build per-name axis dependence over pre_outer (transitive through
    # SSA chains: an Assign / Accum / Select inherits the union of its
    # args' axes_used). Loop / StridedLoop axes are masked out.
    target_axes = frozenset({m_axis, n_axis})
    name_axes: dict[str, frozenset[str]] = _build_name_axes(pre_outer, target_axes)

    # Stmt-level axes_used = union of axes_used over names defined inside
    # that stmt. For Loop wrappers around an Accum (SDPA softmax pre-pass),
    # this picks up the Accum's axis-dep; for top-level Loads / Assigns,
    # it picks up that stmt's own dep.
    pre_axes_used: list[frozenset[str]] = []
    for s in pre_outer:
        defs = set(Body((s,)).definitions)
        if defs:
            axes = frozenset()
            for nm in defs:
                axes = axes | name_axes.get(nm, frozenset())
            pre_axes_used.append(axes)
        else:
            # No SSA defs inside (e.g. a bare Write at top level — unusual);
            # fall back to direct expression-level walk.
            pre_axes_used.append(_axes_used_in_stmt(s, target_axes))

    # K-inner / post_outer locals get the full F² axis-dep — they're
    # produced once per output cell.
    k_inner_locals = set(k_inner.body.definitions)
    for nm in k_inner_locals:
        name_axes[nm] = target_axes
    post_locals = set(post_outer.definitions)
    for nm in post_locals:
        name_axes[nm] = target_axes

    # Replicate K-inner body F² (matmul output is per-cell).
    cells_full = [(i, j) for i in range(factor) for j in range(factor)]
    new_k_inner_body: list[Stmt] = []
    for i, j in cells_full:
        sigma = _cell_sigma(target_axes, m_axis, m_o, n_axis, n_o, i, j, factor)
        rename = _make_rename(name_axes, k_inner_locals, target_axes, m_axis, n_axis, i, j)
        for s in k_inner.body:
            new_k_inner_body.append(s.rewrite(rename, sigma))
    new_outer_body.append(Loop(axis=k_inner.axis, body=new_k_inner_body, unroll=k_inner.unroll))

    # Replicate pre_outer per stmt by its axes_used.
    new_body: list[Stmt] = []
    for s, axes in zip(pre_outer, pre_axes_used, strict=True):
        for i, j in _cell_coords(axes, m_axis, n_axis, factor):
            sigma = _cell_sigma(axes, m_axis, m_o, n_axis, n_o, i, j, factor)
            stmt_locals = set(Body((s,)).definitions)
            rename = _make_rename(name_axes, stmt_locals, axes, m_axis, n_axis, i, j)
            new_body.append(s.rewrite(rename, sigma))

    new_body.append(Loop(axis=k_outer.axis, body=new_outer_body, unroll=k_outer.unroll))

    # Replicate post_outer F² (epilogue depends on per-cell matmul output).
    for i, j in cells_full:
        sigma = _cell_sigma(target_axes, m_axis, m_o, n_axis, n_o, i, j, factor)
        rename = _make_rename(name_axes, post_locals, target_axes, m_axis, n_axis, i, j)
        for s in post_outer:
            new_body.append(s.rewrite(rename, sigma))

    return Tile(axes=new_axes, body=new_body)


def _build_name_axes(pre_outer: Body, target_axes: frozenset[str]) -> dict[str, frozenset[str]]:
    """Per-name axis dependence over ``pre_outer``, transitively closed
    through SSA chains.

    Walks pre_outer in order, recording for each defined SSA name the
    subset of ``target_axes`` its value depends on. Direct dependence
    comes from a ``Load``'s index Vars (after subtracting Loop-bound
    axes); indirect dependence flows through ``Assign.args`` / ``Accum.value``
    / ``Select.branches[].value``.

    A name not present in the returned dict is either a Tile-input
    buffer name (passthrough) or referenced before it was defined
    (which is a structural error — unhandled).
    """
    name_axes: dict[str, frozenset[str]] = {}

    def axes_of_expr(e, bound: frozenset[str]) -> frozenset[str]:
        return frozenset(v for v in e.free_vars() if v in target_axes and v not in bound)

    def axes_of_name(n: str) -> frozenset[str]:
        return name_axes.get(n, frozenset())

    def visit(node: Stmt, bound: frozenset[str]) -> None:
        if isinstance(node, (Loop, StridedLoop)):
            for c in node.body:
                visit(c, bound | {node.axis.name})
            return
        if isinstance(node, Cond):
            for c in node.body:
                visit(c, bound)
            for c in node.else_body:
                visit(c, bound)
            return
        if isinstance(node, Tile):
            new_bound = bound | {ba.axis.name for ba in node.axes}
            for c in node.body:
                visit(c, new_bound)
            return
        if isinstance(node, Load):
            axes: frozenset[str] = frozenset()
            for e in node.index:
                axes = axes | axes_of_expr(e, bound)
            name_axes[node.name] = axes
            return
        if isinstance(node, Assign):
            axes = frozenset()
            for a in node.args:
                axes = axes | axes_of_name(a)
            name_axes[node.name] = axes
            return
        if isinstance(node, Accum):
            existing = name_axes.get(node.name, frozenset())
            name_axes[node.name] = existing | axes_of_name(node.value)
            return
        if isinstance(node, Select):
            axes = frozenset()
            for b in node.branches:
                axes = axes | axes_of_name(b.value) | axes_of_expr(b.select, bound)
            name_axes[node.name] = axes
            return
        # Init / Stage / Write / others — no SSA def to record at this level.

    for s in pre_outer:
        visit(s, frozenset())
    return name_axes


def _axes_used_in_stmt(s: Stmt, target_axes: frozenset[str]) -> frozenset[str]:
    """Subset of ``target_axes`` (axis-name set) appearing as free Vars
    in any expression inside ``s``, excluding axes bound by enclosing
    Loop/StridedLoop wrappers inside ``s`` itself.

    A pre_outer reduce ``Loop(a2, body=[Load(buf[a1, a2]), Accum])``
    has ``a2`` bound by the Loop; only ``a1`` survives the filter and
    if ``a1`` is in ``target_axes`` it appears in the result.
    """
    used: set[str] = set()

    def walk(node: Stmt, bound: frozenset[str]) -> None:
        if isinstance(node, (Loop, StridedLoop)):
            for c in node.body:
                walk(c, bound | {node.axis.name})
            return
        if isinstance(node, Cond):
            for v in node.cond.free_vars():
                if v in target_axes and v not in bound:
                    used.add(v)
            for c in node.body:
                walk(c, bound)
            for c in node.else_body:
                walk(c, bound)
            return
        if isinstance(node, Tile):
            new_bound = bound | {ba.axis.name for ba in node.axes}
            for c in node.body:
                walk(c, new_bound)
            return
        # Leaf-ish stmts: pull every Expr-bearing field's free Vars.
        for e in _exprs_in(node):
            for v in e.free_vars():
                if v in target_axes and v not in bound:
                    used.add(v)

    walk(s, frozenset())
    return frozenset(used)


def _exprs_in(s: Stmt):
    """Yield every direct Expr field of ``s`` (not recursive into bodies)."""
    if isinstance(s, Load):
        yield from s.index
    elif isinstance(s, Write):
        yield from s.index
    elif isinstance(s, Select):
        for b in s.branches:
            yield b.select
    elif isinstance(s, Stage):
        # Stage carries an index expression for its load source.
        yield from getattr(s, "index", ())
    # Assign / Accum / Init have only SSA-name fields (no Exprs).


def _cell_coords(axes_used: frozenset[str], m_axis: str, n_axis: str, factor: int) -> list[tuple[int, int]]:
    """Cell coords to replicate over for a stmt with the given axis-dep.

    Returns (i, j) tuples; ``i`` defaults to 0 when m_axis ∉ axes_used,
    same for ``j``. Defaulting to 0 (rather than None) keeps the σ
    well-formed at the unused axis — its substitution simply produces
    ``axis_o*F + 0`` which is the correct un-replicated reference.
    """
    if not axes_used:
        return [(0, 0)]
    if axes_used == {m_axis}:
        return [(i, 0) for i in range(factor)]
    if axes_used == {n_axis}:
        return [(0, j) for j in range(factor)]
    return [(i, j) for i in range(factor) for j in range(factor)]


def _cell_sigma(
    axes_used: frozenset[str],
    m_axis: str,
    m_o: Axis,
    n_axis: str,
    n_o: Axis,
    i: int,
    j: int,
    factor: int,
) -> Sigma:
    """σ-substitution for cell (i, j) restricted to ``axes_used``."""
    sub: dict[str, object] = {}
    if m_axis in axes_used:
        sub[m_axis] = Var(m_o.name) * Literal(factor, "int") + Literal(i, "int")
    if n_axis in axes_used:
        sub[n_axis] = Var(n_o.name) * Literal(factor, "int") + Literal(j, "int")
    return Sigma(sub) if sub else Sigma.IDENTITY


def _cell_suffix(axes_used: frozenset[str], m_axis: str, n_axis: str, i: int, j: int) -> str:
    """Per-cell SSA-name suffix matching the axis subset."""
    if not axes_used:
        return ""
    if axes_used == {m_axis}:
        return f"_{i}"
    if axes_used == {n_axis}:
        return f"_{j}"
    return f"_{i}_{j}"


def _make_rename(
    name_axes: dict[str, frozenset[str]],
    locals_in_scope: set[str],
    cell_axes: frozenset[str],
    m_axis: str,
    n_axis: str,
    i: int,
    j: int,
):
    """Build a per-cell SSA rename callback.

    ``locals_in_scope`` are the names defined inside the region being
    rewritten — they get renamed to the suffix corresponding to
    ``cell_axes`` (the region's axis-dep: F² for K-inner / post_outer
    and F-or-1 for pre_outer reduce-loop replicas). External reads
    (names defined elsewhere) get the suffix matching the *defining*
    stmt's axis-dep recorded in ``name_axes``.
    """
    local_suffix = _cell_suffix(cell_axes, m_axis, n_axis, i, j)

    def rename(name: str) -> str:
        if name in locals_in_scope:
            return f"{name}{local_suffix}" if local_suffix else name
        axes = name_axes.get(name)
        if axes is None:
            return name  # passthrough — Tile-input or out-of-scope name
        suffix = _cell_suffix(axes, m_axis, n_axis, i, j)
        return f"{name}{suffix}" if suffix else name

    return rename
