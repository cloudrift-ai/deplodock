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
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Load, Loop, Select, Stmt, StridedLoop, Tile, Write
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
    return any(isinstance(c, Loop) and c.is_reduce for c in Body((s,)).iter())


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

    target_axes = frozenset({m_axis, n_axis})
    k_inner_locals = set(k_inner.body.definitions)
    post_locals = set(post_outer.definitions)
    name_axes = _build_name_axes(pre_outer, k_inner_locals, post_locals, target_axes)

    rewriter = CellRewriter(m_axis=m_axis, n_axis=n_axis, m_o=m_o, n_o=n_o, factor=factor, name_axes=name_axes)

    pre_triples = [_pre_triple(s, name_axes, target_axes) for s in pre_outer]
    k_inner_triples = [(s, target_axes, set(Body((s,)).definitions)) for s in k_inner.body]
    post_triples = [(s, target_axes, set(Body((s,)).definitions)) for s in post_outer]

    new_outer_body.append(Loop(axis=k_inner.axis, body=_replicate(k_inner_triples, rewriter), unroll=k_inner.unroll))

    new_body: list[Stmt] = _replicate(pre_triples, rewriter)
    new_body.append(Loop(axis=k_outer.axis, body=new_outer_body, unroll=k_outer.unroll))
    new_body.extend(_replicate(post_triples, rewriter))

    return Tile(axes=new_axes, body=new_body)


def _pre_triple(s: Stmt, name_axes: dict[str, frozenset[str]], target_axes: frozenset[str]) -> tuple[Stmt, frozenset[str], set[str]]:
    """Cell-axes / locals descriptor for one pre_outer stmt.

    Stmt-level axes_used = union of axes_used over names defined inside
    that stmt. For Loop wrappers around an Accum (SDPA softmax pre-pass),
    this picks up the Accum's axis-dep; for top-level Loads / Assigns,
    it picks up that stmt's own dep. Falls back to a direct expression-
    level walk when no SSA defs exist (e.g. a bare top-level Write)."""
    defs = set(Body((s,)).definitions)
    if defs:
        axes: frozenset[str] = frozenset()
        for nm in defs:
            axes = axes | name_axes.get(nm, frozenset())
    else:
        axes = _axes_used_in_stmt(s, target_axes)
    return s, axes, defs


def _replicate(triples: list[tuple[Stmt, frozenset[str], set[str]]], rewriter: CellRewriter) -> list[Stmt]:
    """Per-stmt cell replication. Each (stmt, cell_axes, locals_in_scope)
    triple is rewritten once per cell coord that ``rewriter.cells(cell_axes)``
    enumerates — F² for stmts depending on both M and N, F for one-axis
    deps, 1 for axis-free stmts."""
    out: list[Stmt] = []
    for s, axes, loc in triples:
        for i, j in rewriter.cells(axes):
            out.append(rewriter.rewrite(s, i, j, loc, axes))
    return out


def _build_name_axes(
    pre_outer: Body,
    k_inner_locals: set[str],
    post_locals: set[str],
    target_axes: frozenset[str],
) -> dict[str, frozenset[str]]:
    """Per-name axis dependence covering pre_outer + K-inner + post_outer.

    pre_outer: built via :meth:`Body.fold` — one walk threads ``bound``
    axes through enclosing Loop / StridedLoop / Tile and combines each
    stmt's local axis contribution (Load index Vars filtered by ``bound``,
    Select branch predicates) with the per-dep axis sets pulled from the
    memo. Filter to ``target_axes`` lives in the callback.

    K-inner / post_outer locals are produced once per output cell, so
    they get the full ``target_axes`` axis-dep regardless of upstream
    dependencies — overwritten directly after the fold.

    A name not present in the returned dict is either a Tile-input
    buffer name (passthrough) or referenced before it was defined
    (which is a structural error — unhandled).
    """

    def axes_of_expr(e, bound: frozenset[str]) -> frozenset[str]:
        return frozenset(v for v in e.free_vars() if v in target_axes and v not in bound)

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        own: frozenset[str] = frozenset()
        if isinstance(s, Load):
            for e in s.index:
                own = own | axes_of_expr(e, bound)
        elif isinstance(s, Select):
            for b in s.branches:
                own = own | axes_of_expr(b.select, bound)
        # Assign / Accum / others contribute no direct Expr-level axes —
        # their dependence comes entirely from child_T.
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    memo = pre_outer.fold(fn)
    name_axes: dict[str, frozenset[str]] = {n: memo[id(s)] for s in pre_outer.iter() for n in s.defines()}
    for nm in k_inner_locals:
        name_axes[nm] = target_axes
    for nm in post_locals:
        name_axes[nm] = target_axes
    return name_axes


def _axes_used_in_stmt(s: Stmt, target_axes: frozenset[str]) -> frozenset[str]:
    """Subset of ``target_axes`` (axis-name set) appearing as free Vars
    in any expression inside ``s``'s subtree, excluding axes bound by
    enclosing Loop / StridedLoop / Tile wrappers inside ``s`` itself.

    A pre_outer reduce ``Loop(a2, body=[Load(buf[a1, a2]), Accum])``
    has ``a2`` bound by the Loop; only ``a1`` survives the filter and
    if ``a1`` is in ``target_axes`` it appears in the result.

    Recursive over the nesting tree (``s.nested()``) — *not* the
    def-use DAG — so ``Body.fold`` doesn't fit (its ``child_T`` are
    SSA-deps, not nested-body values). Threads ``bound`` via
    ``Stmt.binds_axes()``, which keeps the recursion type-set-agnostic
    (no isinstance ladder over the block-stmt set)."""
    used: set[str] = set()

    def walk(node: Stmt, bound: frozenset[str]) -> None:
        for e in _exprs_in(node):
            for v in e.free_vars():
                if v in target_axes and v not in bound:
                    used.add(v)
        new_bound = bound | node.binds_axes()
        for child_body in node.nested():
            for c in child_body:
                walk(c, new_bound)

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
    elif isinstance(s, Cond):
        yield s.cond
    elif isinstance(s, Stage):
        # Stage carries an index expression for its load source.
        yield from getattr(s, "index", ())
    # Assign / Accum / Init have only SSA-name fields (no Exprs).


class CellRewriter:
    """Per-cell σ-substitution + SSA-rename packaged together.

    Folds the previous ``_cell_coords`` / ``_cell_sigma`` / ``_cell_suffix``
    / ``_make_rename`` quartet into a single object holding the split-axis
    geometry (``m_o`` / ``n_o`` / ``factor``) and the resolved
    ``name_axes`` map. Call sites get two methods:

    - :meth:`cells` → list of ``(i, j)`` coords for a given axis-dep.
    - :meth:`rewrite` → rewrite one stmt at one cell with proper
      σ + per-cell SSA rename.
    """

    def __init__(
        self,
        m_axis: str,
        n_axis: str,
        m_o: Axis,
        n_o: Axis,
        factor: int,
        name_axes: dict[str, frozenset[str]],
    ) -> None:
        self.m_axis = m_axis
        self.n_axis = n_axis
        self.m_o = m_o
        self.n_o = n_o
        self.factor = factor
        self.name_axes = name_axes

    def cells(self, axes_used: frozenset[str]) -> list[tuple[int, int]]:
        """Cell coords to replicate over for a stmt with the given axis-dep.

        Defaults the unused axis to 0 (rather than None) so σ stays
        well-formed at that axis — substitution produces ``axis_o*F + 0``
        which is the correct un-replicated reference."""
        if not axes_used:
            return [(0, 0)]
        if axes_used == {self.m_axis}:
            return [(i, 0) for i in range(self.factor)]
        if axes_used == {self.n_axis}:
            return [(0, j) for j in range(self.factor)]
        return [(i, j) for i in range(self.factor) for j in range(self.factor)]

    def _suffix(self, axes: frozenset[str], i: int, j: int) -> str:
        if not axes:
            return ""
        if axes == {self.m_axis}:
            return f"_{i}"
        if axes == {self.n_axis}:
            return f"_{j}"
        return f"_{i}_{j}"

    def _sigma(self, cell_axes: frozenset[str], i: int, j: int) -> Sigma:
        sub: dict[str, object] = {}
        if self.m_axis in cell_axes:
            sub[self.m_axis] = Var(self.m_o.name) * Literal(self.factor, "int") + Literal(i, "int")
        if self.n_axis in cell_axes:
            sub[self.n_axis] = Var(self.n_o.name) * Literal(self.factor, "int") + Literal(j, "int")
        return Sigma(sub) if sub else Sigma.IDENTITY

    def rewrite(self, s: Stmt, i: int, j: int, locals_in_scope: set[str], cell_axes: frozenset[str]) -> Stmt:
        """Rewrite ``s`` for cell ``(i, j)``.

        ``locals_in_scope`` are the names defined inside the region being
        rewritten — they get the suffix corresponding to ``cell_axes``
        (the region's axis-dep: F² for K-inner / post_outer and F-or-1
        for pre_outer replicas). External reads (names defined elsewhere)
        get the suffix matching their defining stmt's axis-dep recorded
        in ``name_axes``."""
        sigma = self._sigma(cell_axes, i, j)
        local_suffix = self._suffix(cell_axes, i, j)
        name_axes = self.name_axes

        def rename(name: str) -> str:
            if name in locals_in_scope:
                return f"{name}{local_suffix}" if local_suffix else name
            axes = name_axes.get(name)
            if axes is None:
                return name  # passthrough — Tile-input or out-of-scope name
            suffix = self._suffix(axes, i, j)
            return f"{name}{suffix}" if suffix else name

        return s.rewrite(rename, sigma)
