"""Lower the tensor-core matmul cell to the kernel-IR MMA fragment chain.

The matmul cell arrives in tensor-core form: ``tile/011_lower_atom_cell`` fused
the compute into an :class:`~deplodock.compiler.ir.stmt.Mma` (which names its A
/ B operands by SSA value and carries the ``Atom`` spec) and left the operand
``Load``s **plain**, and the staging passes carried both through (the loads
staged like any other, the ``Mma`` keeping its reduce loop ``is_reduce``). The
partition planner emits one of four cell shapes:

A. Direct K-loop, no staging (operands read from gmem via a gmem-direct fragment
   load — see below):
   ``AtomTile > SerialTile(K_o) > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
B. Single-bundle staged (SYNC or single-buffer ASYNC):
   ``AtomTile > SerialTile(K_o) > StageBundle > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
C. Filtered K (single-iter, no SerialTile): ``AtomTile > [Load a*, Load b*, Mma]`` (inline)
D. Pipelined + buffered (cp.async double-buffered): prologue StageBundle, K_o-1
   loop with issue-next StageBundle + AsyncWait + reduce, AsyncWait, epilogue reduce, Write.

Any of the four shapes may additionally carry a masked-tile boundary ``Cond``
(``σ(axis) < bound``) around the cell or — after
``021_hoist_staged_loads_above_mask`` lifted the K-pipeline out — around just
the ``Write``. The Cond stays in place (a free skip of fully-out-of-range
tiles); ``_boundary_guards`` classifies its predicate against the Write's
M / N coordinates and stamps the per-element range check onto the ``RegStore``
(``m_guard`` / ``n_guard``), because the Cond only gates the atom tile's base
coordinate while a *straddling* tile's trailing rows / cols are out of range.
A masked cell whose gated-axis operand was NOT staged takes the clamped
gmem-direct helper (``LdmatrixLoad.gmem_guard`` → ``dpl_mma_load_*_*clamp``)
— the staged slab is in-bounds by construction, the gmem-direct fallback
clamps its lane coordinate instead — so every enumerated variant lowers.

This pass walks the ``TileOp`` body, finds each ``AtomTile``, and rewrites its
cell into ``RegFragment`` decls + per-reduce ``LdmatrixLoad a + LdmatrixLoad b
+ MmaSyncPtx`` + final ``RegStore``, then strips the ``AtomTile`` wrapper. The
fragment SSA names are seeded once from the FIRST reduce site (stable across
prologue/inner/epilogue, which is what the per-cell replicator in
``010_split_register_axes`` expects). Operands are matched per reduce site by
the co-located ``Mma``'s A / B SSA names; the slab ``src_index`` / ``ldm`` come from re-harvesting
the live ``Source`` (``_mma_src_index``), so the phase-prefix prepend for
double-buffered slabs stays correct.

The s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` path is the sole
tensor-core family. ``ldmatrix`` is smem→register only, so :func:`_emit_chain`
picks each operand's transport by whether an enclosing ``StageBundle`` staged
it: staged → ``LdmatrixLoad`` (smem); unstaged → ``LdmatrixLoad(staged=False)``,
which renders a **gmem-direct fragment load** (``dpl_mma_load_{a,b}_gmem``,
replicating the PTX m16n8k16 lane→element map without ldmatrix). That way an
MMA tile whose operands the staging passes declined to stage (e.g. slabs don't
fit the smem budget) still compiles — slower than the staged path (no smem
reuse), but correct — instead of crashing, so the search needn't avoid it.

EXCEPTION — masked-K: the gmem-direct load only clamps the M/N (output) lane
axis (``gmem_guard``); it has NO K zero-fill. So an unstaged operand whose
REDUCE (K) dim is symbolic (SDPA P@V over ``seq_len``) would read the partial
final K tile past the mask-padded extent — garbage (wrong result / hang) or an
OOB illegal access (the masked-K mma bench_fails). Only the staged path
zero-fills the partial slab (``_stage_expand``). :func:`_unstaged_masked_k`
detects this and :func:`_emit_chain` raises ``LoweringError`` to drop the
candidate, so the search uses the staged (zero-filling) transport or the scalar
path instead.

Each cell's :class:`~deplodock.compiler.ir.tile.ir.Atom` spec (shape + operand
dtypes) is read straight off its ``Mma`` — no ``ATOM_KIND`` knob lookup. The
``rewrite`` entry point and its lowering helpers all live in this one module.
Eligibility: an ``AtomTile`` in the body (scalar TileOps have none → skip).
Idempotence: after this pass the ``AtomTile`` is gone, so a second visit finds
nothing and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.dtype import DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel.ir import EpilogueLoad, LdmatrixLoad, MmaSyncPtx, RegEpilogue, RegFragment, RegStore
from deplodock.compiler.ir.stmt import Body, Cond, Load, Mma, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Atom,
    AtomTile,
    SerialTile,
    Source,
    StageBundle,
    TileOp,
    map_staged,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    # No knob lookup: each cell's atom spec is read off its ``Mma`` (the
    # ``AtomTile`` presence is the signal). Scalar / already-lowered TileOps
    # have no ``AtomTile`` → found is False → skip.
    lowered, found = lower_atom_cells(op.body, smem_sources={}, graph=match.graph)
    if not found:
        raise RuleSkipped("no AtomTile in body (scalar, or already lowered)")
    return TileOp(body=lowered, name=op.name, knobs=op.knobs)


def lower_atom_cells(body: Body, *, smem_sources: dict[str, Source], graph: Graph | None = None) -> tuple[Body, bool]:
    """For each ``AtomTile`` in ``body``, lower its matmul cell to the kernel-IR
    fragment chain and strip the wrapper. Runs through the shared
    :func:`~deplodock.compiler.ir.tile.ir.map_staged` traversal, which threads
    the in-scope ``Source`` table from enclosing ``StageBundle`` /
    ``WarpSpecialize`` scopes so each operand's slab addressing resolves from the
    live ``Source``. The atom spec is read off each cell's ``Mma`` (no knob
    lookup). Returns ``(new_body, found_any)``."""
    found = False
    outer_loads = _collect_outer_loads(body)

    def handler(s: Stmt, sources: dict[str, Source]) -> tuple[Stmt, ...] | None:
        nonlocal found
        if isinstance(s, AtomTile):
            found = True
            return _lower_cell(s.body, smem_sources=sources, graph=graph, outer_loads=outer_loads)
        return None

    return map_staged(body, handler, sources=smem_sources), found


def _collect_outer_loads(body: Body) -> dict[str, Load]:
    """Single-name ``Load``s in ``body`` that sit OUTSIDE every ``AtomTile``
    and outside any reduce loop — the loop-invariant scalar prologue loads the
    splicer parks at the TileOp root (a real trace's f32 constants). Passed to
    the epilogue classifier as fallback leaf definitions so the fold sees the
    same leaves the Loop-IR eligibility gate admitted (SSA names are unique
    per kernel, so a flat name map is unambiguous)."""
    outer: dict[str, Load] = {}

    def _walk(stmts: Body) -> None:
        for s in stmts:
            if isinstance(s, AtomTile):
                continue
            if isinstance(s, Load) and len(s.names) == 1:
                outer.setdefault(s.names[0], s)
                continue
            if isinstance(s, SerialTile) and s.is_reduce:
                continue
            for sub in s.nested():
                _walk(sub)

    _walk(body)
    return outer


def _lower_cell(
    atom_body: Body, *, smem_sources: dict[str, Source], graph: Graph | None = None, outer_loads: dict[str, Load] | None = None
) -> tuple[Stmt, ...]:
    """Lower one AtomTile body (operand Loads + ``Mma``) to the fragment chain.

    The atom :class:`Atom` spec comes from the cell's ``Mma``. Prepends the
    ``RegFragment`` decls (seeding stable fragment SSA names from the FIRST
    reduce site, so the per-cell replicator in ``010_split_register_axes``
    renames them consistently across prologue/inner/epilogue), then for shapes
    A/B/D maps the body (each reduce ``SerialTile`` → chain, ``Write`` →
    ``RegStore``); shape C inlines a single chain + store."""
    # Flatten *every* bundle in the cell up front: a double-buffered (shape D)
    # epilogue reduce is a sibling of the staging bundles, not nested inside
    # them, so descent-threading alone wouldn't reach the slab Sources. Seed the
    # rewrite with this union so a sibling cell still resolves its operands. The
    # gather rides ``map_staged`` as a pure visitor — the rebuilt body is
    # discarded; ``_gather`` accumulates every bundle's Sources via closure.
    bundle_sources = dict(smem_sources)

    def _gather(s: Stmt, _scope: dict[str, Source]) -> None:
        if isinstance(s, StageBundle):
            for src in s.sources:
                bundle_sources[src.name] = src
        return None

    map_staged(atom_body, _gather)

    write_stmt, a_seed, b_seed, c_seed, has_reduce, spec = _scan_cell(atom_body)
    if write_stmt is None:
        raise RuleSkipped("AtomTile body unrecognised — no Write")
    if spec is None or a_seed is None or b_seed is None or c_seed is None:
        raise RuleSkipped(f"AtomTile body unrecognised — expected operand Loads + Mma (got a={a_seed!r}, b={b_seed!r}, c={c_seed!r})")
    m_guard, n_guard = _boundary_guards(atom_body, write_stmt)
    epilogue, strip_ids = _scan_epilogue(atom_body, acc_name=c_seed, graph=graph, outer_loads=outer_loads)

    c_frag, a_frag, b_frag = f"{c_seed}_frag", f"{a_seed}_frag", f"{b_seed}_frag"
    fragments = _emit_fragments(spec, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, c_dtype=spec.operand_dtype("c"))

    if has_reduce:

        def handler(s: Stmt, sources: dict[str, Source]) -> tuple[Stmt, ...] | None:
            # The fused pointwise epilogue rides the RegStore (see
            # _scan_epilogue); its scalar Loads / Assigns would reference the
            # accumulator SSA name the fragment path never defines — drop them.
            if id(s) in strip_ids:
                return ()
            if isinstance(s, Write):
                return (
                    _emit_store(
                        spec, dst_buffer=s.output, dst_index=s.index, c_frag=c_frag, epilogue=epilogue, m_guard=m_guard, n_guard=n_guard
                    ),
                )
            if isinstance(s, SerialTile) and s.is_reduce:
                chain = _build_chain(
                    s.body,
                    c_frag=c_frag,
                    a_frag=a_frag,
                    b_frag=b_frag,
                    smem_sources=sources,
                    spec=spec,
                    m_guard=m_guard,
                    n_guard=n_guard,
                    graph=graph,
                )
                return (s.with_bodies((Body(chain),)),)
            return None

        transformed = map_staged(atom_body, handler, sources=bundle_sources)
        return (*fragments, *transformed)

    # Shape C: K filtered — inline chain from the body's operand Loads + store.
    # A masked cell's boundary Cond is dropped here (the rebuilt body is just
    # chain + store): correctness doesn't need it — gmem-direct loads clamp
    # via ``gmem_guard``, staged loads read the in-bounds slab, and the store
    # carries the per-element guards. The Cond was only a whole-tile skip.
    a_load, b_load, b_trans = _find_role_loads(atom_body)
    if a_load is None or b_load is None:
        raise RuleSkipped("Atom body (shape C) missing its Mma / A/B loads")
    chain = _emit_chain(
        spec,
        a_load=a_load,
        b_load=b_load,
        a_frag=a_frag,
        b_frag=b_frag,
        c_frag=c_frag,
        smem_sources=bundle_sources,
        m_guard=m_guard,
        n_guard=n_guard,
        b_trans=b_trans,
        graph=graph,
    )
    store = _emit_store(
        spec, dst_buffer=write_stmt.output, dst_index=write_stmt.index, c_frag=c_frag, epilogue=epilogue, m_guard=m_guard, n_guard=n_guard
    )
    return (*fragments, *chain, store)


def _boundary_guards(body: Body, write_stmt: Write) -> tuple[tuple[Expr, Expr] | None, tuple[Expr, Expr] | None]:
    """Classify the masked-tile boundary ``Cond``s enclosing the cell's Write
    into ``(m_guard, n_guard)`` for the ``RegStore``.

    A boundary Cond's predicate (``σ(axis) < bound``) gates only the atom
    tile's BASE coordinate — the fragment lane offsets (``_g`` / ``_t``) are
    render-local and invisible to σ — so a tile *straddling* the bound passes
    the Cond while its trailing rows / cols are out of range. The Cond stays
    in the body (a free skip of fully-out-of-range tiles); the per-element
    range check moves onto the store via the guards.

    Classification matches the predicate's LHS structurally against the
    Write's M / N coordinate exprs (the second-to-last / last VAR-BEARING
    index dims — the same convention as ``_atom.classify_fragment_epilogue``;
    real-trace outputs pad with literal dims, e.g. ``o[0, m, 0, n]``); both
    sides were produced by the same planner σ, so a mismatch means the
    planner and this pass disagree about the masked axis — fail loud
    (``RuleSkipped`` → the variant pins a ``bench_fail`` row) rather than
    emit an unguarded straddling store."""
    conds = [
        s
        for s in body.iter_of_type(Cond)
        if isinstance(s.cond, BinaryExpr) and s.cond.op == "<" and any(w is write_stmt for w in s.body.iter_of_type(Write))
    ]
    if not conds:
        return None, None
    var_dims = [e for e in write_stmt.index if e.free_vars()]
    if len(var_dims) < 2:
        raise RuleSkipped(f"masked AtomTile Write has {len(var_dims)} var-bearing index dims — no (m, n) coordinates to guard")
    m_expr, n_expr = var_dims[-2], var_dims[-1]
    m_guard: tuple[Expr, Expr] | None = None
    n_guard: tuple[Expr, Expr] | None = None
    for c in conds:
        lhs, rhs = c.cond.left, c.cond.right
        if lhs == m_expr and m_guard is None:
            m_guard = (lhs, rhs)
        elif lhs == n_expr and n_guard is None:
            n_guard = (lhs, rhs)
        else:
            raise RuleSkipped(f"masked AtomTile boundary predicate {c.cond.pretty()} matches no Write coordinate — gate out of sync?")
    return m_guard, n_guard


def _scan_cell(body: Body) -> tuple[Write | None, str | None, str | None, str | None, bool, Atom | None]:
    """Recursively scan the AtomTile body. Returns the first ``Write``, the
    seed ``(a_name, b_name, c_name)`` + the :class:`Atom` spec from the first
    ``Mma`` (every reduce site shares the same operand/accumulator SSA names and
    spec, so any ``Mma`` seeds stable fragment names), and ``has_reduce``."""
    write_stmt: Write | None = None
    seed: tuple[str, str, str] | None = None
    spec: Atom | None = None
    has_reduce = False

    def _walk(stmts: Body) -> None:
        nonlocal write_stmt, seed, spec, has_reduce
        for s in stmts:
            if isinstance(s, Write):
                if write_stmt is None:
                    write_stmt = s
                continue
            if isinstance(s, Mma):
                if seed is None:
                    seed = (s.a, s.b, s.c)
                    spec = s.atom
                continue
            if isinstance(s, SerialTile) and s.is_reduce:
                has_reduce = True
                _walk(s.body)
                continue
            if s.nested():
                for sub in s.nested():
                    _walk(sub)

    _walk(body)
    a_seed, b_seed, c_seed = seed if seed is not None else (None, None, None)
    return write_stmt, a_seed, b_seed, c_seed, has_reduce, spec


def _replace_subexprs(e, pairs):
    """Structurally replace whole sub-expressions: any node equal to a ``target``
    becomes its ``replacement``. Recurses through ``BinaryExpr`` (leaves pass
    through). Used to swap a Select predicate's M / N coordinate expressions for
    the ``__M__`` / ``__N__`` placeholders without touching their internal vars."""
    from deplodock.compiler.ir.expr import BinaryExpr  # noqa: PLC0415

    for tgt, rep in pairs:
        if e == tgt:
            return rep
    if isinstance(e, BinaryExpr):
        return BinaryExpr(e.op, _replace_subexprs(e.left, pairs), _replace_subexprs(e.right, pairs))
    return e


def _scan_epilogue(
    body: Body, *, acc_name: str, graph: Graph | None, outer_loads: dict[str, Load] | None = None
) -> tuple[RegEpilogue | None, frozenset[int]]:
    """Classify the AtomTile body's post-reduce epilogue via the shared
    negative-form classifier (``tile/_atom.classify_fragment_epilogue`` — the
    same walk the planner gate ran on the Loop-IR body, here on the staged
    Tile-IR body where the indices carry the partition decomposition). Returns
    the :class:`RegEpilogue` payload for the store plus the ``id()`` set of
    the scalar stmts to strip (the slice Assigns + leaf Loads, whose
    accumulator SSA name doesn't exist on the fragment path).

    The gate admits a shape only when the classifier reports no blocker, so a
    blocker here means the gate and this fold disagree — fail loud rather than
    emit a kernel referencing the undefined scalar accumulator."""
    from deplodock.compiler.ir.stmt import Write  # noqa: PLC0415
    from deplodock.compiler.pipeline.passes.lowering.kernel._atom import classify_fragment_epilogue  # noqa: PLC0415

    produced = {w.output for w in body.iter_of_type(Write)}

    def _leaf_dtype(buf: str) -> str | None:
        if graph is None or buf not in graph.nodes:
            return None
        return graph.nodes[buf].output.dtype.name

    slice_, blocker = classify_fragment_epilogue(body, {acc_name}, produced=produced, leaf_dtype=_leaf_dtype, outer_loads=outer_loads)
    if blocker is not None:
        raise RuleSkipped(f"AtomTile epilogue consuming {acc_name!r} is not foldable ({blocker}) — gate out of sync?")
    if slice_ is None:
        return None, frozenset()
    # Rewrite each folded Select's predicate from the cell's M / N coordinate
    # EXPRESSIONS to the ``__M__`` / ``__N__`` placeholders the RegStore
    # substitutes with the fragment element's own (row, col). Whole-subexpression
    # replacement, not var-level: post-partition a coordinate is a compound expr
    # (``a1*128 + a3*64 + …``), so replacing its vars one-by-one would corrupt
    # the arithmetic. The predicate's coordinate operands are struct-equal to the
    # Write's M / N index dims (same partition σ) — the last two var-bearing dims.
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

    w_var_dims = [e for e in slice_.write.index if e.free_vars()]
    pairs = []
    if len(w_var_dims) >= 2:
        pairs.append((w_var_dims[-2], Var("__M__")))
    if w_var_dims:
        pairs.append((w_var_dims[-1], Var("__N__")))
    selects = tuple((sel.name, tuple((_replace_subexprs(br.select, pairs), br.value) for br in sel.branches)) for sel in slice_.selects)
    # After replacement a predicate may reference only the ``__M__`` / ``__N__``
    # placeholders (+ constants). A leftover coordinate var means the structural
    # match failed (gate admitted a shape the fold can't render) — fail loud.
    for _name, branches in selects:
        for cond, _value in branches:
            if cond.free_vars() - {"__M__", "__N__"}:
                raise RuleSkipped(f"Select predicate {cond.pretty()} not reducible to (__M__, __N__) — gate out of sync?")
    epilogue = RegEpilogue(
        acc=slice_.acc,
        loads=tuple(
            EpilogueLoad(name=ld.names[0], buffer=ld.input, index=tuple(ld.index), roles=roles)
            for ld, roles in zip(slice_.loads, slice_.load_roles, strict=True)
        ),
        ops=tuple((a.name, a.op.name, tuple(a.args)) for a in slice_.assigns),
        result=slice_.write.value,
        selects=selects,
    )
    return epilogue, frozenset(map(id, (*slice_.assigns, *slice_.loads, *slice_.selects)))


def _find_role_loads(body: Body) -> tuple[Load | None, Load | None, bool]:
    """The A / B operand Loads of the cell in ``body`` (+ the B operand's
    ``b_trans`` flag) — identified via the co-located ``Mma``, which names its
    operands by SSA value (so the operand Loads need no tensor-core tag of their
    own). Recursive: a masked cell wraps the K-filtered (shape C) body in the
    boundary ``Cond``, so the Loads + ``Mma`` sit one level down — SSA names are
    unique per kernel, so the deep search is unambiguous."""
    mma = next(iter(body.iter_of_type(Mma)), None)
    if mma is None:
        return None, None, False
    by_name = {ld.names[0]: ld for ld in body.iter_of_type(Load) if ld.names}
    return by_name.get(mma.a), by_name.get(mma.b), mma.b_trans


def _build_chain(
    reduce_body: Body,
    *,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    smem_sources: dict[str, Source],
    spec: Atom,
    m_guard: tuple[Expr, Expr] | None = None,
    n_guard: tuple[Expr, Expr] | None = None,
    graph: Graph | None = None,
) -> tuple[Stmt, ...]:
    """Build the ``ldmatrix a + ldmatrix b + mma.sync`` chain that replaces a
    reduce SerialTile's ``[Load a, Load b, Mma]`` body — operands matched via
    the body's ``Mma`` (which names its A/B operands by SSA value)."""
    a_load, b_load, b_trans = _find_role_loads(reduce_body)
    if a_load is None or b_load is None:
        raise RuleSkipped("reduce SerialTile body missing its Mma / A/B Loads")
    return _emit_chain(
        spec,
        a_load=a_load,
        b_load=b_load,
        a_frag=a_frag,
        b_frag=b_frag,
        c_frag=c_frag,
        smem_sources=smem_sources,
        m_guard=m_guard,
        n_guard=n_guard,
        b_trans=b_trans,
        graph=graph,
    )


# --- Per-instruction leaf emitters -----------------------------------------


def _emit_fragments(spec: Atom, *, c_frag: str, a_frag: str, b_frag: str, c_dtype: DataType) -> tuple[Stmt, ...]:
    """Register-array declarations. Three ``RegFragment`` decls; the ``c``
    array is zero-initialised at declaration, so there's no separate fill."""
    return (
        RegFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
        RegFragment(name=a_frag, role="a", shape=spec.shape, dtype=spec.operand_dtype("a")),
        RegFragment(name=b_frag, role="b", shape=spec.shape, dtype=spec.operand_dtype("b")),
    )


def _unstaged_masked_k(load: Load, role: str, graph: Graph | None) -> bool:
    """An unstaged mma operand whose REDUCE (K) gmem dim is symbolic — the
    masked-K case (SDPA P@V over ``seq_len``).

    The gmem-direct fragment load (``dpl_mma_load_{a,b}_gmem*``) iterates K to the
    tile's ``BK·atom_k`` and only clamps the M/N (output, lane-varying) axis via
    ``gmem_guard`` — it has **no K zero-fill**. So a symbolic, mask-padded K is
    read past its runtime extent (``ceil(seq/64)*64`` < ``BK·atom_k``): garbage
    into the accumulation (wrong result, or a hang on the slow path) or an OOB
    illegal access. Only the STAGED path zero-fills the partial K slab
    (``_stage_expand``). Detect so the caller bails — forcing the staged
    transport, or the (correct) scalar fallback. A's K is the last index dim
    (canonical ``[…, M, K]``); B's K is a non-last dim (N is last)."""
    if graph is None:
        return False
    node = graph.nodes.get(load.input)
    if node is None or not node.output.shape:
        return False
    shape = node.output.shape
    if role == "a":
        return not shape[-1].is_static
    return any(not d.is_static for d in shape[:-1])


def _masked_k_zero(load: Load, src_index: tuple, role: str, graph: Graph | None) -> tuple[Expr, Expr] | None:
    """``(k_base, bound)`` for an unstaged masked-K operand's ``*_kzero`` helper:
    ``k_base`` is the tile's K coordinate (the operand's reduce-dim index, to which
    the helper adds the lane's K offset) and ``bound`` is the runtime K extent
    (``seq_len``) — so a half whose ``k_base + offset >= seq_len`` reads +0.0.
    Mirrors the staged ``Source.kmask`` zero-fill. K is the last index dim for
    role ``a``; the first symbolic non-last dim for role ``b`` (N is last). The
    bound is the sole free var of the (tile-padded) extent expr — the runtime
    ``seq_len``. ``None`` when not masked-K or the extent isn't a single symbol
    (caller bails)."""
    if graph is None:
        return None
    node = graph.nodes.get(load.input)
    if node is None or not node.output.shape:
        return None
    shape = node.output.shape
    if role == "a":
        kd = len(shape) - 1
        if shape[kd].is_static:
            return None
    else:
        kd = next((i for i, d in enumerate(shape[:-1]) if not d.is_static), None)
        if kd is None:
            return None
    if kd >= len(src_index):
        return None
    fvs = shape[kd].expr.free_vars()
    if len(fvs) != 1:
        return None
    return (src_index[kd], Var(next(iter(fvs))))


def _emit_chain(
    spec: Atom,
    *,
    a_load: Load,
    b_load: Load,
    a_frag: str,
    b_frag: str,
    c_frag: str,
    smem_sources: dict[str, Source],
    m_guard: tuple[Expr, Expr] | None = None,
    n_guard: tuple[Expr, Expr] | None = None,
    b_trans: bool = False,
    graph: Graph | None = None,
) -> tuple[Stmt, ...]:
    """The per-reduce ``ldmatrix a + ldmatrix b + mma.sync`` chain for one K-step."""
    a_src_index, a_ldm = _mma_src_index(a_load, smem_sources)
    b_src_index, b_ldm = _mma_src_index(b_load, smem_sources)
    # ldmatrix is smem→register only, so each operand's transport depends on
    # whether an enclosing StageBundle staged it. Staged → ldmatrix (with the
    # Source's TMA swizzle); unstaged → a gmem-direct fragment load (the staging
    # passes legitimately decline to stage, e.g. when the operand slabs don't fit
    # the smem budget — see ``tile/020_stage_inputs``; ``_mma_src_index`` returns
    # the gmem index + ``ldm=0`` for that case). The gmem path is slower (no smem
    # reuse) but lets a tensor-core variant compile instead of crashing, so the
    # search needn't avoid unstageable MMA tiles.
    a_staged = a_load.input in smem_sources
    b_staged = b_load.input in smem_sources
    # Masked-K (symbolic reduce): an unstaged operand whose reduce (K) gmem dim is
    # symbolic takes the gmem-direct ``*_kzero`` helper, which ZERO-FILLS the
    # partial final K tile past ``seq_len`` (a clamp would duplicate and corrupt
    # the accumulation). Mirrors the staged ``_stage_expand`` zero-fill for the
    # unstaged transport, so a masked-K mma lowers gmem-direct (slower than staged
    # — no smem reuse — but correct) instead of bailing. A transposed-B operand
    # keeps the clamp path (its K is the contiguous dim).
    a_kz = _masked_k_zero(a_load, a_src_index, "a", graph) if not a_staged else None
    b_kz = _masked_k_zero(b_load, b_src_index, "b", graph) if (not b_staged and not b_trans) else None
    if (not a_staged and a_kz is None and _unstaged_masked_k(a_load, "a", graph)) or (
        not b_staged and not b_trans and b_kz is None and _unstaged_masked_k(b_load, "b", graph)
    ):
        # The masked-K extent isn't a single symbol → can't build the ``*_kzero``
        # bound. Bail (LoweringError) so greedy retries / the search picks a staged
        # or scalar variant rather than emit an un-zero-filled gmem-direct read.
        from deplodock.compiler.pipeline.pipeline import LoweringError  # noqa: PLC0415

        raise LoweringError(
            "masked-K (symbolic reduce) mma operand can't lower gmem-direct — reduce extent "
            "is not a single symbol, so the *_kzero bound can't be built; needs staging"
        )
    # Masked cells with an UNSTAGED gated-axis operand take the clamped
    # gmem-direct helper: the staged slab is in-bounds by construction (its
    # gmem fill is clamped — 021 + _stage_expand), but a plain gmem-direct
    # fragment load at straddling-tile coords would read past the
    # runtime-sized buffer, so the lane coordinate clamps to the boundary
    # instead (``LdmatrixLoad.gmem_guard``). Keeps every enumerated variant
    # lowerable — the staging passes legitimately decline (smem budget), and
    # a greedy pick must not crash on the fallback.
    a_guard = m_guard if (m_guard is not None and not a_staged) else None
    b_guard = n_guard if (n_guard is not None and not b_staged) else None
    a_swz = smem_sources[a_load.input].swizzle.value if a_staged else "NONE"
    b_swz = smem_sources[b_load.input].swizzle.value if b_staged else "NONE"
    return (
        LdmatrixLoad(
            frag=a_frag,
            src_buffer=a_load.input,
            src_index=a_src_index,
            role="a",
            ldm=a_ldm,
            swizzle=a_swz,
            staged=a_staged,
            gmem_guard=a_guard,
            k_zero=a_kz,
        ),
        LdmatrixLoad(
            frag=b_frag,
            src_buffer=b_load.input,
            src_index=b_src_index,
            role="b",
            ldm=b_ldm,
            swizzle=b_swz,
            staged=b_staged,
            gmem_guard=b_guard,
            b_trans=b_trans,
            k_zero=b_kz,
        ),
        MmaSyncPtx(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, shape=spec.shape, ab_dtype=spec.operand_dtype("a").name),
    )


def _emit_store(
    spec: Atom,
    *,
    dst_buffer: str,
    dst_index: tuple,
    c_frag: str,
    epilogue: RegEpilogue | None = None,
    m_guard: tuple[Expr, Expr] | None = None,
    n_guard: tuple[Expr, Expr] | None = None,
) -> Stmt:
    """The accumulator → output store (with epilogue downconvert), carrying
    the fused pointwise chain when ``_scan_epilogue`` found one — the RegStore
    evaluates it per fragment element at the element's own coordinates — and
    the masked-tile boundary guards when ``_boundary_guards`` classified an
    enclosing boundary ``Cond``."""
    return RegStore(
        dst_buffer=dst_buffer, dst_index=dst_index, frag=c_frag, shape=spec.shape, epilogue=epilogue, m_guard=m_guard, n_guard=n_guard
    )


def _mma_src_index(load: Load, smem_sources: dict[str, Source]) -> tuple:
    """Choose the right ``src_index`` for an MMA fragment Load.

    Unstaged (load.input is the gmem buffer): use the gmem ``load.index``
    verbatim — pre-M5 behavior.

    Staged single-buffered (load.input is an smem name registered by an
    enclosing ``StageBundle`` with ``buffer_count == 1``): the consumer
    Load index is the bare cache-coord tuple. ``AffineAddressing.block``
    threads ``Var(cache_ax) * block`` per cache axis, relative to a zero
    origin.

    Staged double-buffered (``buffer_count >= 2``, M2 of
    plans/mma-perf-closures.md): the slab is allocated as ``[phase,
    *cache_axes]`` (rank-prepended); ``Load.index`` carries the leading
    phase Expr followed by the cache vars. Detect via
    ``len(load.index) > len(cache_axes)`` and splice the leading prefix
    in front of the computed cache coords so the MmaLoad reads from the
    right slot. ``ldm`` stays per-cache-axis (the phase dim is uniform
    across the slab — doesn't change the inner-source-dim row stride).
    """
    src = smem_sources.get(load.input)
    if src is None:
        # Unstaged: gmem-direct load. ``ldm=0`` triggers the render-time
        # ``ctx.shapes[gmem_buf][-1]`` lookup, which is the gmem tensor's
        # inner extent — correct for the rank-2 gmem operand.
        return load.index, 0
    if not isinstance(src.addressing, AffineAddressing):
        # Template-addressed Sources don't carry the block multiplier; the
        # cache vars in load.index decode verbatim through ``addressing.exprs``,
        # which the kernel renderer already handles via the standard Load path.
        return load.index, 0
    # The smem slab is rank == len(cache_axes); render_index expects an index
    # tuple of the SAME rank so its row-major flatten lines up with
    # ``Source.alloc_extents``. The per-cache-axis slab coord is
    # ``Var(ax) * block_ax`` (scalar paths have block=() → bare Var).
    cache_axes = src.cache_axes
    block = src.addressing.block
    dims = src.addressing.dims
    cache_coords: list = []
    for i, ax in enumerate(cache_axes):
        b = block[i] if block else 1
        if b == 1:
            cache_coords.append(Var(ax.name))
        else:
            cache_coords.append(Var(ax.name) * Literal(b, "int"))
    # M2 of plans/mma-perf-closures.md (Bug B): a buffered slab is allocated as
    # ``[phase, *cache_axes]``. ``020_stage_inputs`` / ``040_use_ring_buffers``
    # rewrites the consumer Load index to ``(phase_expr, *cache_vars)`` — phase
    # is the leading dim. Splice the leading prefix in front of the computed
    # cache-coord tuple so the MmaLoad reads from the right buffer slot.
    n_phase_dims = max(0, len(load.index) - len(cache_axes))
    phase_prefix = tuple(load.index[:n_phase_dims])
    out_index: tuple = phase_prefix + tuple(cache_coords)
    # ldm for row_major matrix_a / matrix_b is the row stride along the leading
    # source dim — equivalently, the product of slab dims for the *inner* source
    # dim. The auto-ldm path picks ``ctx.shapes[name][-1]`` which collapses to
    # the last alloc extent, wrong when several cache axes share a source dim
    # (e.g. an MMA matmul whose N-side splits into warp + register cells).
    # Compute explicitly: ldm = ∏ alloc_extents[i] for i where dims[i] is the
    # inner source dim. The phase prefix is uniform across the slab.
    alloc_extents = src.alloc_extents
    ldm_dim = max(dims) if dims else 0
    ldm = 1
    for i, d in enumerate(dims):
        if d == ldm_dim:
            ldm *= alloc_extents[i]
    return out_index, ldm
