"""Lower the tensor-core matmul cell to the kernel-IR MMA fragment chain.

The matmul cell arrives in tensor-core form: ``tile/011_lower_atom_cell`` fused
the compute into an :class:`~deplodock.compiler.ir.stmt.Mma` (which names its A
/ B operands by SSA value and carries the ``Atom`` spec) and left the operand
``Load``s **plain**, and the staging passes carried both through (the loads
staged like any other, the ``Mma`` keeping its reduce loop ``is_reduce``). The
partition planner emits one of four cell shapes:

A. Direct K-loop, no staging (pruned for mma.sync — ldmatrix is smem-only):
   ``AtomTile > SerialTile(K_o) > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
B. Single-bundle staged (SYNC or single-buffer ASYNC):
   ``AtomTile > SerialTile(K_o) > StageBundle > SerialTile(K_i, reduce) > [Load a*, Load b*, Mma]``
C. Filtered K (single-iter, no SerialTile): ``AtomTile > [Load a*, Load b*, Mma]`` (inline)
D. Pipelined + buffered (cp.async double-buffered): prologue StageBundle, K_o-1
   loop with issue-next StageBundle + AsyncWait + reduce, AsyncWait, epilogue reduce, Write.

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
tensor-core family; it has no gmem-direct load (ldmatrix is smem→register
only), so :func:`_emit_chain` raises an actionable ``LoweringError`` on an
unstaged operand. By the time this pass runs the atom-vs-scalar fork is already
decided (back at ``tile/010_partition_loops``), so there is no scalar sibling
left to fall back to — a silent skip would leak the unconsumed ``AtomTile`` to
render and crash with an opaque ``NotImplementedError``; the error names the
``DEPLODOCK_MMA=0`` (scalar) / ``TMA=1`` (stage the atom path) remedies instead.

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
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore
from deplodock.compiler.ir.stmt import Body, Load, Mma, Stmt, Write
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
from deplodock.compiler.pipeline import LoweringError, Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    # No knob lookup: each cell's atom spec is read off its ``Mma`` (the
    # ``AtomTile`` presence is the signal). Scalar / already-lowered TileOps
    # have no ``AtomTile`` → found is False → skip.
    lowered, found = lower_atom_cells(op.body, smem_sources={})
    if not found:
        raise RuleSkipped("no AtomTile in body (scalar, or already lowered)")
    return TileOp(body=lowered, name=op.name, knobs=op.knobs)


def lower_atom_cells(body: Body, *, smem_sources: dict[str, Source]) -> tuple[Body, bool]:
    """For each ``AtomTile`` in ``body``, lower its matmul cell to the kernel-IR
    fragment chain and strip the wrapper. Runs through the shared
    :func:`~deplodock.compiler.ir.tile.ir.map_staged` traversal, which threads
    the in-scope ``Source`` table from enclosing ``StageBundle`` /
    ``WarpSpecialize`` scopes so each operand's slab addressing resolves from the
    live ``Source``. The atom spec is read off each cell's ``Mma`` (no knob
    lookup). Returns ``(new_body, found_any)``."""
    found = False

    def handler(s: Stmt, sources: dict[str, Source]) -> tuple[Stmt, ...] | None:
        nonlocal found
        if isinstance(s, AtomTile):
            found = True
            return _lower_cell(s.body, smem_sources=sources)
        return None

    return map_staged(body, handler, sources=smem_sources), found


def _lower_cell(atom_body: Body, *, smem_sources: dict[str, Source]) -> tuple[Stmt, ...]:
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
            for stage in s.stages:
                for src in stage.sources:
                    bundle_sources[src.name] = src
        return None

    map_staged(atom_body, _gather)

    write_stmt, a_seed, b_seed, c_seed, has_reduce, spec = _scan_cell(atom_body)
    if write_stmt is None:
        raise RuleSkipped("AtomTile body unrecognised — no Write")
    if spec is None or a_seed is None or b_seed is None or c_seed is None:
        raise RuleSkipped(f"AtomTile body unrecognised — expected operand Loads + Mma (got a={a_seed!r}, b={b_seed!r}, c={c_seed!r})")

    c_frag, a_frag, b_frag = f"{c_seed}_frag", f"{a_seed}_frag", f"{b_seed}_frag"
    fragments = _emit_fragments(spec, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, c_dtype=spec.operand_dtype("c"))

    if has_reduce:

        def handler(s: Stmt, sources: dict[str, Source]) -> tuple[Stmt, ...] | None:
            if isinstance(s, Write):
                return (_emit_store(spec, dst_buffer=s.output, dst_index=s.index, c_frag=c_frag),)
            if isinstance(s, SerialTile) and s.is_reduce:
                chain = _build_chain(s.body, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, smem_sources=sources, spec=spec)
                return (s.with_bodies((Body(chain),)),)
            return None

        transformed = map_staged(atom_body, handler, sources=bundle_sources)
        return (*fragments, *transformed)

    # Shape C: K filtered — inline chain from the body's operand Loads + store.
    a_load, b_load = _find_role_loads(atom_body)
    if a_load is None or b_load is None:
        raise RuleSkipped("Atom body (shape C) missing its Mma / A/B loads")
    chain = _emit_chain(spec, a_load=a_load, b_load=b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=bundle_sources)
    store = _emit_store(spec, dst_buffer=write_stmt.output, dst_index=write_stmt.index, c_frag=c_frag)
    return (*fragments, *chain, store)


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


def _find_role_loads(body: Body) -> tuple[Load | None, Load | None]:
    """The A / B operand Loads of the cell in ``body`` — identified via the
    co-located ``Mma``, which names its operands by SSA value (so the operand
    Loads need no tensor-core tag of their own)."""
    mma = next((s for s in body if isinstance(s, Mma)), None)
    if mma is None:
        return None, None
    by_name = {ld.names[0]: ld for ld in body if isinstance(ld, Load) and ld.names}
    return by_name.get(mma.a), by_name.get(mma.b)


def _build_chain(
    reduce_body: Body,
    *,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    smem_sources: dict[str, Source],
    spec: Atom,
) -> tuple[Stmt, ...]:
    """Build the ``ldmatrix a + ldmatrix b + mma.sync`` chain that replaces a
    reduce SerialTile's ``[Load a, Load b, Mma]`` body — operands matched via
    the body's ``Mma`` (which names its A/B operands by SSA value)."""
    a_load, b_load = _find_role_loads(reduce_body)
    if a_load is None or b_load is None:
        raise RuleSkipped("reduce SerialTile body missing its Mma / A/B Loads")
    return _emit_chain(spec, a_load=a_load, b_load=b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=smem_sources)


# --- Per-instruction leaf emitters -----------------------------------------


def _emit_fragments(spec: Atom, *, c_frag: str, a_frag: str, b_frag: str, c_dtype: DataType) -> tuple[Stmt, ...]:
    """Register-array declarations. Three ``RegFragment`` decls; the ``c``
    array is zero-initialised at declaration, so there's no separate fill."""
    return (
        RegFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
        RegFragment(name=a_frag, role="a", shape=spec.shape, dtype=spec.operand_dtype("a")),
        RegFragment(name=b_frag, role="b", shape=spec.shape, dtype=spec.operand_dtype("b")),
    )


def _emit_chain(
    spec: Atom,
    *,
    a_load: Load,
    b_load: Load,
    a_frag: str,
    b_frag: str,
    c_frag: str,
    smem_sources: dict[str, Source],
) -> tuple[Stmt, ...]:
    """The per-reduce ``ldmatrix a + ldmatrix b + mma.sync`` chain for one K-step."""
    a_src_index, a_ldm = _mma_src_index(a_load, smem_sources)
    b_src_index, b_ldm = _mma_src_index(b_load, smem_sources)
    # ldmatrix is smem→register only — both operands must be staged. We reach
    # here only once the atom (warp-tier) variant has been committed (the
    # atom-vs-scalar fork is decided back at the tile planner), so there is no
    # scalar sibling left to fall back to: a RuleSkip would silently leak the
    # unconsumed AtomTile to render and crash with an opaque NotImplementedError.
    # Raise an actionable error naming the remedies instead (mirrors the
    # "TMA cannot fire" guard in tile/050_use_tma).
    if a_load.input not in smem_sources or b_load.input not in smem_sources:
        raise LoweringError(
            "tensor-core (mma.sync) matmul selected but its operands are not staged in shared "
            "memory for ldmatrix (ldmatrix has no gmem-direct path) — typically the atom path was "
            "chosen without staging in scope (e.g. TMA=0). Force the scalar FMA path with "
            "DEPLODOCK_MMA=0, or enable staging for the atom path with TMA=1."
        )
    # Thread the per-Source TMA swizzle mode so the ldmatrix consumer applies
    # the matching per-lane chunk XOR. NONE when the source wasn't swizzled.
    a_swz = smem_sources[a_load.input].swizzle.value
    b_swz = smem_sources[b_load.input].swizzle.value
    return (
        LdmatrixLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_src_index, role="a", ldm=a_ldm, swizzle=a_swz),
        LdmatrixLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_src_index, role="b", ldm=b_ldm, swizzle=b_swz),
        MmaSyncPtx(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, shape=spec.shape, ab_dtype=spec.operand_dtype("a").name),
    )


def _emit_store(spec: Atom, *, dst_buffer: str, dst_index: tuple, c_frag: str) -> Stmt:
    """The accumulator → output store (with epilogue downconvert)."""
    return RegStore(dst_buffer=dst_buffer, dst_index=dst_index, frag=c_frag, shape=spec.shape)


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
