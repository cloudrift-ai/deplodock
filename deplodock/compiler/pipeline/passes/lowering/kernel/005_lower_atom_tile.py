"""Lower AtomTile to MMA fragment Stmts — M5 of
``plans/mma-fragment-factorization.md`` plus M5 of
``plans/mma-smem-staging.md`` plus M2 of
``plans/mma-perf-closures.md``.

Runs *before* ``010_split_register_axes`` in the kernel chain. The
partition planner emits one of four AtomTile body shapes:

A. Direct K-loop, no staging (gmem-direct MMA):
   ``AtomTile > SerialTile(K_o) > SerialTile(K_i, reduce) > Load+Accum``
B. Single-bundle staged (SYNC or single-buffer ASYNC):
   ``AtomTile > SerialTile(K_o) > StageBundle > SerialTile(K_i, reduce) > Load+Accum``
C. Filtered K (single-iter, no SerialTile):
   ``AtomTile > Load+Assign+Accum`` (inline)
D. Pipelined + buffered (cp.async double-buffered, M2 of mma-perf-closures):
   ``AtomTile > StageBundle(prologue) > SerialTile(K_o-1) {
       StageBundle(issue next), AsyncWait, SerialTile(K_i, reduce), AsyncWait
   }, AsyncWait, SerialTile(K_i, reduce) (epilogue), Write``

For shapes A/B/D the rewrite is a **transform walk** that preserves every
structural Stmt (StageBundle wraps, AsyncWait, K_o SerialTile,
prologue/epilogue) and only rewrites:

- every ``SerialTile(is_reduce=True)`` body → ``MmaLoad a + MmaLoad b + MmaSync`` chain,
  with the ``is_reduce`` flag cleared (no more Accum inside).
- every ``Write`` → ``MmaStore``.

A pre-scan finds one ``(a_load, b_load)`` pair to seed the fragment SSA
names and the dtypes from the atom spec; the chain inside each per-reduce
``SerialTile`` re-classifies its own loads (shapes D has prologue/epilogue
reduces with the same K name, so the classification is stable across all
reduce sites).

For shape C the rewrite emits the Mma chain inline alongside an MmaStore.

The transform walk approach replaces the pre-2026 pattern-match-and-rebuild
path which captured exactly one ``(outer_st, reduce_st, enclosing_bundle)``
triple and rebuilt the AtomTile body from it — losing the prologue
StageBundle, the epilogue AsyncWait, and the epilogue reduce SerialTile
that shape D produces. The plan B/C bench gates need the pipelined path
working to measure the double-buffered cp.async lever.

**Phase-prefix prepend** (M2 Bug B of plans/mma-perf-closures.md). For
BUFFERED / ASYNC stages with ``buffer_count >= 2`` the slab is allocated
as ``[phase, …cache_axes…]`` (rank-prepended). The consumer Load gets
rewritten by 020_stage_inputs to ``Load(input='b_smem',
index=(phase_expr, cache_var_0, cache_var_1, …))`` — phase is the leading
dim. ``_mma_src_index`` preserves that leading prefix on the MmaLoad
``src_index`` by detecting ``len(load.index) > len(cache_axes)`` and
splicing the prefix in front of the cache-coord tuple. The ``ldm``
calculation stays per-cache-axis (the phase dim doesn't change the
inner-source-dim row stride).

**A/B classification for staged loads** (M2 Bug C). The pre-pipelined
heuristic keys off ``K_name in load.index[-1]`` → A vs ``in load.index[0]``
→ B. For staged smem loads the index is multi-dim slab coords (e.g.
``(phase, a2, a4, a6)``) where the K axis sits in the *middle*. When
``load.input`` resolves to a staged smem name, classify A vs B by reading
``Source.cache_dims`` — the cache axis whose ``axis.name == K_name`` has
``source_dim == 1`` for A (K inner) or ``0`` for B (K outer).

Eligibility: ``op.knobs["ATOM_KIND"]`` set (only warp-tier matmul rows
carry this knob — the scalar planner branch leaves it unset and this
pass skips). Idempotence: after this pass the AtomTile is gone, so on
a second visit the pattern doesn't match and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.dtype import DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    LdmatrixLoad,
    MmaFill,
    MmaFragment,
    MmaLoad,
    MmaStore,
    MmaSync,
    MmaSyncPtx,
    RegFragment,
    RegStore,
)
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    AtomTile,
    SerialTile,
    Source,
    StageBundle,
    TileOp,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._atom import AtomSpec, atom_spec
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    parallel_tile_of,
    replace_parallel_tile_body,
    single_tile,
)

PATTERN = [Pattern("root", TileOp)]


# --- Per-instruction leaf emitters -----------------------------------------
#
# The transform-walk skeleton (``_atom_body_to_mma`` / ``_transform_walk`` /
# ``_build_mma_chain``) is shared across instruction families; only these leaf
# factories differ. ``spec.instruction == "wmma"`` → opaque ``nvcuda::wmma``
# nodes; ``"mma_sync"`` → the ``ldmatrix`` + ``mma.sync.aligned`` register-array
# nodes (the s16816 path). The mma.sync path has no gmem-direct load — its
# operands MUST be staged smem (``_emit_chain`` RuleSkips an unstaged leaf).


def _emit_fragments(spec: AtomSpec, *, c_frag: str, a_frag: str, b_frag: str, c_dtype: DataType) -> tuple[Stmt, ...]:
    """Fragment / register-array declarations + accumulator zero-init.

    WMMA returns three ``MmaFragment`` decls plus a trailing ``MmaFill``;
    mma.sync returns three ``RegFragment`` decls (the ``c`` array is
    zero-initialised at declaration, so no separate fill)."""
    if spec.instruction == "mma_sync":
        return (
            RegFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
            RegFragment(name=a_frag, role="a", shape=spec.shape, dtype=spec.operand_dtypes["a"]),
            RegFragment(name=b_frag, role="b", shape=spec.shape, dtype=spec.operand_dtypes["b"]),
        )
    return (
        MmaFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
        MmaFragment(name=a_frag, role="a", shape=spec.shape, dtype=spec.operand_dtypes["a"]),
        MmaFragment(name=b_frag, role="b", shape=spec.shape, dtype=spec.operand_dtypes["b"]),
        MmaFill(frag=c_frag, value=0.0),
    )


def _emit_chain(
    spec: AtomSpec,
    *,
    a_load: Load,
    b_load: Load,
    a_frag: str,
    b_frag: str,
    c_frag: str,
    smem_sources: dict[str, Source],
) -> tuple[Stmt, ...]:
    """The per-reduce ``load a + load b + mma`` chain for one K-step."""
    a_src_index, a_ldm = _mma_src_index(a_load, smem_sources)
    b_src_index, b_ldm = _mma_src_index(b_load, smem_sources)
    if spec.instruction == "mma_sync":
        # ldmatrix is smem→register only — both operands must be staged.
        # The WMMA sibling row covers the gmem-direct shape, so RuleSkipping
        # here just drops the mma.sync variant, never the whole lowering.
        if a_load.input not in smem_sources or b_load.input not in smem_sources:
            raise RuleSkipped("mma.sync requires staged smem operands (ldmatrix has no gmem-direct path)")
        return (
            LdmatrixLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_src_index, role="a", ldm=a_ldm),
            LdmatrixLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_src_index, role="b", ldm=b_ldm),
            MmaSyncPtx(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, shape=spec.shape),
        )
    return (
        MmaLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_src_index, ldm=a_ldm),
        MmaLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_src_index, ldm=b_ldm),
        MmaSync(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag),
    )


def _emit_store(spec: AtomSpec, *, dst_buffer: str, dst_index: tuple, c_frag: str) -> Stmt:
    """The accumulator → output store (with epilogue downconvert)."""
    if spec.instruction == "mma_sync":
        return RegStore(dst_buffer=dst_buffer, dst_index=dst_index, frag=c_frag, shape=spec.shape)
    return MmaStore(dst_buffer=dst_buffer, dst_index=dst_index, frag=c_frag, ldm=0, shape=spec.shape)


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    atom_kind = op.knobs.get("ATOM_KIND")
    if not atom_kind:
        raise RuleSkipped("not an MMA TileOp (no ATOM_KIND knob)")
    spec = atom_spec(atom_kind)
    body = op.body
    idx, outer = single_tile(body)
    tt = parallel_tile_of(outer)

    # Accumulate in fp32 (the spec's c dtype) regardless of output buffer
    # dtype — matches cuBLAS / PyTorch fp16-GEMM precision. When the output
    # buffer is narrower (``__half*`` / ``__nv_bfloat16*``), ``MmaStore``
    # emits an epilogue downconvert so ``store_matrix_sync``'s element-type
    # match still holds. See ``_resolve_c_dtype`` + ``MmaStore`` docstrings.
    c_dtype_override = _resolve_c_dtype(root, spec.operand_dtypes["c"])

    lowered, found = _lower_atom_tiles(tt.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources={})
    if not found:
        # Could happen on a second visit (AtomTile already consumed).
        raise RuleSkipped("no AtomTile in body — already lowered")

    rebuilt = replace_parallel_tile_body(outer, lowered)
    return TileOp(body=body[:idx] + (rebuilt,) + body[idx + 1 :], name=op.name, knobs=op.knobs)


def _resolve_c_dtype(root: Node, spec_c_dtype: DataType) -> DataType:  # noqa: ARG001
    """Pick the c-fragment dtype for the WMMA accumulator.

    **Always accumulate in fp32** (the spec's c dtype — F32 for every
    registered WMMA kind). This matches cuBLAS / PyTorch fp16-GEMM
    precision: fp16 accumulation drifts ~3-4 ulp per step, which over a
    long K reduction (2048+) compounds into multi-percent error. The
    previous F16-output → F16-accumulator path was ~2× faster on GeForce
    (where the fp32-accumulate tensor pipe is rate-limited) but only
    passed the loose fp16 accuracy gate, not a real precision bar.

    WMMA's ``store_matrix_sync`` requires the destination pointer's element
    type to match the fragment's element type, so an fp32 accumulator
    can't store straight into a ``__half*`` buffer. :class:`MmaStore`
    handles that with an epilogue downconvert (declare a dst-dtype
    fragment, element-wise ``__float2half`` / ``__float2bfloat16``, then
    store) — see its docstring. ``root`` is no longer read (the dtype no
    longer tracks the output buffer) but kept for call-site stability.
    """
    return spec_c_dtype


def _lower_atom_tiles(
    body: Body,
    *,
    spec: AtomSpec,
    c_dtype_override: DataType,
    smem_sources: dict[str, Source],
) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile`` encountered, rewrite its
    interior matmul-shape body into an Mma* fragment chain and strip
    the AtomTile wrapper. Recurses into non-AtomTile block stmts so a
    deep-nested AtomTile (under RegisterTile / SerialTile / Cond / ...)
    is reached. Returns ``(new_body, found_any)``.

    ``smem_sources`` is a flat ``smem_name → Source`` map of Sources
    in-scope from enclosing ``StageBundle``s. When ``_atom_body_to_mma``
    encounters a Load reading from a staged smem buffer, it rebuilds
    ``src_index`` via the Source's ``AffineAddressing.source_index`` so
    the MMA fragment lands on the correct (block-scaled) slab offset.
    M2-M5 of ``plans/mma-smem-staging.md``.
    """
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            new_stmts = _atom_body_to_mma(s.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources=smem_sources)
            out.extend(new_stmts)
            found = True
            continue
        if isinstance(s, StageBundle):
            # Register every Source in the bundle so child AtomTile
            # Loads reading from this slab can resolve the addressing.
            # Bundles can nest (cooperative-K + outer staging); we copy
            # the dict so a child bundle's additions don't leak out to
            # sibling subtrees.
            inner_sources = dict(smem_sources)
            for stage in s.stages:
                for src in stage.sources:
                    inner_sources[src.name] = src
            new_bundle_body, body_found = _lower_atom_tiles(
                s.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources=inner_sources
            )
            # ``StageBundle.with_bodies`` expects ``(stages_body, body)`` —
            # stages stay byte-clean (no AtomTile lives inside producer
            # Sources), we only rebuild the consumer body.
            new_stages_body = Body(s.stages)
            out.append(s.with_bodies((new_stages_body, new_bundle_body)))
            found = found or body_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_lowered = False
            for sub in s.nested():
                new_sub, sub_found = _lower_atom_tiles(sub, spec=spec, c_dtype_override=c_dtype_override, smem_sources=smem_sources)
                new_bodies.append(new_sub)
                any_lowered = any_lowered or sub_found
            found = found or any_lowered
            out.append(s.with_bodies(tuple(new_bodies)))
            continue
        out.append(s)
    return Body(out), found


def _atom_body_to_mma(
    body: Body,
    *,
    spec: AtomSpec,
    c_dtype_override: DataType,
    smem_sources: dict[str, Source],
) -> tuple[Stmt, ...]:
    """Rewrite the AtomTile body to an Mma* fragment chain.

    Two cases:

    - **Shape A/B/D** (has at least one ``SerialTile(is_reduce=True)``):
      transform-walk the body in place. Each reduce SerialTile's body is
      replaced with the Mma chain; the ``is_reduce`` flag is cleared
      (no more Accum inside). The Write is replaced with MmaStore.
      Every other structural Stmt (StageBundle wraps, AsyncWait, K_o
      SerialTile, prologue/epilogue) flows through unchanged.

    - **Shape C** (no reduce SerialTile — single-iter K filtered away):
      emit the Mma chain inline at the AtomTile body level alongside an
      MmaStore.

    Both cases prepend ``MmaFragment(c/a/b)`` decls + ``MmaFill(c_frag, 0)``
    at the head of the returned tuple.
    """
    # Local copy so a bundle nested inside the AtomTile's body contributes
    # its Sources to the lookup ``_mma_src_index`` consults below — without
    # leaking back to siblings.
    bundle_sources = dict(smem_sources)
    for stage_bundle in _iter_bundles(body):
        for stage in stage_bundle.stages:
            for src in stage.sources:
                bundle_sources[src.name] = src

    # Pre-scan: find one Write, one Accum, and one (a_load, b_load) sample
    # pair to seed fragment SSA names + dtypes from the atom spec.
    write_stmt, accum, sample_a_load, sample_b_load, has_reduce = _scan_atom_body(body, bundle_sources)

    if write_stmt is None:
        raise RuleSkipped("AtomTile body unrecognised — no Write")
    if accum is None or sample_a_load is None or sample_b_load is None:
        raise RuleSkipped(
            f"AtomTile body unrecognised — expected Load+Load+Accum chain (got accum={accum!r}, a={sample_a_load!r}, b={sample_b_load!r})"
        )

    c_frag = f"{accum.name}_frag"
    a_frag = f"{sample_a_load.names[0]}_frag"
    b_frag = f"{sample_b_load.names[0]}_frag"

    # Fragment / register-array decls (+ fill for WMMA; mma.sync zero-inits
    # its ``c`` array in the decl, so ``_emit_fragments`` carries the right head).
    fragments = _emit_fragments(spec, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, c_dtype=c_dtype_override)

    if has_reduce:
        transformed = _transform_walk(
            body,
            c_frag=c_frag,
            a_frag=a_frag,
            b_frag=b_frag,
            smem_sources=bundle_sources,
            spec=spec,
        )
        return (*fragments, *transformed)

    # Shape C: K filtered, inline Mma chain + store.
    chain = _emit_chain(
        spec, a_load=sample_a_load, b_load=sample_b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=bundle_sources
    )
    store = _emit_store(spec, dst_buffer=write_stmt.output, dst_index=write_stmt.index, c_frag=c_frag)
    return (*fragments, *chain, store)


def _iter_bundles(body: Body):
    """Yield every ``StageBundle`` reachable inside ``body``. Used by the
    pre-scan to flatten all in-scope Sources before A/B classification
    walks per-reduce sites — the pipelined-staged shape (D) has bundles
    at the AtomTile-body level *and* inside the K_o SerialTile, both
    feeding loads inside reduce SerialTiles."""
    for s in body:
        if isinstance(s, StageBundle):
            yield s
            yield from _iter_bundles(s.body)
            continue
        if s.nested():
            for sub in s.nested():
                yield from _iter_bundles(sub)


def _scan_atom_body(body: Body, smem_sources: dict[str, Source]) -> tuple[Write | None, Accum | None, Load | None, Load | None, bool]:
    """Recursively walk the AtomTile body. Returns the first ``Write``,
    the first ``Accum``, one sample ``(a_load, b_load)`` pair, and
    ``has_reduce`` (True if any reduce SerialTile was found).

    The sample (a, b) pair is pulled from the FIRST reduce SerialTile if
    present, falling back to the bare body's Loads (shape C). The
    classification heuristic for the sample pair is the same one
    ``_build_mma_chain`` re-uses for every reduce site — pulling it out
    of one fixed location keeps the fragment SSA names stable across
    prologue / inner / epilogue chains (which is what the per-cell
    replicator in 010_split_register_axes expects).
    """
    write_stmt: Write | None = None
    accum: Accum | None = None
    sample_a: Load | None = None
    sample_b: Load | None = None
    has_reduce = False
    # Fallback loads at the bare body level (shape C — no reduce ST).
    flat_loads: list[Load] = []

    def _walk(stmts: Body) -> None:
        nonlocal write_stmt, accum, sample_a, sample_b, has_reduce
        for s in stmts:
            if isinstance(s, Write):
                if write_stmt is None:
                    write_stmt = s
                continue
            if isinstance(s, Accum):
                if accum is None:
                    accum = s
                continue
            if isinstance(s, Load):
                flat_loads.append(s)
                continue
            if isinstance(s, SerialTile) and s.is_reduce:
                has_reduce = True
                # First reduce ST seeds the sample (a, b).
                if sample_a is None or sample_b is None:
                    loads = [c for c in s.body if isinstance(c, Load)]
                    if len(loads) == 2:
                        a, b = _classify_ab(loads, k_name=s.axis.name, smem_sources=smem_sources, write=write_stmt)
                        if a is not None and b is not None:
                            sample_a, sample_b = a, b
                _walk(s.body)
                continue
            if s.nested():
                for sub in s.nested():
                    _walk(sub)

    _walk(body)

    # Shape C fallback: no reduce ST, fall back to flat Load+Accum at the
    # bare body level. Classify by Write index (M in outer dim → A,
    # N in inner dim → B), matching the pre-2026 heuristic.
    if not has_reduce and sample_a is None and sample_b is None and len(flat_loads) == 2 and write_stmt is not None:
        a, b = _classify_ab(flat_loads, k_name=None, smem_sources=smem_sources, write=write_stmt)
        if a is not None and b is not None:
            sample_a, sample_b = a, b

    return write_stmt, accum, sample_a, sample_b, has_reduce


def _classify_ab(
    loads: list[Load],
    *,
    k_name: str | None,
    smem_sources: dict[str, Source],
    write: Write | None,
) -> tuple[Load | None, Load | None]:
    """Identify which Load is A (M×K) and which is B (K×N) inside a
    matmul-reduce cell body.

    Two cases:

    1. **Staged smem load** (``load.input`` is in ``smem_sources``):
       look up the Source's ``cache_dims``; find the cache axis whose
       ``axis.name == k_name``. Its ``source_dim`` is 1 (K inner) for A,
       0 (K outer) for B. The pre-2026 heuristic ("K_name in
       ``load.index[-1]``") fails here because the staged Load index is
       ``(phase, a8, a3, a5)`` where K sits in the middle dim.

    2. **gmem-direct load** (no smem source): legacy heuristic — K in
       ``index[-1]`` → A; K in ``index[0]`` → B. For shape C
       (``k_name is None``) heuristic falls back to the Write's M/N free
       vars.
    """
    a_load: Load | None = None
    b_load: Load | None = None

    if k_name is not None:
        for ld in loads:
            src = smem_sources.get(ld.input)
            if src is not None:
                # Staged: classify via cache_dims.
                for cd in src.cache_dims:
                    if cd.axis.name == k_name:
                        if cd.source_dim == 1:
                            a_load = ld
                        elif cd.source_dim == 0:
                            b_load = ld
                        break
                continue
            # gmem-direct: legacy K-position heuristic.
            k_in_last = k_name in {v for e in ld.index[-1:] for v in e.free_vars()}
            k_in_first = k_name in {v for e in ld.index[:1] for v in e.free_vars()}
            if k_in_last and not k_in_first:
                a_load = ld
            elif k_in_first and not k_in_last:
                b_load = ld
        return a_load, b_load

    # Shape C — K filtered. Use Write's M/N free vars as the carrier.
    if write is None:
        return None, None
    w_m_vars = set(write.index[0].free_vars()) if write.index else set()
    w_n_vars = set(write.index[-1].free_vars()) if write.index else set()
    for ld in loads:
        ld_outer_vars = set(ld.index[0].free_vars()) if ld.index else set()
        ld_inner_vars = set(ld.index[-1].free_vars()) if ld.index else set()
        if w_m_vars & ld_outer_vars and not (w_n_vars & ld_inner_vars):
            a_load = ld
        elif w_n_vars & ld_inner_vars and not (w_m_vars & ld_outer_vars):
            b_load = ld
    return a_load, b_load


def _transform_walk(
    body: Body,
    *,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    smem_sources: dict[str, Source],
    spec: AtomSpec,
) -> tuple[Stmt, ...]:
    """Recursively rewrite ``body`` in place. Replaces:

    - every ``SerialTile(is_reduce=True)`` body with an Mma chain
      (clearing the ``is_reduce`` flag — no Accum remains inside).
    - every ``Write`` with ``MmaStore``.

    Preserves every other Stmt structurally — StageBundle wraps + their
    Stages, AsyncWait, non-reduce SerialTile (K_o outer), Cond, etc.
    Descends into nested bodies via ``s.nested()`` so deeply-nested
    reduce sites (e.g. shape D's inner-K_o body) are reached.

    ``smem_sources`` is threaded through StageBundle descent so the Mma
    chain's A/B classification + phase-prefix prepend has the full
    in-scope Source table.
    """
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Write):
            out.append(_emit_store(spec, dst_buffer=s.output, dst_index=s.index, c_frag=c_frag))
            continue
        if isinstance(s, SerialTile) and s.is_reduce:
            chain = _build_mma_chain(
                s.body,
                k_name=s.axis.name,
                c_frag=c_frag,
                a_frag=a_frag,
                b_frag=b_frag,
                smem_sources=smem_sources,
                spec=spec,
            )
            # ``SerialTile.is_reduce`` is a derived property (True iff the
            # body contains an Accum). The Mma chain has no Accum — c_frag
            # owns the accumulation through MmaSync — so swapping the body
            # via ``with_bodies`` implicitly flips is_reduce off. Kind is
            # preserved (stage_inner / serial_outer / …).
            out.append(s.with_bodies((Body(chain),)))
            continue
        if isinstance(s, StageBundle):
            inner_sources = dict(smem_sources)
            for stage in s.stages:
                for src in stage.sources:
                    inner_sources[src.name] = src
            new_body = _transform_walk(s.body, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, smem_sources=inner_sources, spec=spec)
            # Stages stay byte-clean — no Mma rewrite happens inside a
            # producer scope.
            out.append(s.with_bodies((Body(s.stages), Body(new_body))))
            continue
        if s.nested():
            new_bodies = tuple(
                Body(_transform_walk(sub, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, smem_sources=smem_sources, spec=spec))
                for sub in s.nested()
            )
            out.append(s.with_bodies(new_bodies))
            continue
        # AsyncWait + anything else without nested bodies: keep as-is.
        out.append(s)
    return tuple(out)


def _build_mma_chain(
    reduce_body: Body,
    *,
    k_name: str,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    smem_sources: dict[str, Source],
    spec: AtomSpec,
) -> tuple[Stmt, ...]:
    """Build the per-reduce ``load a + load b + mma`` chain that replaces a
    reduce SerialTile's body. Re-classifies A/B from this reduce site's
    Loads — shape D has prologue/inner/epilogue reduces with the same K
    name and the classification is stable. The instruction-specific node
    choice (WMMA vs mma.sync) is delegated to :func:`_emit_chain`.
    """
    loads = [s for s in reduce_body if isinstance(s, Load)]
    if len(loads) != 2:
        raise RuleSkipped(f"reduce SerialTile body unrecognised — expected 2 Loads, got {len(loads)}")
    a_load, b_load = _classify_ab(loads, k_name=k_name, smem_sources=smem_sources, write=None)
    if a_load is None or b_load is None:
        raise RuleSkipped("reduce SerialTile Loads didn't match A=[M,K], B=[K,N] shape")
    return _emit_chain(spec, a_load=a_load, b_load=b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=smem_sources)


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
        # Unstaged: gmem-direct WMMA load. ``ldm=0`` triggers the
        # render-time ``ctx.shapes[gmem_buf][-1]`` lookup, which is the
        # gmem tensor's inner extent — correct for the rank-2 gmem
        # operand.
        return load.index, 0
    if not isinstance(src.addressing, AffineAddressing):
        # Template-addressed Sources don't carry the block multiplier;
        # the cache vars in load.index decode verbatim through
        # ``addressing.exprs``, which the kernel renderer already
        # handles via the standard Load path. Fall back to the gmem-
        # style passthrough — same as the pre-M5 defensive branch.
        return load.index, 0
    # The smem slab is rank == len(cache_axes); render_index expects an
    # index tuple of the SAME rank so its row-major flatten lines up
    # with ``Source.alloc_extents``. The per-cache-axis slab coord is
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
    # M2 of plans/mma-perf-closures.md (Bug B): a buffered slab is
    # allocated as ``[phase, *cache_axes]``. ``020_stage_inputs`` /
    # ``040_use_ring_buffers`` rewrites the consumer Load index to
    # ``(phase_expr, *cache_vars)`` — phase is the leading dim. Splice
    # the leading prefix (zero or one dim today; the design admits more)
    # in front of the computed cache-coord tuple so the MmaLoad reads
    # from the right buffer slot.
    n_phase_dims = max(0, len(load.index) - len(cache_axes))
    phase_prefix = tuple(load.index[:n_phase_dims])
    out_index: tuple = phase_prefix + tuple(cache_coords)
    # ldm for WMMA row_major matrix_a / matrix_b is the row stride along
    # the leading source dim — equivalently, the product of slab dims
    # for the *inner* source dim (dim 1 here). The auto-ldm path picks
    # ``ctx.shapes[name][-1]`` which collapses to the last alloc extent,
    # wrong when several cache axes share a source dim (e.g. an MMA
    # matmul whose N-side splits into warp + register cells). Compute
    # explicitly: ldm = ∏ alloc_extents[i] for i where dims[i] is the
    # inner source dim. The phase prefix is uniform across the slab —
    # doesn't change the row stride.
    alloc_extents = src.alloc_extents
    ldm_dim = max(dims) if dims else 0
    ldm = 1
    for i, d in enumerate(dims):
        if d == ldm_dim:
            ldm *= alloc_extents[i]
    return out_index, ldm
