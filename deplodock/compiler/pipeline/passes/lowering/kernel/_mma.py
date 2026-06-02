"""MMA-fragment helpers for the ``Atom`` lowering passes.

Two halves, around the Tile-IR :class:`~deplodock.compiler.ir.tile.ir.Atom`
node:

- **Recognition** (``kernel/005_lower_atom_tile``): :func:`lower_to_atoms`
  walks a TileOp body, finds each ``AtomTile``, and :func:`build_atom`
  classifies its A/B operands + accumulator + store target (via
  :func:`_scan_atom_body` / :func:`_classify_ab` / :func:`_iter_bundles`) and
  packages the whole per-cell computation into one ``Atom``.
- **Expansion** (``kernel/006_expand_atom``): :func:`expand_atoms` walks a
  TileOp body, finds each ``Atom``, and :func:`expand_atom` lowers it to the
  kernel-IR ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore``
  chain (the ``_emit_*`` leaf emitters + :func:`_transform_walk` /
  :func:`_build_mma_chain` / :func:`_mma_src_index`) — the shape the downstream
  passes (``010_split_register_axes`` …) consume.

The s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` path is the sole
tensor-core family (``instruction="mma_sync"``). It has no gmem-direct load
(ldmatrix is smem→register only), so :func:`build_atom` RuleSkips an unstaged
operand — pruning the warp-tier variant so the scalar tier covers that shape.
See the ``005_lower_atom_tile`` module docstring for the four AtomTile body
shapes (A/B/C/D), the phase-prefix prepend, and the A/B classification rules.
"""

from __future__ import annotations

from deplodock.compiler.dtype import DataType
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    LdmatrixLoad,
    MmaSyncPtx,
    RegFragment,
    RegStore,
)
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Atom,
    AtomTile,
    SerialTile,
    Source,
    StageBundle,
    WarpSpecialize,
)
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._atom import AtomSpec, atom_spec

# ===========================================================================
# Recognition: AtomTile -> Atom  (kernel/005_lower_atom_tile)
# ===========================================================================


def resolve_c_dtype(root: Node, spec_c_dtype: DataType) -> DataType:  # noqa: ARG001
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


def lower_to_atoms(
    body: Body,
    *,
    spec_kind: str,
    c_dtype: DataType,
    smem_sources: dict[str, Source],
) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile`` encountered, package its
    interior matmul-shape body into an :class:`Atom` node and strip the
    AtomTile wrapper. Recurses into non-AtomTile block stmts so a
    deep-nested AtomTile (under RegisterTile / SerialTile / Cond / ...)
    is reached. Returns ``(new_body, found_any)``.

    ``smem_sources`` is a flat ``smem_name → Source`` map of Sources
    in-scope from enclosing ``StageBundle``s. :func:`build_atom` uses it to
    classify the A/B operands and to snapshot their ``Source``s onto the
    ``Atom`` (so the expander resolves slab addressing without re-threading
    enclosing scope). M2-M5 of ``plans/mma-smem-staging.md``.
    """
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            out.append(build_atom(s.body, spec_kind=spec_kind, c_dtype=c_dtype, smem_sources=smem_sources))
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
            new_bundle_body, body_found = lower_to_atoms(s.body, spec_kind=spec_kind, c_dtype=c_dtype, smem_sources=inner_sources)
            # ``StageBundle.with_bodies`` expects ``(stages_body, body)`` —
            # stages stay byte-clean (no AtomTile lives inside producer
            # Sources), we only rebuild the consumer body.
            out.append(s.with_bodies((Body(s.stages), new_bundle_body)))
            found = found or body_found
            continue
        if isinstance(s, WarpSpecialize):
            # Warp-tier WS hoisted the staging ``StageBundle`` into the
            # producer body, away from the consumer ``AtomTile`` that reads
            # its slab. Harvest the producer's Sources into scope so the
            # consumer Loads resolve their smem addressing (the producer body
            # holds only TMA-issue scaffolding — no AtomTile). The producer
            # body is left untouched.
            ws_sources = dict(smem_sources)
            for st in s.producer_body.iter():
                if isinstance(st, StageBundle):
                    for stage in st.stages:
                        for src in stage.sources:
                            ws_sources[src.name] = src
            new_consumer, cons_found = lower_to_atoms(s.consumer_body, spec_kind=spec_kind, c_dtype=c_dtype, smem_sources=ws_sources)
            out.append(s.with_bodies((s.producer_body, new_consumer)))
            found = found or cons_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_lowered = False
            for sub in s.nested():
                new_sub, sub_found = lower_to_atoms(sub, spec_kind=spec_kind, c_dtype=c_dtype, smem_sources=smem_sources)
                new_bodies.append(new_sub)
                any_lowered = any_lowered or sub_found
            found = found or any_lowered
            out.append(s.with_bodies(tuple(new_bodies)))
            continue
        out.append(s)
    return Body(out), found


def build_atom(
    atom_body: Body,
    *,
    spec_kind: str,
    c_dtype: DataType,
    smem_sources: dict[str, Source],
) -> Atom:
    """Package an AtomTile body into one :class:`Atom`.

    Pre-scans the body for the ``Write`` + ``Accum`` + a sample
    ``(a_load, b_load)`` pair, classifies which staged smem buffer is the A
    (M×K) and which is the B (K×N) operand, and hoists the ``Write`` out into
    the ``out_*`` fields. Only the buffer *names* are recorded — the expander
    re-harvests the live ``Source`` (its ``cache_axes`` track axis renumbering
    that a 005-time snapshot would not).

    mma.sync is smem→register only (no gmem-direct ldmatrix), so an unstaged
    operand RuleSkips here — pruning the warp-tier variant so the scalar tier
    covers that shape.
    """
    # Flatten every in-scope Source (enclosing + bundles nested inside the
    # AtomTile body) so the pre-scan classification + Source snapshot see the
    # full table. A local copy so sibling subtrees don't leak.
    bundle_sources = dict(smem_sources)
    for stage_bundle in _iter_bundles(atom_body):
        for stage in stage_bundle.stages:
            for src in stage.sources:
                bundle_sources[src.name] = src

    write_stmt, accum, a_load, b_load, _has_reduce = _scan_atom_body(atom_body, bundle_sources)
    if write_stmt is None:
        raise RuleSkipped("AtomTile body unrecognised — no Write")
    if accum is None or a_load is None or b_load is None:
        raise RuleSkipped(f"AtomTile body unrecognised — expected Load+Load+Accum chain (got accum={accum!r}, a={a_load!r}, b={b_load!r})")

    a_buffer, b_buffer = a_load.input, b_load.input
    if a_buffer not in bundle_sources or b_buffer not in bundle_sources:
        raise RuleSkipped("mma.sync requires staged smem operands (ldmatrix has no gmem-direct path)")

    # Hoist the store target out of the body — the expander re-emits it as a
    # ``RegStore`` epilogue. The Write is always a top-level child of the
    # AtomTile body (after the K_o loop / epilogue).
    new_body = Body(tuple(s for s in atom_body if not isinstance(s, Write)))
    return Atom(
        spec_kind=spec_kind,
        body=new_body,
        a_buffer=a_buffer,
        b_buffer=b_buffer,
        c_name=accum.name,
        a_name=a_load.names[0],
        b_name=b_load.names[0],
        out_buffer=write_stmt.output,
        out_index=write_stmt.index,
        c_dtype=c_dtype,
    )


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
    present, falling back to the bare body's Loads (shape C). Classifying
    the pair once (and matching every other reduce site against its
    buffers at expansion) keeps the fragment SSA names stable across
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


# ===========================================================================
# Expansion: Atom -> kernel fragment chain  (kernel/006_expand_atom)
# ===========================================================================


def expand_atoms(body: Body, *, smem_sources: dict[str, Source]) -> tuple[Body, bool]:
    """Walk ``body``; replace every :class:`Atom` with its expanded kernel
    fragment chain (:func:`expand_atom`). Threads ``smem_sources`` from
    enclosing ``StageBundle`` / ``WarpSpecialize`` scopes (mirroring 005's
    :func:`lower_to_atoms`) so the expander resolves the operand slab
    addressing from the *live* ``Source`` — its ``cache_axes`` track the axis
    renumbering applied since the ``Atom`` was emitted. Returns
    ``(new_body, found_any)``."""
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, Atom):
            out.extend(expand_atom(s, smem_sources=smem_sources))
            found = True
            continue
        if isinstance(s, StageBundle):
            inner_sources = dict(smem_sources)
            for stage in s.stages:
                for src in stage.sources:
                    inner_sources[src.name] = src
            new_bundle_body, body_found = expand_atoms(s.body, smem_sources=inner_sources)
            out.append(s.with_bodies((Body(s.stages), new_bundle_body)))
            found = found or body_found
            continue
        if isinstance(s, WarpSpecialize):
            ws_sources = dict(smem_sources)
            for st in s.producer_body.iter():
                if isinstance(st, StageBundle):
                    for stage in st.stages:
                        for src in stage.sources:
                            ws_sources[src.name] = src
            new_consumer, cons_found = expand_atoms(s.consumer_body, smem_sources=ws_sources)
            out.append(s.with_bodies((s.producer_body, new_consumer)))
            found = found or cons_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_found = False
            for sub in s.nested():
                new_sub, sub_found = expand_atoms(sub, smem_sources=smem_sources)
                new_bodies.append(new_sub)
                any_found = any_found or sub_found
            found = found or any_found
            out.append(s.with_bodies(tuple(new_bodies)))
            continue
        out.append(s)
    return Body(out), found


def expand_atom(atom: Atom, *, smem_sources: dict[str, Source]) -> tuple[Stmt, ...]:
    """Lower one :class:`Atom` to its kernel-IR fragment chain.

    Prepends the ``RegFragment`` decls, rewrites the body (each reduce
    SerialTile → ``ldmatrix a + ldmatrix b + mma.sync``; shape C → inline
    chain), and appends the ``RegStore`` epilogue from the hoisted ``out_*``.
    ``smem_sources`` carries the enclosing-scope ``Source``s; the bundles
    nested inside ``atom.body`` are harvested on top so per-operand slab
    addressing resolves from the live table."""
    spec = atom_spec(atom.spec_kind)
    c_frag = f"{atom.c_name}_frag"
    a_frag = f"{atom.a_name}_frag"
    b_frag = f"{atom.b_name}_frag"
    fragments = _emit_fragments(spec, c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, c_dtype=atom.c_dtype)
    store = _emit_store(spec, dst_buffer=atom.out_buffer, dst_index=atom.out_index, c_frag=c_frag)

    # Flatten every in-scope Source (enclosing + bundles nested inside the
    # body) so ``_mma_src_index`` resolves each operand's slab addressing.
    bundle_sources = dict(smem_sources)
    for stage_bundle in _iter_bundles(atom.body):
        for stage in stage_bundle.stages:
            for src in stage.sources:
                bundle_sources[src.name] = src

    has_reduce = any(st.is_reduce for st in atom.body.iter_of_type(SerialTile))
    if has_reduce:
        transformed = _transform_walk(
            atom.body,
            c_frag=c_frag,
            a_frag=a_frag,
            b_frag=b_frag,
            a_buffer=atom.a_buffer,
            b_buffer=atom.b_buffer,
            smem_sources=bundle_sources,
            spec=spec,
        )
        return (*fragments, *transformed, store)

    # Shape C: K filtered — inline chain from the body's two Loads.
    loads = [s for s in atom.body if isinstance(s, Load)]
    a_load = next((ld for ld in loads if ld.input == atom.a_buffer), None)
    b_load = next((ld for ld in loads if ld.input == atom.b_buffer), None)
    if a_load is None or b_load is None:
        raise RuleSkipped("Atom body (shape C) didn't match the A/B buffers")
    chain = _emit_chain(spec, a_load=a_load, b_load=b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=bundle_sources)
    return (*fragments, *chain, store)


def _transform_walk(
    body: Body,
    *,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    a_buffer: str,
    b_buffer: str,
    smem_sources: dict[str, Source],
    spec: AtomSpec,
) -> tuple[Stmt, ...]:
    """Recursively rewrite ``body`` in place. Replaces every
    ``SerialTile(is_reduce=True)`` body with an Mma chain (clearing the
    ``is_reduce`` flag — no Accum remains inside). Preserves every other Stmt
    structurally — StageBundle wraps + their Stages, AsyncWait, non-reduce
    SerialTile (K_o outer), Cond, etc. Descends into nested bodies via
    ``s.nested()`` so deeply-nested reduce sites (e.g. shape D's inner-K_o
    body) are reached. The ``Write`` was already hoisted off the ``Atom`` by
    :func:`build_atom`, so it never appears here."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, SerialTile) and s.is_reduce:
            chain = _build_mma_chain(
                s.body,
                a_buffer=a_buffer,
                b_buffer=b_buffer,
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
            new_body = _transform_walk(
                s.body,
                c_frag=c_frag,
                a_frag=a_frag,
                b_frag=b_frag,
                a_buffer=a_buffer,
                b_buffer=b_buffer,
                smem_sources=smem_sources,
                spec=spec,
            )
            # Stages stay byte-clean — no Mma rewrite happens inside a producer
            # scope. (All in-scope Sources are already flattened into
            # ``smem_sources`` by the caller, so no per-bundle re-harvest.)
            out.append(s.with_bodies((Body(s.stages), Body(new_body))))
            continue
        if s.nested():
            new_bodies = tuple(
                Body(
                    _transform_walk(
                        sub,
                        c_frag=c_frag,
                        a_frag=a_frag,
                        b_frag=b_frag,
                        a_buffer=a_buffer,
                        b_buffer=b_buffer,
                        smem_sources=smem_sources,
                        spec=spec,
                    )
                )
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
    a_buffer: str,
    b_buffer: str,
    c_frag: str,
    a_frag: str,
    b_frag: str,
    smem_sources: dict[str, Source],
    spec: AtomSpec,
) -> tuple[Stmt, ...]:
    """Build the per-reduce ``load a + load b + mma`` chain that replaces a
    reduce SerialTile's body. Matches this reduce site's Loads against the
    ``Atom``'s classified A/B buffers — shape D has prologue/inner/epilogue
    reduces all reading the same staged buffers, so the match is stable."""
    loads = [s for s in reduce_body if isinstance(s, Load)]
    if len(loads) != 2:
        raise RuleSkipped(f"reduce SerialTile body unrecognised — expected 2 Loads, got {len(loads)}")
    a_load = next((ld for ld in loads if ld.input == a_buffer), None)
    b_load = next((ld for ld in loads if ld.input == b_buffer), None)
    if a_load is None or b_load is None:
        raise RuleSkipped("reduce SerialTile Loads didn't match the Atom's A/B buffers")
    return _emit_chain(spec, a_load=a_load, b_load=b_load, a_frag=a_frag, b_frag=b_frag, c_frag=c_frag, smem_sources=smem_sources)


# --- Per-instruction leaf emitters -----------------------------------------


def _emit_fragments(spec: AtomSpec, *, c_frag: str, a_frag: str, b_frag: str, c_dtype: DataType) -> tuple[Stmt, ...]:
    """Register-array declarations. Three ``RegFragment`` decls; the ``c``
    array is zero-initialised at declaration, so there's no separate fill."""
    return (
        RegFragment(name=c_frag, role="c", shape=spec.shape, dtype=c_dtype),
        RegFragment(name=a_frag, role="a", shape=spec.shape, dtype=spec.operand_dtypes["a"]),
        RegFragment(name=b_frag, role="b", shape=spec.shape, dtype=spec.operand_dtypes["b"]),
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
    """The per-reduce ``ldmatrix a + ldmatrix b + mma.sync`` chain for one K-step."""
    a_src_index, a_ldm = _mma_src_index(a_load, smem_sources)
    b_src_index, b_ldm = _mma_src_index(b_load, smem_sources)
    # ldmatrix is smem→register only — both operands must be staged. RuleSkipping
    # here drops this warp-tier variant so the scalar tier covers the shape.
    if a_load.input not in smem_sources or b_load.input not in smem_sources:
        raise RuleSkipped("mma.sync requires staged smem operands (ldmatrix has no gmem-direct path)")
    # Thread the per-Source TMA swizzle mode (S3 of
    # plans/mma-sync-smem-swizzle.md) so the ldmatrix consumer applies the
    # matching per-lane chunk XOR. NONE when the source wasn't swizzled.
    a_swz = smem_sources[a_load.input].swizzle.value
    b_swz = smem_sources[b_load.input].swizzle.value
    return (
        LdmatrixLoad(frag=a_frag, src_buffer=a_load.input, src_index=a_src_index, role="a", ldm=a_ldm, swizzle=a_swz),
        LdmatrixLoad(frag=b_frag, src_buffer=b_load.input, src_index=b_src_index, role="b", ldm=b_ldm, swizzle=b_swz),
        MmaSyncPtx(c_frag=c_frag, a_frag=a_frag, b_frag=b_frag, shape=spec.shape, ab_dtype=spec.operand_dtypes["a"].name),
    )


def _emit_store(spec: AtomSpec, *, dst_buffer: str, dst_index: tuple, c_frag: str) -> Stmt:
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
