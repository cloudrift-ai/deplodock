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

The MMA-fragment lowering helpers (``lower_atom_tiles`` and the per-reduce
chain builders) live in the sibling ``_mma`` module; this file is the pass
entry point. Eligibility: ``op.knobs["ATOM_KIND"]`` set (only warp-tier
matmul rows carry this knob — the scalar planner branch leaves it unset and
this pass skips). Idempotence: after this pass the AtomTile is gone, so on
a second visit the pattern doesn't match and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._mma import lower_atom_tiles, resolve_c_dtype
from deplodock.compiler.pipeline.passes.lowering.tile._atom import atom_spec
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    parallel_tile_of,
    replace_parallel_tile_body,
    single_tile,
)

PATTERN = [Pattern("root", TileOp)]


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
    # match still holds. See ``resolve_c_dtype`` + ``MmaStore`` docstrings.
    c_dtype_override = resolve_c_dtype(root, spec.operand_dtypes["c"])

    lowered, found = lower_atom_tiles(tt.body, spec=spec, c_dtype_override=c_dtype_override, smem_sources={})
    if not found:
        # Could happen on a second visit (AtomTile already consumed).
        raise RuleSkipped("no AtomTile in body — already lowered")

    rebuilt = replace_parallel_tile_body(outer, lowered)
    return TileOp(body=body[:idx] + (rebuilt,) + body[idx + 1 :], name=op.name, knobs=op.knobs)
