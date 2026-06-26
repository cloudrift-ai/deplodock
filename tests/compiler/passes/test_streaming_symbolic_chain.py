"""Symbolic streaming flash routes through ``chain_build`` (the FA-2 shared-score form).

A streaming-flash nest whose KV (stream) axis is **symbolic** (``seq_len``) can't be
cooperatively tiled (``BR=BK=1``), so ``monoid_build`` would emit a fully-serial stream
that recomputes the QK^T score per P@V output ``d`` — O(D) redundant, catastrophically
slow (Finding 1, ``plans/qwen3-embedding-0.6b-layer0-tune-findings.md``). ``chain_build``
restructures it: ``d`` rides a REGISTER vector ``O[d]`` and the score is computed ONCE per
KV step and shared across ``d``. The KV stream stays a serial runtime-bounded loop (no
tiling → no masking), so ``chain_build`` covers a symbolic hinge. ``070_coop_reduce``
therefore makes ``chain_build`` the **enumeration default** for a symbolic streaming flash
(it stays a pin-gated opt-in for a static stream — greedy keeps the scalar nest there).
This is the **fallback** for the symbolic shapes the tensor-core warp chain declines: as of
Phase 3 of ``plans/smem-tiled-symbolic-flash.md``, an *eligible* symbolic flash (fp16/bf16,
``D%16==0``, equal-head or GQA) is intercepted **before** enumeration by
the ``070_coop_reduce`` warp-flash fork and deployed as the smem-tiled tensor-core warp chain (the perf
win); ``chain_build`` then serves only the non-eligible symbolic flashes (fp32, odd ``D``,
additive mask). These tests seed the enumeration directly (the buffers are **fp32**, so the
warp chain doesn't apply) — they pin the ``chain_build`` fallback routing.

Accuracy of the symbolic chain path is guarded end-to-end by the ``*_dynamic_matches_torch``
flash tests in ``tests/compiler/e2e/test_flash_attention.py`` (SDPA / GQA+causal / additive
mask, all over symbolic ``seq_len``); these tests pin the enumeration ROUTING (CPU only).
"""

from __future__ import annotations

import importlib

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Tensor
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import Binding, Buffer, Space, TileGraphOp
from deplodock.compiler.pipeline.passes.loop.recognize._flash import build_flash_frag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag

_coop = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.enumeration.070_coop_reduce")


def _flash_op(s_dim: Dim) -> TileGraphOp:
    """Seed a ``TileGraphOp`` for a small ``(B,H,S,D)`` flash whose S (query + KV) axis is
    ``s_dim`` — built straight from the recognizer + the 010_build seed, no torch."""
    B, H, D = 1, 2, 8
    shp = (Dim(B), Dim(H), s_dim, Dim(D))
    out = Tensor("o", shp, F32)
    loop = build_flash_frag("q", "k", "v", shp, shp, shp, out, causal=False).nodes["o"].op
    dag = iter_dag(loop)
    regime = classify(dag)
    assert regime is not None and regime.algebra is AlgebraKind.MONOID and dag.streaming and dag.chain is not None
    buffers = {n: Buffer(name=n, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM) for n, t in loop.inputs.items()}
    for t in loop.outputs.values():
        buffers[t.name] = Buffer(name=t.name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM)
    return TileGraphOp(
        name=loop.name,
        tilegraph=seed_graph(dag, kernel_name=loop.name, buffers=buffers),
        dag=dag,
        algebra=regime.algebra,
        target_names=regime.target_names,
        leading=tuple(dag.leading),
        seed_key=loop.body.structural_key(),
        buffers=buffers,
    )


def _has_shared_score_register(leaf: TileGraphOp) -> bool:
    """A chain_build leaf carries exactly one non-degenerate REGISTER domain axis — the P@V
    output ``d`` the score is shared across (``O[d]``). A monoid_build leaf has none
    (register forced to 1)."""
    binding = leaf.tilegraph.schedule.binding
    block = leaf.tilegraph.blocks[0]
    reg = [a for a in block.domain if binding.get(a.name) is Binding.REGISTER and a.extent.as_static() > 1]
    return len(reg) == 1


def test_symbolic_stream_routes_to_chain_build():
    """A symbolic KV stream routes EVERY streaming leaf through chain_build: the score edge
    is placed INLINE and the P@V output ``d`` rides a register vector — the shared-score
    form, not the per-``d``-recompute serial stream."""
    op = _flash_op(Dim("seq_len"))
    assert op.dag.k_bound is not None, "the streaming axis must be symbolic for this case"
    leaves = _coop._streaming_leaves(op)
    assert leaves, "symbolic streaming flash produced no leaves"
    score_key = fam.place_key(op.dag.chain.score)
    for leaf in leaves:
        assert leaf.knobs.get(score_key) == fam.INLINE, "symbolic chain leaf must place the score INLINE"
        assert _has_shared_score_register(leaf), "symbolic chain leaf must carry the O[d] register vector"


def test_static_stream_stays_scalar_monoid_by_default():
    """A static KV stream (no ``CHAIN`` pin) keeps the scalar ``monoid_build`` streaming nest
    — the chain restructuring stays a pin-gated opt-in there, so the deployed static flash is
    unchanged."""
    op = _flash_op(Dim(64))
    assert op.dag.k_bound is None, "the streaming axis must be static for this case"
    score_key = fam.place_key(op.dag.chain.score)
    for leaf in _coop._streaming_leaves(op):
        assert score_key not in leaf.knobs, "static default must not place the score INLINE (no chain)"
        assert not _has_shared_score_register(leaf), "static default must stay the scalar monoid stream"
