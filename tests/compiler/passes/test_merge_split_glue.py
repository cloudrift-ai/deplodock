"""``lowering/tile/006_merge_split_glue`` — post-split re-fusion of glue kernels.

Structural (no-GPU) coverage of the guard set and the target merges on the synthetic
split shapes from ``test_split_demoted``. The CUDA accuracy of the flagship merge (the
gated-MLP combine folded into a gemm epilogue, on mma.sync) is pinned by
``test_split_demoted.test_gated_mlp_split_mma_accuracy_cuda`` (3-kernel contract). The
norm→RoPE *backward* merge (a reduce-heavy producer read through two Loads — the case the
wrapper's dropped multi-load guard exists for) has no faithful synthetic: the loop tier
either fuses the norm into the cone or the both-Accum guard correctly blocks; it is
exercised end-to-end on the real model (Qwen3-Embedding layer 0 deploys the merged
``k_mean_linear_reduce_*`` kernels — plans/qwen3-embedding-layer0-tune-findings.md
finding 1's fix)."""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import Accum, LoopOp
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.pipeline import Run
from deplodock.compiler.pipeline.search.two_level import outer_pipeline

from .test_split_demoted import _collapsed_matmul_graph, _gated_mlp_graph, _norm_linear_graph

_CTX = Context.from_target((12, 0))


_GLUE_HINT = "tile.split_glue"


def _resolve_outer(graph: Graph) -> Graph:
    """Drive the outer pipeline (frontend + loop + the pre-partition tile head,
    which now includes the 006-009 re-fusion aliases) to its terminal."""
    run = Run(pipeline=outer_pipeline(), ctx=_CTX)
    terminal, _ = run.resolve(graph, lambda fp: fp.options[0])
    return terminal


def _loop_ops(graph: Graph) -> dict[str, LoopOp]:
    return {nid: n.op for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp)}


def _glued_ids(graph: Graph) -> list[str]:
    """Node ids carrying the re-fused-composite hint (the marker lives on
    ``Node.hints``, NOT ``op.knobs`` — it must never become a prior feature)."""
    return [nid for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp) and n.hints.get(_GLUE_HINT)]


def _has_accum(op: LoopOp) -> bool:
    return any(isinstance(s, Accum) for s in op.body.iter())


def test_gated_mlp_combine_merges_into_gemm(monkeypatch) -> None:
    """The split's 4 kernels (xn + mm0 + mm1 + combine) re-fuse to 3: one gemm
    absorbs the pointwise combine as its epilogue. The merged op carries the
    decision stamps, fresh structural features, a name, and stays atom-eligible
    (the whole point — glue removal without losing the MMA tier)."""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    terminal = _resolve_outer(_gated_mlp_graph())
    ops = _loop_ops(terminal)
    assert len(ops) == 3, f"xn + gemm + merged gemm-with-epilogue, got {sorted(ops)}"
    glued = _glued_ids(terminal)
    assert len(glued) == 1, f"exactly one re-fused composite, got {glued}"
    merged = terminal.nodes[glued[0]].op
    assert merged.knobs.get("SPLIT_CONE") is True, "005's idempotence stamp must survive the merge"
    assert "SPLIT_GLUE" not in merged.knobs, "the composite marker must stay out of op.knobs (prior training features)"
    assert any(k.startswith("S_") for k in merged.knobs), "the 009 alias must restamp structural features"
    assert merged.name, "the 008 alias must restamp the kernel name"
    assert _has_accum(merged), "the merged op is the gemm (the combine rode along as epilogue)"
    assert any(is_atom_eligible(atom, merged, _CTX, graph=terminal) for atom in ATOM_REGISTRY.values()), (
        "the merged gemm must stay atom-eligible (guard 6's contract)"
    )


def test_gated_mlp_second_gemm_does_not_remerge(monkeypatch) -> None:
    """One-gemm-per-kernel: after the first gemm absorbs the combine, the other
    gemm must NOT merge into the composite (both bodies carry an Accum) — that
    would rebuild the dual-accum kernel the split exists to escape."""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    terminal = _resolve_outer(_gated_mlp_graph())
    ops = _loop_ops(terminal)
    glued = set(_glued_ids(terminal))
    standalone_gemms = [nid for nid, op in ops.items() if _has_accum(op) and nid not in glued and "__mm" in nid]
    assert len(standalone_gemms) == 1, f"the second extracted gemm must survive un-merged, got {sorted(ops)}"


def test_split_glue_pin_off_keeps_all_four(monkeypatch) -> None:
    """``DEPLODOCK_SPLIT_GLUE=0`` is the with/without-re-fusion A/B switch: the
    split's full 4-kernel set survives and nothing carries the stamp."""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    monkeypatch.setenv("DEPLODOCK_SPLIT_GLUE", "0")
    terminal = _resolve_outer(_gated_mlp_graph())
    assert len(_loop_ops(terminal)) == 4, "pin off must keep xn + mm0 + mm1 + combine"
    assert not _glued_ids(terminal)


def test_norm_linear_xn_not_reinlined(monkeypatch) -> None:
    """K-cell protection: the single-accum split's xn producer feeds the gemm's
    K loop — re-inlining it would re-demote the cell, so the pair must stay
    split (and un-stamped)."""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    terminal = _resolve_outer(_norm_linear_graph())
    ops = _loop_ops(terminal)
    assert len(ops) == 2, f"xn + consumer must stay separate, got {sorted(ops)}"
    assert any(nid.endswith("__xn") for nid in ops)
    assert not _glued_ids(terminal)


def test_collapsed_layout_pair_stays_split(monkeypatch) -> None:
    """The o_proj relayout shape: the contiguizing copy feeds the gemm's K cell
    (K-cell guard), and the reverse (gemm→copy) pair only exists with an
    upstream producer — within this slice nothing merges. Splice support for
    the deployed-graph attn@V→copy backward merge is the documented v2 gap."""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    terminal = _resolve_outer(_collapsed_matmul_graph())
    assert len(_loop_ops(terminal)) == 2, "copy producer + gemm must stay separate"
    assert not _glued_ids(terminal)


def test_loop_tier_fusion_unchanged() -> None:
    """The wrapper lives only in lowering/tile: a loop-tier-only pipeline never
    fires it (scope key needs SPLIT stamps that don't exist there) and the
    fused gated-MLP kernel set is what it always was — one kernel."""
    out = Pipeline.build(LOOP_PASSES).run(_gated_mlp_graph(), ctx=_CTX)
    ops = _loop_ops(out)
    assert len(ops) == 1, f"loop tier must still fully fuse the gated MLP, got {sorted(ops)}"
    assert not any("SPLIT_CONE" in op.knobs for op in ops.values())
    assert not _glued_ids(out)


def test_outer_terminal_matches_greedy_kernel_count(monkeypatch) -> None:
    """Outer ≡ greedy on the kernel-set cardinality: the outer head (looped to
    quiescence) and the full greedy pipeline (one LoopOp batch per scan) must
    realize the same 3-kernel set for the pinned gated-MLP split — the
    one-batch contract guard 2 (SPLIT_GLUE marker) exists to keep true."""
    from deplodock.compiler.ir.cuda.ir import CudaOp
    from deplodock.compiler.pipeline import CUDA_PASSES
    from deplodock.compiler.pipeline.search.db import SearchDB

    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    outer_ops = _loop_ops(_resolve_outer(_gated_mlp_graph()))
    full = Pipeline.build(CUDA_PASSES).run(_gated_mlp_graph(), ctx=_CTX, db=SearchDB())
    cuda_ids = [nid for nid, n in full.nodes.items() if isinstance(n.op, CudaOp)]
    assert len(outer_ops) == len(cuda_ids) == 3, f"outer={sorted(outer_ops)} greedy={sorted(cuda_ids)}"
