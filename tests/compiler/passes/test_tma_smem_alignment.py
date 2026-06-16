"""Regression — TMA smem destinations must be 128-byte aligned end-to-end.

``cp.async.bulk.tensor`` raises ``Misaligned shared or local address`` at
runtime when its smem destination isn't aligned to the TMA programming
guide's recommendation. Two things have to land together:

1. The buffer base needs ``__align__(128)``.
2. Each ring slot's stride (= the buffer's inner extent in bytes) needs to
   be a multiple of 128 B, so successive slots stay 128 B-aligned from the
   128 B-aligned base.

Before this fix the materializer's pre-emitted Smem decl used dtype-only
alignment (``align=0`` for fp32, ``align=16`` for fp16) and inherited the
TMA box's natural inner extent — on small-BK degenerate variants
(``BK=16, BM=1, BN=128, FM=1`` on the SDPA-prologue + linear-reduce
kernels) the second ring slot landed at a 64 B offset from a 128 B base.
The in-flight TMA never completed, the consumer ``mbarrier.wait`` spun
forever, the bench watchdog timed out at 1000 ms, and the autotune pinned
those variants ``bench_fail`` even though the structure was sound.

This test pins both invariants on a Tile-IR built end-to-end through the
real kernel passes — the assertion runs on the rendered CUDA source so a
regression that drops either piece is loud and fast.
"""

from __future__ import annotations

import re

import pytest

from deplodock.compiler import target as target_mod
from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp


def _matmul_graph(m: int = 32, k: int = 1024, n: int = 1024) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _norm_linear_graph(dtype, s: int = 32, h: int = 128, i: int = 512) -> Graph:
    """``RmsNorm(x) @ W.T`` — the fused norm+matmul shape of Qwen3-Embedding's
    ``k_linear_mean_reduce``. The norm reduction stages ``x`` in ``BK``-element
    inner slabs; at ``BK=32`` that slab is 32 elems = 128 B in fp32 but only
    64 B in fp16 — the dtype the TMA-alignment eligibility gate must respect."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, s, h), dtype=dtype), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (h,), dtype=dtype), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (i, h), dtype=dtype), node_id="w")
    g.add_node(op=RmsNormOp(), inputs=["x", "wn"], output=Tensor("xn", (1, s, h), dtype=dtype), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "w"], output=Tensor("y", (1, s, i), dtype=dtype), node_id="y")
    g.inputs = ["x", "wn", "w"]
    g.outputs = ["y"]
    return g


@pytest.fixture
def _tma_eligible_context():
    # Force Hopper+ so ``050_use_tma`` admits TMA promotion regardless of the
    # live device — the gate reads ``ctx.compute_capability``, which honors the
    # ``set_target`` override (NOT an env var), so on a GPU-less CI runner
    # ``compute_capability()`` would otherwise fall back to ``(0, 0)`` and
    # decline TMA. The materializer's smem-prologue logic is the surface under
    # test; the live device's compute capability is irrelevant.
    target_mod.set_target((9, 0))
    try:
        yield
    finally:
        target_mod.set_target(None)


def _smem_decl_align(cuda_src: str, buf_name: str) -> int | None:
    """Return the ``__align__(N)`` byte count stamped on ``buf_name``'s Smem
    decl, or ``None`` if the decl is missing / unaligned."""
    pattern = rf"__shared__\s+__align__\((\d+)\)\s+\w+\s+{re.escape(buf_name)}\s*\[(\d+)\];"
    m = re.search(pattern, cuda_src)
    return int(m.group(1)) if m else None


def _smem_decl_total_elems(cuda_src: str, buf_name: str) -> int | None:
    pattern = rf"__shared__\s+(?:__align__\(\d+\)\s+)?\w+\s+{re.escape(buf_name)}\s*\[(\d+)\];"
    m = re.search(pattern, cuda_src)
    return int(m.group(1)) if m else None


def _compile_with_knobs(monkeypatch, knobs: dict[str, str]) -> str:
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    backend = CudaBackend()
    compiled = backend.compile(_matmul_graph())
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops, "expected at least one CudaOp"
    return "\n".join(op.kernel_source for op in cuda_ops)


def test_tma_smem_base_is_128_byte_aligned(monkeypatch, _tma_eligible_context):
    """Every TMA-targeted Smem buffer must carry ``__align__(128)``.

    Knobs chosen to force the materializer's TMA promotion path. The cap-at-128
    enumeration still admits ``BK=16`` so the failing-variant geometry from
    the Qwen3-Embedding tune surfaces here too.
    """
    cuda_src = _compile_with_knobs(
        monkeypatch,
        knobs={"BK": "16", "BM": "1", "BN": "128", "BR": "1", "FM": "1", "FN": "1", "SPLITK": "1", "STAGE": "1"},
    )
    # ``cp.async.bulk.tensor`` is the unambiguous TMA marker; if it's absent
    # the planner declined to promote and the test asserts nothing useful.
    if "cp.async.bulk.tensor" not in cuda_src:
        pytest.skip("TMA promotion did not fire for this knob mix")
    aligns = re.findall(r"__shared__\s+__align__\((\d+)\)\s+\w+\s+(\w+_smem)\b", cuda_src)
    assert aligns, "no aligned __shared__ decls found; TMA emit must stamp __align__(128)"
    for align_str, name in aligns:
        assert int(align_str) >= 128, f"{name!r} declared with __align__({align_str}); TMA needs >= 128"


def test_tma_smem_slot_stride_is_128_byte_multiple(monkeypatch, _tma_eligible_context):
    """The buffer total must be a multiple of 128 B / element size so each
    ring slot's offset (``slot * inner_extent``) stays 128 B-aligned. For
    fp32 that's ``total_elems % 32 == 0``; previously fp32 slabs with
    ``BK=16`` (inner=16) and ``buffer_count=2`` landed at 32 elements ⇒
    slot 1 at offset 16 ⇒ 64 B = non-128 B-aligned ⇒ misaligned TMA.
    """
    cuda_src = _compile_with_knobs(
        monkeypatch,
        knobs={"BK": "16", "BM": "1", "BN": "128", "BR": "1", "FM": "1", "FN": "1", "SPLITK": "1", "STAGE": "1"},
    )
    if "cp.async.bulk.tensor" not in cuda_src:
        pytest.skip("TMA promotion did not fire for this knob mix")
    for name in re.findall(r"__shared__\s+__align__\(128\)\s+\w+\s+(\w+_smem)\s*\[", cuda_src):
        total = _smem_decl_total_elems(cuda_src, name)
        assert total is not None
        # 128 B / sizeof(float) == 32 elements; the slot stride is
        # ``total // buffer_count``. We don't know buffer_count here, but
        # ``total`` must itself be a 32-multiple for every buffer_count <= 4.
        assert total % 32 == 0, (
            f"{name!r}: total_elems={total} is not a 32-element (128 B) multiple — "
            "ring slot 1 lands at a non-128 B-aligned offset, misaligning TMA"
        )


def _compile_norm_linear(monkeypatch, dtype, knobs: dict[str, str]) -> str:
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    compiled = CudaBackend().compile(_norm_linear_graph(dtype))
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops, "expected at least one CudaOp"
    return "\n".join(op.kernel_source for op in cuda_ops)


# Knobs reproducing the #244 dynamic-tune wedge: a scalar (``MMA=0``)
# cooperative norm-reduce whose ``BK=32`` inner slab is double-buffered
# (``RING=2``). Its per-slot box is a single 32-elem axis — 64 B in fp16, NOT a
# 128 B multiple — so the second ring slot would land at a 64 B offset and
# ``cp.async.bulk.tensor`` faults (``CUDA_ERROR_MISALIGNED_ADDRESS`` → 1 s
# watchdog hang → bench_fail). The eligibility gate must decline TMA here.
_WEDGE_KNOBS = {"BK": "32", "BM": "1", "BN": "128", "BR": "1", "FM": "1", "FN": "1", "SPLITK": "1", "STAGE": "1", "MMA": "0", "RING": "2"}


def test_fp16_subaligned_ring_slot_declines_tma_fp32_keeps_it(monkeypatch, _tma_eligible_context):
    """Dtype-aware ring-slot eligibility: the same scalar norm+matmul, same knobs,
    must DECLINE TMA in fp16 (per-slot box = 32 elems = 64 B, not a 128 B multiple
    at ``RING=2``) but still PROMOTE in fp32 (32 elems = 128 B). The 128 B slot
    check sized off the fp32 ``BYTES_PER_ELEM`` constant let the fp16 64 B slab
    through as 128 B, the materializer's slot pad (same constant) left it
    unpadded, and the second ring slot landed at a 64 B offset →
    ``cp.async.bulk.tensor`` device hang (``CUDA_ERROR_MISALIGNED_ADDRESS``, the
    #244 ``k_linear_mean_reduce`` wedge). Declining routes the fp16 slab to
    cp.async (no 128 B rule). Compile-only — no CUDA device needed."""
    fp32_src = _compile_norm_linear(monkeypatch, F32, _WEDGE_KNOBS)
    assert "cp.async.bulk.tensor" in fp32_src, (
        "fp32 32-elem slot is 128 B and must still promote to TMA — the dtype-aware guard must not over-decline the correctly-aligned case"
    )
    fp16_src = _compile_norm_linear(monkeypatch, F16, _WEDGE_KNOBS)
    assert "cp.async.bulk.tensor" not in fp16_src, (
        "fp16 32-elem ring slot is 64 B (not a 128 B multiple) — TMA must be declined "
        "(fall back to cp.async), else cp.async.bulk.tensor stores misalign and hang"
    )
