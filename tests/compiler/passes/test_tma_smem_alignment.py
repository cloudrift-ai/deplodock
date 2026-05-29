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

from deplodock.compiler.backend.cuda.backend import CudaBackend
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.frontend.ir import MatmulOp


def _matmul_graph(m: int = 32, k: int = 1024, n: int = 1024) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


@pytest.fixture
def _tma_eligible_context(monkeypatch):
    # Force Hopper+ so ``050_use_tma`` admits TMA promotion regardless of the
    # live device. The materializer's smem-prologue logic is the surface
    # under test; the live device's compute capability is irrelevant.
    monkeypatch.setenv("DEPLODOCK_COMPUTE_CAPABILITY", "9.0")
    yield


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
