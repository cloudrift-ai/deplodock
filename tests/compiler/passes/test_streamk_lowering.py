"""Stream-K M2 — KernelOp(PersistentTile) lowering + work-range plumbing.

Exercises the CudaOp lowering surface for a hand-built persistent kernel (no
upstream pass emits PersistentTile until M3): the grid resolves to the reserved
``num_sms`` sym name, the two work-range arrays land in ``arg_order`` and the
kernel signature as ``const int*``, the streamk metadata is carried on the
CudaOp, and the per-CTA range partition tiles the work with one owner per unit.
"""

from __future__ import annotations

import importlib

import numpy as np

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.cuda.ir import STREAMK_NUM_SMS
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import GridTile, PersistentTile, RegisterTile, ThreadTile, TileOp

lower = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.cuda.010_lower_kernelop")
program = importlib.import_module("deplodock.compiler.backend.cuda.program")
streamk = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.098_persistent_streamk")


class _StubNode:
    def __init__(self, op):
        self.op = op
        self.id = "k"


def _persistent_kernelop(m_b: int = 13, n_b: int = 13) -> KernelOp:
    """``PersistentTile(M_b, N_b) > ThreadTile > Load+Write`` — minimal matmul
    shape so the lowering derives a real signature (one input ``A``, output
    ``C``) and persistent launch geometry."""
    inner = ThreadTile(
        axes=(Axis("m_t", 16), Axis("n_t", 16)),
        body=Body((Load(name="a", input="A", index=(Var("m_t"), Var("n_t"))), Write(output="C", index=(Var("m_t"),), value="a"))),
    )
    pt = PersistentTile(axes=(Axis("m_b", m_b), Axis("n_b", n_b)), body=Body((inner,)))
    return KernelOp(body=Body((pt,)), name="k_streamk")


def test_launch_geometry_grid_is_num_sms():
    geo = lower._launch_geometry(_persistent_kernelop())
    grid, block, runtime_args, streamk = geo
    assert grid == ((STREAMK_NUM_SMS,), (1,), (1,))
    # Block factors stay un-multiplied (resolve_dim folds them at launch → 256).
    assert block == ((16, 16), (1,), (1,))
    assert streamk is not None
    start, end, total = streamk
    assert (start, end) == ("streamk_work_start", "streamk_work_end")
    assert total == 13 * 13  # product of block-axis extents = tile count


def test_cudaop_threads_work_arrays_through():
    op = lower.rewrite(None, _StubNode(_persistent_kernelop()))
    # Work arrays slot after buffers/descriptors in arg_order.
    assert op.arg_order[-2:] == ("streamk_work_start", "streamk_work_end")
    assert op.streamk_work_arrays == ("streamk_work_start", "streamk_work_end")
    assert op.streamk_total_units == 169
    assert op.grid == ((STREAMK_NUM_SMS,), (1,), (1,))
    # Signature declares both as const int* pointers.
    assert "const int* __restrict__ streamk_work_start" in op.kernel_source
    assert "const int* __restrict__ streamk_work_end" in op.kernel_source
    # And the body reads them per-CTA.
    assert "streamk_work_start[blockIdx.x]" in op.kernel_source
    assert "for (int tile_iter = __wbeg; tile_iter < __wend; tile_iter++)" in op.kernel_source


def test_streamk_ranges_partition_work_one_owner():
    # 169 tiles over 170 CTAs: per_cta=1, CTAs 0..168 own one tile, CTA 169 idle.
    starts, ends = program._streamk_ranges(169, 170)
    assert len(starts) == len(ends) == 170
    assert starts.dtype == np.int32 and ends.dtype == np.int32
    # Contiguous, non-overlapping, covers exactly [0, 169).
    covered = np.concatenate([np.arange(s, e) for s, e in zip(starts.tolist(), ends.tolist(), strict=True)])
    assert covered.tolist() == list(range(169))
    assert (ends[169], starts[169]) == (169, 169)  # trailing CTA idles


def test_streamk_ranges_two_waves():
    # 340 tiles over 170 CTAs: per_cta=2, every CTA owns exactly 2 tiles.
    starts, ends = program._streamk_ranges(340, 170)
    assert (ends - starts).tolist() == [2] * 170
    assert ends[-1] == 340


def _matmul_tileop(m_b: int, n_b: int) -> TileOp:
    inner = ThreadTile(
        axes=(Axis("m_t", 16), Axis("n_t", 16)),
        body=Body((RegisterTile(axes=(Axis("m_r", 1),), body=Body((Write(output="C", index=(Var("m_t"),), value="acc"),))),)),
    )
    grid = GridTile(axes=(Axis("m_b", m_b), Axis("n_b", n_b)), body=Body((inner,)))
    return TileOp(body=Body((grid,)), name="k_matmul", knobs={"FM": 1, "FN": 1, "BM": 16, "BN": 16})


def _ctx(num_sms: int = 170):
    from deplodock.compiler.context import Context

    return Context.from_target((12, 0)).__class__(compute_capability=(12, 0), num_sms=num_sms)


def test_streamk_fork_offered_in_wave_tail_regime(monkeypatch):
    monkeypatch.delenv("DEPLODOCK_STREAMK", raising=False)
    node = _StubNode(_matmul_tileop(13, 13))  # 169 CTAs ≈ 170 SMs → tail regime
    variants = streamk.rewrite(_ctx(), None, node)
    knobs = sorted(v.knobs["STREAMK"] for v in variants)
    assert knobs == [False, True]


def test_streamk_fork_skipped_when_waves_saturated(monkeypatch):
    import pytest

    from deplodock.compiler.pipeline import RuleSkipped

    monkeypatch.delenv("DEPLODOCK_STREAMK", raising=False)
    node = _StubNode(_matmul_tileop(200, 200))  # 40000 CTAs ≫ 8·170 → saturated
    with pytest.raises(RuleSkipped, match="saturated"):
        streamk.rewrite(_ctx(), None, node)


def test_streamk_env_pin_bypasses_wave_gate(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_STREAMK", "1")
    node = _StubNode(_matmul_tileop(200, 200))  # saturated, but pin is authoritative
    variants = streamk.rewrite(_ctx(), None, node)
    assert [v.knobs["STREAMK"] for v in variants] == [True]


def test_streamk_ranges_ragged_tail():
    # 175 tiles over 170 CTAs: per_cta=2; first CTAs get 2, work runs out early,
    # remainder idle — still exactly tiles the range with one owner each.
    starts, ends = program._streamk_ranges(175, 170)
    covered = np.concatenate([np.arange(s, e) for s, e in zip(starts.tolist(), ends.tolist(), strict=True)])
    assert covered.tolist() == list(range(175))
    assert int(ends.max()) == 175
