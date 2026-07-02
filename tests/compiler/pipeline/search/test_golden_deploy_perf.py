"""``diagnostics.golden_deploy_perf`` — the no-rebench ``vs gold`` column for
``emmy eval prior``: per golden shape, the deployable (-O3) latency of the
prior's predicted-best reservoir config over the golden's recorded ``emmy_us``.

The load-bearing case is dtype separation: an fp32 square and its ``.fp16`` twin
share (free-dim product, reduce extent), so the shape key MUST split on
``S_dtype_f32`` — keying on an mma marker (which the fp16 row may not carry) merges
them and steals the fp16 latency for the fp32 row.
"""

from __future__ import annotations

import pytest

from emmy.compiler.pipeline.search import golden as golden_mod
from emmy.compiler.pipeline.search.golden import goldens_by_name
from emmy.compiler.pipeline.search.prior import diagnostics


@pytest.fixture(autouse=True)
def _single_gpu_goldens(monkeypatch):
    """Pin the golden set to one card (RTX 5090). With multiple per-GPU files a name
    recurs once per card and the GPU-blind ``ShapeKey`` join would mix their
    latencies (5090 / PRO 6000 even share ``compute_cap (12, 0)``), making these
    shape-keyed assertions depend on which goldens dir is checked in. Off-GPU here,
    so ``goldens_for_live_gpu`` can't auto-scope — inject the single-card set."""
    one = [g for g in golden_mod.GOLDEN_CONFIGS if g.gpu_name == "NVIDIA GeForce RTX 5090"]
    monkeypatch.setattr(golden_mod, "GOLDEN_CONFIGS", one)


class _FakePrior:
    """A prior with a hand-built reservoir; ``mean_score`` is constant (each group
    here has one -O3 row, so the argmin pick is unambiguous). ``pick`` mirrors the
    real ``Prior.pick`` model-argmin fallback (constant scores → first row)."""

    def __init__(self, rows):
        self._dataset = rows  # list[(stamped_knobs, latency_us)]

    def mean_score(self, _feats):
        return 0.0

    def pick(self, rows):
        scores = [self.mean_score(r) for r in rows]
        best_i = min(range(len(scores)), key=scores.__getitem__)
        return best_i, scores[best_i]


def _row(free_prod, reduce_max, *, fp32, h_opt, latency):
    knobs = {
        "H_opt": float(h_opt),
        "S_ext_free_prod": float(free_prod),
        "S_ext_reduce_max": float(reduce_max),
        ("S_dtype_f32" if fp32 else "S_dtype_f16"): 2.0,
        # The matmul histogram markers (_matmul_sig): product → reduce-add, 2 inputs.
        "S_reduce_add": 1.0,
        "S_pw_multiply": 1.0,
        "S_n_distinct_input": 2.0,
        "BM": 8,
        "BN": 16,
    }
    return (knobs, latency)


def test_dtype_separation_and_o3_filter():
    g32 = goldens_by_name("square.512")[0].emmy_us  # fp32 golden latency (-O3)
    g16 = goldens_by_name("square.512.fp16")[0].emmy_us
    fp = 512 * 512  # both square.512 and .fp16 share (free_prod, reduce)

    prior = _FakePrior(
        [
            _row(fp, 512, fp32=True, h_opt=3, latency=g32 * 2.0),  # fp32 -O3 winner → ratio 2.0
            _row(fp, 512, fp32=False, h_opt=3, latency=g16 * 0.5),  # fp16 -O3 winner → ratio 0.5
            _row(fp, 512, fp32=True, h_opt=1, latency=g32 * 9.0),  # -O1 row must be IGNORED
        ]
    )
    perf = diagnostics.golden_deploy_perf(prior)

    # Each dtype matches its own -O3 row — no cross-contamination (the merge bug would
    # make the fp32 row pick the smaller fp16 latency).
    assert perf["square.512"] == pytest.approx(2.0)
    assert perf["square.512.fp16"] == pytest.approx(0.5)


def test_shape_without_o3_is_omitted():
    # square.1024 has only an -O1 row → no deployable measurement → omitted ('—').
    prior = _FakePrior([_row(1024 * 1024, 1024, fp32=True, h_opt=1, latency=50.0)])
    perf = diagnostics.golden_deploy_perf(prior)
    assert "square.1024" not in perf


def test_kernel_filter_restricts_shapes():
    g32 = goldens_by_name("square.512")[0].emmy_us
    prior = _FakePrior([_row(512 * 512, 512, fp32=True, h_opt=3, latency=g32)])
    assert set(diagnostics.golden_deploy_perf(prior, "square.512")) <= {"square.512"}


def test_non_matmul_group_with_colliding_extents_is_excluded():
    """A reduce-shaped op group that happens to share a matmul golden's
    (free_prod, reduce_max, dtype) must not satisfy the join — the index admits
    only matmul-histogram groups (``_matmul_sig``)."""
    g32 = goldens_by_name("square.512")[0].emmy_us
    knobs, latency = _row(512 * 512, 512, fp32=True, h_opt=3, latency=g32 * 0.1)
    knobs.pop("S_pw_multiply")  # no product feeding the reduce → not a matmul body
    assert "square.512" not in diagnostics.golden_deploy_perf(_FakePrior([(knobs, latency)]))
