"""``diagnostics.golden_deploy_perf`` — the no-rebench ``vs gold`` column for
``deplodock eval prior``: per golden shape, the deployable (-O3) latency of the
prior's predicted-best reservoir config over the golden's recorded ``deplodock_us``.

The load-bearing case is dtype separation: an fp32 square and its ``.fp16`` twin
share (free-dim product, reduce extent), so the shape key MUST split on
``S_dtype_f32`` — keying on an mma marker (which the fp16 row may not carry) merges
them and steals the fp16 latency for the fp32 row.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.search.golden import goldens_by_name
from deplodock.compiler.pipeline.search.prior import diagnostics


class _FakePrior:
    """A prior with a hand-built reservoir; ``mean_score`` is constant (each group
    here has one -O3 row, so the argmin pick is unambiguous)."""

    def __init__(self, rows):
        self._dataset = rows  # list[(stamped_knobs, latency_us)]

    def mean_score(self, _feats):
        return 0.0


def _row(free_prod, reduce_max, *, fp32, h_opt, latency):
    knobs = {
        "H_opt": float(h_opt),
        "S_ext_free_prod": float(free_prod),
        "S_ext_reduce_max": float(reduce_max),
        ("S_dtype_f32" if fp32 else "S_dtype_f16"): 2.0,
        "BM": 8,
        "BN": 16,
    }
    return (knobs, latency)


def test_dtype_separation_and_o3_filter():
    g32 = goldens_by_name("square.512")[0].deplodock_us  # fp32 golden latency (-O3)
    g16 = goldens_by_name("square.512.fp16")[0].deplodock_us
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
    g32 = goldens_by_name("square.512")[0].deplodock_us
    prior = _FakePrior([_row(512 * 512, 512, fp32=True, h_opt=3, latency=g32)])
    assert set(diagnostics.golden_deploy_perf(prior, "square.512")) <= {"square.512"}
