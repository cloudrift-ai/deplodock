"""Schema + invariants for the golden matmul config set.

These are pure-data checks (no GPU): the records load, the derived ``ratio`` /
``golden`` properties stay consistent, and ``matmul_snippet`` / ``repro_command``
render the canonical form. The actual latencies are produced by
``scripts/find_golden_configs.py`` on a CUDA device.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.search.golden import (
    GOLDEN_CONFIGS,
    GoldenConfig,
    MatmulGoldenConfig,
    matmul_snippet,
)


def test_matmul_snippet_fp32_has_no_dtype_kwarg():
    assert matmul_snippet(2048, 2048, 2048) == "torch.matmul(torch.randn(2048,2048), torch.randn(2048,2048))"
    # Non-square: lhs is (M,K), rhs is (K,N).
    assert matmul_snippet(32, 3072, 1024) == "torch.matmul(torch.randn(32,1024), torch.randn(1024,3072))"


def test_matmul_snippet_typed():
    assert "dtype=torch.float16" in matmul_snippet(128, 128, 128, "fp16")


@pytest.mark.parametrize(
    ("deplodock_us", "cublas_us", "ratio", "golden"),
    [(100.0, 99.0, 0.99, True), (100.0, 95.0, 0.95, True), (100.0, 80.0, 0.80, False), (0.0, 99.0, 0.0, False)],
)
def test_ratio_and_golden_derive(deplodock_us, cublas_us, ratio, golden):
    c = GoldenConfig(name="t", deplodock_us=deplodock_us, cublas_us=cublas_us)
    assert c.ratio == pytest.approx(ratio)
    assert c.golden is golden


def test_repro_command_round_trips_knobs_and_snippet():
    c = MatmulGoldenConfig(name="square.2048", M=2048, N=2048, K=2048, knobs={"BM": 8, "BN": 32, "TMA": 1, "STAGE": "11"})
    cmd = c.repro_command()
    assert 'DEPLODOCK_KNOBS="BM=8,BN=32,TMA=1,STAGE=11"' in cmd
    assert c.snippet() in cmd
    assert "--ir cuda" in cmd


def test_golden_configs_set_is_well_formed():
    for c in GOLDEN_CONFIGS:
        assert isinstance(c, MatmulGoldenConfig), c.name
        assert c.M > 0 and c.N > 0 and c.K > 0, c.name
        assert c.deplodock_us > 0 and c.cublas_us > 0, c.name
        assert c.ratio >= 0.0, c.name
        assert c.golden == (c.ratio >= 0.95), c.name
        assert c.knobs, f"{c.name} has no recorded knobs"
