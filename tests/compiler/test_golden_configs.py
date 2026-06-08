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
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
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
        # Every regime: matmul (M,N,K), reduce (M,K), pointwise (M,N).
        assert isinstance(c, (MatmulGoldenConfig, ReduceGoldenConfig, PointwiseGoldenConfig)), c.name
        if isinstance(c, MatmulGoldenConfig):
            assert c.M > 0 and c.N > 0 and c.K > 0, c.name
        elif isinstance(c, ReduceGoldenConfig):
            assert c.M > 0 and c.K > 0, c.name
            assert c.knobs.get("BR", 1) > 1, f"{c.name} reduce golden must be cooperative (BR>1)"
        else:
            assert c.M > 0 and c.N > 0, c.name
        assert c.deplodock_us > 0 and c.cublas_us > 0, c.name
        assert c.ratio >= 0.0, c.name
        assert c.golden == (c.ratio >= 0.95), c.name
        assert c.knobs, f"{c.name} has no recorded knobs"
        assert c.snippet(), c.name


def _dup(knobs, us):
    return MatmulGoldenConfig(name="dup.512", M=512, N=512, K=512, knobs=knobs, deplodock_us=us, cublas_us=14.0)


def test_goldens_by_name_returns_every_config_under_a_name(monkeypatch):
    """A name may carry several configs (e.g. a newly found faster variant beside
    the old one); ``goldens_by_name`` returns them all, empty for an unknown name."""
    from deplodock.compiler.pipeline.search import golden as gmod

    a, b = _dup({"BM": 8}, 12.0), _dup({"BM": 16}, 14.0)
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [a, b])
    assert gmod.goldens_by_name("dup.512") == [a, b]
    assert gmod.goldens_by_name("nope") == []


def test_resolve_golden_arg_stashes_all_matches(monkeypatch):
    """``--golden NAME`` stashes *every* recorded config under NAME on
    ``args.golden_configs`` (all share the shape, so ``args.code`` is the snippet
    of the first); no ``--golden`` leaves an empty list."""
    from argparse import Namespace

    from deplodock.commands import compile as cmod
    from deplodock.compiler.pipeline.search import golden as gmod

    a, b = _dup({"BM": 8}, 12.0), _dup({"BM": 16}, 14.0)
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [a, b])

    args = Namespace(golden="dup.512", code=None, input=None, ir=None)
    cmod.resolve_golden_arg(args)
    assert args.golden_configs == [a, b]
    assert args.code == a.snippet()

    none = Namespace(golden=None, code=None, input=None, ir=None)
    cmod.resolve_golden_arg(none)
    assert none.golden_configs == []
