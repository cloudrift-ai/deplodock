"""Schema + invariants for the golden matmul config set.

These are pure-data checks (no GPU): the YAML records under ``goldens/`` load
into :class:`MatmulGoldenConfig` instances, the derived ``ratio`` / ``golden``
properties stay consistent, ``matmul_snippet`` / ``repro_command`` render the
canonical form, and the goldens-to-DB materializer is idempotent. The actual
latencies are produced by ``scripts/find_golden_configs.py`` on a CUDA device.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.search import SearchDB
from deplodock.publish.goldens import (
    GoldenConfig,
    MatmulGoldenConfig,
    load_goldens,
    matmul_snippet,
)
from deplodock.publish.goldens_to_db import load_goldens_into


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


def test_yaml_round_trip(tmp_path):
    from deplodock.publish.goldens import dump_goldens

    src = [
        MatmulGoldenConfig(name="t.1", M=64, N=64, K=64, knobs={"BM": 8}, deplodock_us=10.0, cublas_us=9.0),
        MatmulGoldenConfig(name="t.2", M=128, N=64, K=32, dtype="fp16", knobs={"BM": 16}, deplodock_us=5.0, cublas_us=4.5),
    ]
    dump_goldens(src, tmp_path)
    loaded = load_goldens(tmp_path)
    assert {c.name for c in loaded} == {"t.1", "t.2"}
    by_name = {c.name: c for c in loaded}
    assert by_name["t.2"].dtype == "fp16"
    assert by_name["t.1"].knobs == {"BM": 8}


def test_golden_configs_set_is_well_formed():
    configs = load_goldens()
    assert configs, "expected at least one golden config under goldens/"
    for c in configs:
        assert isinstance(c, MatmulGoldenConfig), c.name
        assert c.M > 0 and c.N > 0 and c.K > 0, c.name
        assert c.deplodock_us > 0 and c.cublas_us > 0, c.name
        assert c.ratio >= 0.0, c.name
        assert c.golden == (c.ratio >= 0.95), c.name
        assert c.knobs, f"{c.name} has no recorded knobs"


def test_load_goldens_into_db_is_idempotent():
    db = SearchDB()
    n1 = load_goldens_into(db)
    n2 = load_goldens_into(db)
    assert n1 == n2 > 0
    rows = list(db.iter_goldens())
    assert len(rows) == n1


def test_load_goldens_into_db_filters():
    db = SearchDB()
    load_goldens_into(db)
    matmul_rows = list(db.iter_goldens(kind="matmul"))
    assert all(r["kind"] == "matmul" for r in matmul_rows)
    # Every payload carries the M/N/K shape (matmul-specific) so consumers can query without re-loading YAML.
    for r in matmul_rows:
        assert {"M", "N", "K", "dtype"} <= set(r["payload"])


def test_attach_published_round_trip(tmp_path):
    pub_path = tmp_path / "pub.db"
    pub = SearchDB(path=pub_path)
    load_goldens_into(pub)
    pub.close()

    local = SearchDB()
    local.attach_published(pub_path, alias="pub")
    cur = local._conn.execute("SELECT COUNT(*) FROM pub.golden")  # noqa: SLF001
    assert cur.fetchone()[0] > 0
    # Idempotent: re-attach is a no-op.
    local.attach_published(pub_path, alias="pub")
