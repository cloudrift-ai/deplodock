"""Schema + invariants for the golden matmul config set.

These are pure-data checks (no GPU): the records load from the per-GPU YAML files,
the derived ``ratio`` / ``golden`` properties stay consistent, and ``matmul_snippet``
/ ``repro_command`` render the canonical form. The actual latencies are measured on
a CUDA device via ``deplodock tune --golden NAME --bench`` and recorded into the YAML.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.search.golden import (
    _GOLDENS_DIR,
    _KERNEL_CLASSES,
    GOLDEN_CONFIGS,
    GoldenConfig,
    MatmulGoldenConfig,
    PointwiseGoldenConfig,
    ReduceGoldenConfig,
    _load_goldens,
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


def test_goldens_load_from_yaml():
    """The per-GPU YAML files are the source of truth: at least one file exists,
    every config carries a known ``kernel`` discriminator, and the loaded set is
    non-empty and identical to the import-time :data:`GOLDEN_CONFIGS`."""
    import yaml

    files = sorted(_GOLDENS_DIR.glob("*.yaml"))
    assert files, f"no golden YAML files under {_GOLDENS_DIR}"
    for path in files:
        doc = yaml.safe_load(path.read_text())
        assert isinstance(doc["gpu_name"], str) and len(doc["compute_cap"]) == 2, path.name
        for c in doc["configs"]:
            assert c["kernel"] in _KERNEL_CLASSES, f"{path.name}: unknown kernel {c.get('kernel')!r}"

    loaded = _load_goldens()
    assert loaded  # non-empty
    assert len(loaded) == len(GOLDEN_CONFIGS)


# --- dynamic (symbolic-axis) matmul goldens ----------------------------------


def _dyn(dynamic, M=512):
    return MatmulGoldenConfig(name="square.512.dynM", M=M, N=512, K=512, dynamic=dynamic)


def test_dynamic_golden_specs_snippet_and_repro():
    """A dynamic golden keeps the hint-shaped snippet (M doubles as the hint) and
    additionally carries the ``--dynamic NAME@INPUT:AXIS`` spec for the tracer;
    ``repro_command`` includes the flag so the repro rebuilds the masked-tile
    kernel, not the static twin."""
    c = _dyn({"seq_len": {"input": "x0", "axis": 0}})
    assert c.dynamic_specs() == ["seq_len@x0:0"]
    assert c.snippet() == matmul_snippet(512, 512, 512)  # unchanged hint-shaped code
    assert "--dynamic seq_len@x0:0" in c.repro_command()


def test_static_golden_has_no_dynamic_specs():
    c = MatmulGoldenConfig(name="square.512", M=512, N=512, K=512)
    assert c.dynamic_specs() == []
    assert "--dynamic" not in c.repro_command()


@pytest.mark.parametrize(
    "dynamic",
    [
        {},  # empty mapping
        {"seq_len": {"input": "x0"}},  # missing axis
        {"seq_len": {"input": "x0", "axis": "0"}},  # axis not an int
        {"seq_len": {"input": "x0", "axis": True}},  # bool is not an axis
        {"seq_len": {"input": "x0", "axis": -1}},  # negative axis
        {"seq_len": {"input": "", "axis": 0}},  # empty input name
        {"": {"input": "x0", "axis": 0}},  # empty NAME
        {"seq_len": {"input": "x0", "axis": 1}},  # K axis — lowering not supported yet
        {"seq_len": {"input": "x1", "axis": 1}},  # N axis — lowering not supported yet
    ],
)
def test_dynamic_golden_schema_rejects_malformed(dynamic):
    with pytest.raises(ValueError):
        _dyn(dynamic)


def test_dynamic_golden_hint_must_be_positive():
    with pytest.raises(ValueError, match="hint"):
        _dyn({"seq_len": {"input": "x0", "axis": 0}}, M=0)


def test_dynamic_golden_hint_must_equal_default_seq_hint():
    """The pipeline tiles/benches a symbolic axis at the global ``DEFAULT_SEQ_HINT``,
    not the traced M — an M=1024 dynamic golden would silently be measured at 512
    and duplicate the (N, K, hint-512) shape. Rejected until per-Dim hints exist."""
    from deplodock.compiler.dim import DEFAULT_SEQ_HINT

    with pytest.raises(ValueError, match="DEFAULT_SEQ_HINT"):
        _dyn({"seq_len": {"input": "x0", "axis": 0}}, M=1024)
    _dyn({"seq_len": {"input": "x0", "axis": 0}}, M=DEFAULT_SEQ_HINT)  # the only valid hint today


def test_dynamic_is_matmul_only():
    """``dynamic`` is a MatmulGoldenConfig field only — reduce / pointwise goldens
    reject it at construction (the symbolic lowering for those shapes is the split,
    not a masked golden)."""
    with pytest.raises(TypeError):
        ReduceGoldenConfig(name="r", M=512, K=512, dynamic={"seq_len": {"input": "x0", "axis": 0}})


def test_sample_from_golden_carries_dynamic_specs():
    from deplodock.compiler.pipeline.search.data import Sample

    dyn = Sample.from_golden(_dyn({"seq_len": {"input": "x0", "axis": 0}}))
    assert dyn.dynamic == ("seq_len@x0:0",)
    static = Sample.from_golden(MatmulGoldenConfig(name="square.512", M=512, N=512, K=512))
    assert static.dynamic is None


def test_shape_key_carries_dynamic_flag():
    """``shape_key()`` is the single golden-side join key: static keys are the
    full ``M*N`` product; a dynamic config's key excludes the symbolic M (mirroring
    the 992 stamp) and sets ``is_dyn``, so the twins never share a key."""
    static = MatmulGoldenConfig(name="s", M=512, N=256, K=64)
    assert (static.shape_key().free_prod, static.shape_key().is_dyn) == (512 * 256, False)
    dyn = _dyn({"seq_len": {"input": "x0", "axis": 0}})  # M=N=K=512
    assert (dyn.shape_key().free_prod, dyn.shape_key().is_dyn) == (512, True)
    assert dyn.shape_key() != MatmulGoldenConfig(name="t", M=512, N=512, K=512).shape_key()


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
    ``args.golden_configs`` as :class:`Sample`s (all share the shape, so ``args.code``
    is the snippet of the first); no ``--golden`` leaves an empty list."""
    from argparse import Namespace

    from deplodock.compiler.pipeline.search import golden as gmod

    a, b = _dup({"BM": 8}, 12.0), _dup({"BM": 16}, 14.0)
    # Dataset.from_golden reads golden.GOLDEN_CONFIGS via a lazy import, so this patch is seen.
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [a, b])

    from deplodock.commands import compile as cmod

    args = Namespace(golden="dup.512", code=None, input=None, ir=None)
    cmod.resolve_golden_arg(args)
    assert [s.name for s in args.golden_configs] == ["dup.512", "dup.512"]
    assert [s.knobs for s in args.golden_configs] == [{"BM": 8}, {"BM": 16}]
    assert args.code == a.snippet()
    assert all(s.source == "golden" for s in args.golden_configs)

    none = Namespace(golden=None, code=None, input=None, ir=None)
    cmod.resolve_golden_arg(none)
    assert none.golden_configs == []


def test_resolve_golden_arg_applies_dynamic_spec(monkeypatch):
    """A dynamic golden's recorded spec lands on ``args.dynamic`` (so the trace goes
    symbolic); a CLI ``--dynamic`` next to ``--golden`` is rejected — the spec is
    part of the config, the same way ``--ir`` rejects it."""
    from argparse import Namespace

    from deplodock.commands import compile as cmod
    from deplodock.compiler.pipeline.search import golden as gmod

    dyn = MatmulGoldenConfig(
        name="dup.512.dynM",
        M=512,
        N=512,
        K=512,
        knobs={"BM": 8},
        deplodock_us=12.0,
        cublas_us=14.0,
        dynamic={"seq_len": {"input": "x0", "axis": 0}},
    )
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [dyn])

    args = Namespace(golden="dup.512.dynM", code=None, input=None, ir=None, dynamic=None)
    cmod.resolve_golden_arg(args)
    assert args.dynamic == ["seq_len@x0:0"]
    assert args.code == dyn.snippet()

    clash = Namespace(golden="dup.512.dynM", code=None, input=None, ir=None, dynamic=["seq_len@x0:0"])
    with pytest.raises(SystemExit) as exc:
        cmod.resolve_golden_arg(clash)
    assert exc.value.code == 2
