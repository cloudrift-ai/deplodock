"""CLI tests for ``emmy tune``'s unified target loop.

Golden vs non-golden differ in exactly one place — ``_tune_targets`` (dataset
construction). Everything else (`handle_tune`'s loop, `_tune_one`, the shared DB /
bench worker / prior) is common. Pure arg-wiring checks (no GPU): target
construction + guards, and the loop calling the shared `_tune_one` once per target.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emmy.commands import tune


def _args(**over):
    base = dict(
        dataset=None,
        kernel=None,
        code=None,
        input=None,
        golden=None,
        dynamic=None,
        output=None,
        ucb_c=1.4142,
        seed=0,
        patience=None,
        explore_eps=None,
        nvcc_flags=None,
        dump_dir=None,
        quiet=False,
        verbose=0,
        bench=False,
        clean=False,
        bench_backends="eager,emmy",
        warmup=10,
        iters=100,
        gpus=None,
        devices=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --- _tune_targets: the one place golden / non-golden diverge ---------------


def test_single_code_target():
    targets = tune._tune_targets(_args(code="torch.matmul(torch.randn(8, 8), torch.randn(8, 8))"))
    assert targets == [
        (
            "torch.matmul(torch.randn(8, 8), torch.randn(8, 8))",
            "torch.matmul(torch.randn(8, 8), torch.randn(8, 8))",
            None,
            None,
        )
    ]


def test_single_code_target_keeps_cli_dynamic():
    """An ad-hoc --code target traces with the CLI ``--dynamic`` specs."""
    targets = tune._tune_targets(_args(code="torch.matmul(x, torch.randn(8, 8))", dynamic=["seq_len@x:0"]))
    assert [t[3] for t in targets] == [["seq_len@x:0"]]


def test_golden_dataset_targets_dedup():
    targets = tune._tune_targets(_args(dataset="golden", kernel="square"))
    names = [t[0] for t in targets]
    assert len(names) >= 4
    assert names == list(dict.fromkeys(names))  # de-duplicated by shape name
    assert all(code.startswith("torch.matmul(") and inp is None for _name, code, inp, _dyn in targets)


def _dyn_golden(name="square.512.dynM"):
    from emmy.compiler.pipeline.search.golden import MatmulGoldenConfig

    return MatmulGoldenConfig(
        name=name,
        M=512,
        N=512,
        K=512,
        knobs={"TILE": "n16x8/f2x2"},
        emmy_us=10.0,
        cublas_us=12.0,
        dynamic={"seq_len": {"input": "x0", "axis": 0}},
    )


def test_golden_dataset_target_carries_dynamic_spec(monkeypatch):
    """A dynamic golden expands to a target carrying its own ``--dynamic`` spec; a
    static golden's target carries ``None``."""
    from emmy.compiler.pipeline.search import golden as gmod

    static = gmod.MatmulGoldenConfig(name="square.512", M=512, N=512, K=512, knobs={"TILE": "n16x8/f2x2"}, emmy_us=9.0, cublas_us=14.0)
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [static, _dyn_golden()])
    targets = tune._tune_targets(_args(dataset="golden"))
    by_name = {name: dyn for name, _code, _inp, dyn in targets}
    assert by_name == {"square.512": None, "square.512.dynM": ["seq_len@x0:0"]}


def test_dataset_golden_rejects_cli_dynamic():
    with pytest.raises(SystemExit) as exc:
        tune._tune_targets(_args(dataset="golden", dynamic=["seq_len@x0:0"]))
    assert exc.value.code == 2


def test_db_source_rejected():
    with pytest.raises(SystemExit) as exc:
        tune._tune_targets(_args(dataset="db"))
    assert exc.value.code == 2


def test_dataset_golden_mutually_exclusive_with_code():
    with pytest.raises(SystemExit) as exc:
        tune._tune_targets(_args(dataset="golden", golden="square.512"))
    assert exc.value.code == 2


def test_unknown_kernel_filter_errors():
    with pytest.raises(SystemExit) as exc:
        tune._tune_targets(_args(dataset="golden", kernel="NO_SUCH_SHAPE"))
    assert exc.value.code == 2


# --- handle_tune loop (heavy deps stubbed) ----------------------------------


def _stub_runtime(monkeypatch):
    """Stub everything that would touch a GPU / disk; turn ``_exit_flushed`` into a
    ``SystemExit``. Returns ``(tuned_codes, cleaned)`` capture lists."""
    tuned_codes: list[str] = []
    cleaned: list[object] = []

    def fake_tune_one(a, **_kw):
        tuned_codes.append(a.code)
        return SimpleNamespace(best_reward=None, assembled=None), None

    monkeypatch.setattr(tune, "_tune_one", fake_tune_one)
    monkeypatch.setattr(tune, "_tune_backend", lambda device_id=None: object())
    monkeypatch.setattr(tune, "_bench_dump", lambda a: (None, None))
    monkeypatch.setattr(tune, "_clean_caches", lambda p: cleaned.append(p))
    monkeypatch.setattr(tune, "setup_pipeline_runtime", lambda a: None)
    monkeypatch.setattr(tune, "apply_nvcc_flags", lambda a, default: "-Xcicc -O1")
    monkeypatch.setattr(tune, "resolve_tune_db", lambda: "/tmp/golden_test.db")
    monkeypatch.setattr("emmy.compiler.pipeline.search.SearchDB", lambda path: object())
    monkeypatch.setattr("emmy.compiler.context.Context.probe", staticmethod(lambda: object()))

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(tune, "_exit_flushed", fake_exit)
    return tuned_codes, cleaned


def test_loop_one_tune_per_golden_shape(monkeypatch):
    tuned_codes, cleaned = _stub_runtime(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(dataset="golden", kernel="square", clean=True))
    assert exc.value.code == 0
    assert len(tuned_codes) >= 4
    assert all(c.startswith("torch.matmul(") for c in tuned_codes)
    assert len(cleaned) == 1  # --clean wipes once up front, not per shape


def test_single_shape_uses_same_loop(monkeypatch):
    tuned_codes, cleaned = _stub_runtime(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(code="torch.matmul(torch.randn(8, 8), torch.randn(8, 8))"))
    assert exc.value.code == 0
    assert tuned_codes == ["torch.matmul(torch.randn(8, 8), torch.randn(8, 8))"]
    assert cleaned == []


def test_loop_sets_dynamic_per_target(monkeypatch):
    """The loop threads each target's dynamic spec onto ``args.dynamic`` before
    ``_tune_one`` (which traces via ``load_or_trace``), so a dynamic golden in the
    sweep traces symbolically and its static neighbors don't inherit the spec."""
    from emmy.compiler.pipeline.search import golden as gmod

    _stub_runtime(monkeypatch)
    seen: list[tuple[str, object]] = []

    def capture(a, **_kw):
        seen.append((a.code, a.dynamic))
        return SimpleNamespace(best_reward=None, assembled=None), None

    monkeypatch.setattr(tune, "_tune_one", capture)
    static = gmod.MatmulGoldenConfig(name="square.512", M=512, N=512, K=512, knobs={"TILE": "n16x8/f2x2"}, emmy_us=9.0, cublas_us=14.0)
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [static, _dyn_golden()])
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(dataset="golden"))
    assert exc.value.code == 0
    assert sorted(dyn is not None for _code, dyn in seen) == [False, True]
    assert [dyn for _code, dyn in seen if dyn] == [["seq_len@x0:0"]]


def test_runtime_error_aborts_sweep(monkeypatch):
    """A saturated-queue RuntimeError mid-sweep leaves the parent stream dirty, so
    the loop aborts (exit 1) rather than press on into an unreliable context."""
    _stub_runtime(monkeypatch)
    calls = {"n": 0}

    def boom(a, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bench watchdog couldn't bail")
        return SimpleNamespace(best_reward=None, assembled=None), None

    monkeypatch.setattr(tune, "_tune_one", boom)
    with pytest.raises(SystemExit) as exc:
        tune.handle_tune(_args(dataset="golden", kernel="square"))
    assert exc.value.code == 1
    assert calls["n"] == 2  # stopped at the failing shape
