"""CLI tests for ``deplodock tune``'s unified target loop.

Golden vs non-golden differ in exactly one place — ``_tune_targets`` (dataset
construction). Everything else (`handle_tune`'s loop, `_tune_one`, the shared DB /
bench worker / prior) is common. Pure arg-wiring checks (no GPU): target
construction + guards, and the loop calling the shared `_tune_one` once per target.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deplodock.commands import tune


def _args(**over):
    base = dict(
        dataset=None,
        kernel=None,
        code=None,
        input=None,
        golden=None,
        output=None,
        ucb_c=1.4142,
        bench_timeout=20.0,
        seed=0,
        patience=None,
        explore_eps=None,
        nvcc_flags=None,
        dump_dir=None,
        quiet=False,
        verbose=0,
        bench=False,
        clean=False,
        bench_backends="eager,deplodock",
        warmup=10,
        iters=100,
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
        )
    ]


def test_golden_dataset_targets_dedup():
    targets = tune._tune_targets(_args(dataset="golden", kernel="square"))
    names = [t[0] for t in targets]
    assert len(names) >= 4
    assert names == list(dict.fromkeys(names))  # de-duplicated by shape name
    assert all(code.startswith("torch.matmul(") and inp is None for _name, code, inp in targets)


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
    monkeypatch.setattr(tune, "_tune_backend", lambda: object())
    monkeypatch.setattr(tune, "_bench_dump", lambda a: (None, None))
    monkeypatch.setattr(tune, "_clean_caches", lambda p: cleaned.append(p))
    monkeypatch.setattr(tune, "setup_pipeline_runtime", lambda a: None)
    monkeypatch.setattr(tune, "apply_nvcc_flags", lambda a, default: "-Xcicc -O1")
    monkeypatch.setattr(tune, "resolve_tune_db", lambda: "/tmp/golden_test.db")
    monkeypatch.setattr("deplodock.compiler.pipeline.search.SearchDB", lambda path: object())
    monkeypatch.setattr("deplodock.compiler.context.Context.probe", staticmethod(lambda: object()))

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
