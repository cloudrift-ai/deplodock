"""CLI tests for ``deplodock tune --dataset golden`` — the golden-set orchestrator.

Pure arg-wiring / orchestration checks (no GPU): the guards reject degenerate
sources, and ``_tune_golden_dataset`` fans out one ``tune --golden NAME``
subprocess per golden shape with ``--clean`` only on the first.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from deplodock.commands import tune


def _args(**over):
    base = dict(
        dataset="golden",
        kernel=None,
        code=None,
        input=None,
        golden=None,
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


def _capture(args):
    """Run the orchestrator with subprocess.call stubbed; return the captured cmds."""
    calls: list[list[str]] = []
    with mock.patch("subprocess.call", lambda cmd: calls.append(cmd) or 0), pytest.raises(SystemExit) as exc:
        tune._tune_golden_dataset(args)
    return calls, exc.value.code


def test_db_source_rejected():
    with pytest.raises(SystemExit) as exc:
        tune._tune_golden_dataset(_args(dataset="db"))
    assert exc.value.code == 2


def test_mutually_exclusive_with_golden_name():
    with pytest.raises(SystemExit) as exc:
        tune._tune_golden_dataset(_args(golden="square.512"))
    assert exc.value.code == 2


def test_unknown_kernel_filter_errors():
    with pytest.raises(SystemExit) as exc:
        tune._tune_golden_dataset(_args(kernel="NO_SUCH_SHAPE"))
    assert exc.value.code == 2


def test_fans_out_one_subprocess_per_shape_clean_first():
    calls, code = _capture(_args(kernel="square", clean=True))
    assert code == 0
    # --kernel square matches the fp32 + fp16 square shapes; each is one tune call.
    assert len(calls) >= 4
    names = [c[c.index("--golden") + 1] for c in calls]
    assert names == sorted(set(names), key=names.index)  # de-duplicated, order-preserving
    assert all("square" in n for n in names)
    # --clean only on the first shape.
    assert "--clean" in calls[0]
    assert all("--clean" not in c for c in calls[1:])
    # every call drives the CLI entry point for `tune --golden`.
    assert calls[0][:3] == [sys.executable, "-m", "deplodock.deplodock"]
    assert calls[0][3:5] == ["tune", "--golden"]


def test_passthrough_flags_forwarded():
    calls, _ = _capture(_args(kernel="square.512", patience=80, nvcc_flags="-Xcicc -O3", bench=True, verbose=2))
    cmd = calls[0]
    assert "--patience" in cmd and cmd[cmd.index("--patience") + 1] == "80"
    assert "--nvcc-flags" in cmd and cmd[cmd.index("--nvcc-flags") + 1] == "-Xcicc -O3"
    assert "--bench" in cmd and "--bench-backends" in cmd
    assert cmd.count("-v") == 2
