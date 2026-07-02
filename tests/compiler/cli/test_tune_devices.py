"""CLI tests for ``emmy tune --gpus / --devices`` device resolution (no GPU).

``_resolve_devices`` maps the flags to a device-id list (``--devices`` wins) and,
for two or more devices, enforces homogeneity (one perf key per tune). The
homogeneity probe needs cupy, so it's exercised here by monkeypatching the
device-properties call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from emmy.commands import tune


def _args(*, gpus=None, devices=None):
    return SimpleNamespace(gpus=gpus, devices=devices)


def test_default_is_single_unpinned_slot() -> None:
    assert tune._resolve_devices(_args()) == [None]


def _device_count() -> int:
    """Physical GPU count, or 0 if cupy can't probe (absent / no driver)."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        return 0


def _gate_multi_gpu(max_ordinal: int) -> None:
    """Multi-device resolution runs the homogeneity probe, which queries each
    GPU. With cupy present but too few GPUs the probe ``sys.exit(2)``s, so skip
    unless the box has the ordinals (cupy absent → probe is a no-op, test runs)."""
    count = _device_count()
    if 0 < count <= max_ordinal:
        pytest.skip(f"needs GPU ordinal {max_ordinal} (homogeneity probe), have {count}")


def test_gpus_n_expands_to_range() -> None:
    # cupy absent → probe is a no-op; present → needs ordinals 0,1,2.
    _gate_multi_gpu(2)
    assert tune._resolve_devices(_args(gpus=3)) == [0, 1, 2]


def test_devices_list_wins_over_gpus() -> None:
    _gate_multi_gpu(5)
    assert tune._resolve_devices(_args(gpus=8, devices="0,2,5")) == [0, 2, 5]


def test_single_device_skips_homogeneity() -> None:
    assert tune._resolve_devices(_args(devices="1")) == [1]


def test_bad_devices_exits(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(devices="0,x,2"))
    assert e.value.code == 2


def test_gpus_below_one_exits() -> None:
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(gpus=0))
    assert e.value.code == 2


def test_heterogeneous_devices_rejected(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda d: {"major": 8 if d == 0 else 9, "minor": 0}))
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)
    with pytest.raises(SystemExit) as e:
        tune._resolve_devices(_args(devices="0,1"))
    assert e.value.code == 2


def test_homogeneous_devices_accepted(monkeypatch) -> None:
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda d: {"major": 9, "minor": 0})))
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)
    assert tune._resolve_devices(_args(devices="0,1,2")) == [0, 1, 2]
