"""CLI tests for ``emmy compare`` — diffing two dump dirs' bench artifacts.

Each test writes the synthetic dump JSONs inline (the three artifacts the
command reads: ``60_bench_compare.json``, ``62_kernel_bench.json``,
``60_benchmark.json``), so the suite stays hermetic — no GPU, no real dumps.
"""

from __future__ import annotations

import json


def _dump(tmp_path, name: str, **files) -> str:
    d = tmp_path / name
    d.mkdir()
    for fname, payload in files.items():
        (d / fname.replace("__", ".")).write_text(json.dumps(payload))
    return str(d)


def _full_model(**backends):
    return {"warmup": 10, "iters": 100, "backends": {k: {"latency_us": v} for k, v in backends.items()}}


def _kernel_bench(rows):
    return [{"kernel": k, "label": k.rsplit("_", 1)[0], "captured": True, "backends": {"Emmy": us}} for k, us in rows]


def test_compare_full_model_and_kernels(run_cli, tmp_path):
    """Both sections render: per-backend full-model ratios and the per-kernel
    diff (exact-name match first, hash-moved kernels matched by base name and
    shown as ``a -> b``), plus a matched TOTAL row."""
    a = _dump(
        tmp_path,
        "a",
        **{
            "60_bench_compare__json": _full_model(**{"Eager PyTorch": 96.0, "Emmy": 193.0}),
            "62_kernel_bench__json": _kernel_bench([("k_mean_aaaaaa", 2.0), ("k_linear_reduce_111111", 29.0)]),
        },
    )
    b = _dump(
        tmp_path,
        "b",
        **{
            "60_bench_compare__json": _full_model(**{"Eager PyTorch": 95.0, "Emmy": 138.0}),
            # same hash for k_mean (exact match); k_linear_reduce re-tuned → new hash (base-name match)
            "62_kernel_bench__json": _kernel_bench([("k_mean_aaaaaa", 2.1), ("k_linear_reduce_222222", 8.0)]),
        },
    )
    rc, stdout, stderr = run_cli("compare", a, b)
    assert rc == 0, f"stderr: {stderr}"
    assert "Full model" in stdout
    assert "0.72x" in stdout  # Emmy 138/193
    assert "Per-kernel emmy -O3" in stdout
    assert "k_linear_reduce_111111 -> k_linear_reduce_222222" in stdout  # hash moved, base-matched
    assert "0.28x" in stdout  # 8/29
    assert "TOTAL (matched)" in stdout


def test_compare_kernel_set_change_listed(run_cli, tmp_path):
    """A kernel present on one side only is reported as a kernel-set change, not
    silently dropped."""
    a = _dump(tmp_path, "a", **{"62_kernel_bench__json": _kernel_bench([("k_fused_abc123", 60.0)])})
    b = _dump(
        tmp_path,
        "b",
        **{"62_kernel_bench__json": _kernel_bench([("k_fused_abc123", 30.0), ("k_xn_def456", 2.0)])},
    )
    rc, stdout, stderr = run_cli("compare", a, b)
    assert rc == 0, f"stderr: {stderr}"
    assert "only in B: k_xn_def456" in stdout
    assert "kernel-set change" in stdout


def test_compare_per_launch_fallback(run_cli, tmp_path):
    """Dumps without a per-kernel bench still diff via 60_benchmark.json's raw
    per-launch times."""

    def bench(times):
        return {
            "time_ms": sum(times),
            "min_ms": None,
            "max_ms": None,
            "num_launches": len(times),
            "per_launch": [{"idx": i, "kernel_name": f"k_op_{i}{'a' * 5}", "time_ms": t} for i, t in enumerate(times)],
        }

    a = _dump(tmp_path, "a", **{"60_benchmark__json": bench([0.010, 0.020])})
    b = _dump(tmp_path, "b", **{"60_benchmark__json": bench([0.005, 0.020])})
    rc, stdout, stderr = run_cli("compare", a, b)
    assert rc == 0, f"stderr: {stderr}"
    assert "Per-launch emmy" in stdout
    assert "0.50x" in stdout  # 5us / 10us


def test_compare_no_artifacts_errors(run_cli, tmp_path):
    a = _dump(tmp_path, "a")
    b = _dump(tmp_path, "b")
    rc, stdout, stderr = run_cli("compare", a, b)
    assert rc == 2
    assert "no comparable bench artifacts" in (stdout + stderr)


def test_compare_missing_dir_errors(run_cli, tmp_path):
    rc, stdout, stderr = run_cli("compare", str(tmp_path / "nope"), str(tmp_path))
    assert rc == 2
    assert "not a dump dir" in (stdout + stderr)
