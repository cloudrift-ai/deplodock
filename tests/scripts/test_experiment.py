"""Tests for scripts/experiment.py — the pure record assembly (no GPU) — plus the
search-speed fields the experiment tracker reads off the two-level result objects."""

import sys
from pathlib import Path

import pytest

from deplodock.compiler.pipeline.search.two_level import InnerReward, OpResult


@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts_dir)
    yield
    sys.path.remove(scripts_dir)


# ── search-speed fields are carried + aggregated ────────────────────────────────────


def test_op_result_carries_search_speed_fields():
    r = OpResult(name="k_rms_norm", op_key="abc", best_us=4.2, multiplicity=24, benches=30, benches_to_best=12, stop_reason="patience")
    assert (r.benches, r.benches_to_best, r.stop_reason) == (30, 12, "patience")


def test_op_result_search_speed_defaults_are_inert():
    # Existing callers that don't pass the new fields stay valid (additive defaults).
    r = OpResult(name="k", op_key="x", best_us=1.0)
    assert r.benches == 0 and r.benches_to_best is None and r.stop_reason is None


def test_inner_reward_total_benches_field():
    rew = InnerReward(total_us=10.0, ok=True, per_op=[], total_benches=42)
    assert rew.total_benches == 42


# ── build_record schema (pure) ──────────────────────────────────────────────────────


def test_build_record_schema_and_aggregation():
    import experiment

    per_shape = [
        {
            "name": "square.512",
            "wall_s": 3.0,
            "total_benches": 30,
            "per_op": [{"name": "k", "benches": 30, "benches_to_best": 12, "stop_reason": "patience", "best_us": 4.2}],
        },
        {"name": "square.1024", "wall_s": 5.0, "total_benches": 20, "per_op": []},
    ]
    quality = {
        "reachability": {"mean_ratio": 1.1, "median_ratio": 1.0, "worst_ratio": 1.4, "n_ops": 7},
        "deploy_perf": {"median_vs_golden": 0.98, "n_better_than_golden": 5, "n_shapes": 8, "per_shape": {}},
        "calibration_rho": 0.83,
        "golden_coverage": [8, 30],
    }
    rec = experiment.build_record(
        name="baseline",
        config_dict={"patience": 50},
        per_shape=per_shape,
        quality=quality,
        timestamp="2026-06-19T10:00:00",
        git_commit="deadbee",
        gpu="RTX 4070 Ti",
    )
    assert rec["name"] == "baseline" and rec["git_commit"] == "deadbee"
    # Search-speed totals are summed across shapes.
    assert rec["search_speed"]["total_benches"] == 50
    assert rec["search_speed"]["total_wall_s"] == 8.0
    assert rec["search_speed"]["n_shapes"] == 2
    assert rec["kernel_quality"]["reachability"]["median_ratio"] == 1.0
    assert rec["failures"] == []  # defaults to empty when none passed


def test_build_record_carries_failures():
    import experiment

    rec = experiment.build_record(
        name="baseline",
        config_dict={},
        per_shape=[],
        quality={},
        timestamp="t",
        git_commit="g",
        gpu="gpu",
        failures=[{"name": "qwen3_06b.gate_up_proj.s32", "error": "LoweringError: smem over budget"}],
    )
    assert len(rec["failures"]) == 1 and rec["failures"][0]["name"] == "qwen3_06b.gate_up_proj.s32"


def test_slug_sanitizes_names():
    import experiment

    assert experiment._slug("low pat / fp16!") == "low-pat-fp16"
    assert experiment._slug("///") == "exp"
