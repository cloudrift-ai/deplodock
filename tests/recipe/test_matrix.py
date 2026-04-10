"""Unit tests for matrix expansion logic."""

import pytest

from deplodock.recipe.matrix import (
    _expand_cross,
    _expand_zip,
    build_override,
    dot_to_nested,
    expand_matrix,
    filter_combinations,
)

# ── dot_to_nested ──────────────────────────────────────────────────


def test_dot_to_nested_single_level():
    result = dot_to_nested("gpu", "NVIDIA RTX 5090")
    assert result == {"gpu": "NVIDIA RTX 5090"}


def test_dot_to_nested_two_levels():
    result = dot_to_nested("deploy.gpu", "NVIDIA RTX 5090")
    assert result == {"deploy": {"gpu": "NVIDIA RTX 5090"}}


def test_dot_to_nested_three_levels():
    result = dot_to_nested("engine.llm.max_concurrent_requests", 256)
    assert result == {"engine": {"llm": {"max_concurrent_requests": 256}}}


# ── _expand_cross ─────────────────────────────────────────────────


def test_cross_all_scalars():
    result = _expand_cross({"a": 1, "b": 2})
    assert result == [{"a": 1, "b": 2}]


def test_cross_one_list():
    result = _expand_cross({"a": 1, "b": [10, 20]})
    assert result == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


def test_cross_two_lists():
    result = _expand_cross({"a": [1, 2], "b": [10, 20]})
    assert len(result) == 4
    assert {"a": 1, "b": 10} in result
    assert {"a": 1, "b": 20} in result
    assert {"a": 2, "b": 10} in result
    assert {"a": 2, "b": 20} in result


def test_cross_with_nested_zip():
    """zip inside cross bundles its lists into one compound axis."""
    result = _expand_cross(
        {
            "gpu": ["A", "B"],
            "zip": {
                "sizes": ["small", "large"],
                "iters": [10, 30],
            },
        }
    )
    assert len(result) == 4  # 2 gpus × 2 zip combos
    assert {"gpu": "A", "sizes": "small", "iters": 10} in result
    assert {"gpu": "A", "sizes": "large", "iters": 30} in result
    assert {"gpu": "B", "sizes": "small", "iters": 10} in result
    assert {"gpu": "B", "sizes": "large", "iters": 30} in result


def test_cross_empty_list_produces_nothing():
    result = _expand_cross({"a": 1, "b": []})
    assert result == []


def test_cross_single_element_list():
    result = _expand_cross({"a": [1], "b": [10, 20]})
    assert len(result) == 2
    assert result == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


# ── _expand_zip ───────────────────────────────────────────────────


def test_zip_all_scalars():
    result = _expand_zip({"a": 1, "b": 2})
    assert result == [{"a": 1, "b": 2}]


def test_zip_single_list():
    result = _expand_zip({"a": "x", "b": [1, 2, 3]})
    assert len(result) == 3
    assert result[0] == {"a": "x", "b": 1}
    assert result[2] == {"a": "x", "b": 3}


def test_zip_multiple_lists():
    result = _expand_zip(
        {
            "deploy.gpu": "RTX",
            "mcr": [128, 256],
            "mc": [128, 256],
        }
    )
    assert len(result) == 2
    assert result[0] == {"deploy.gpu": "RTX", "mcr": 128, "mc": 128}
    assert result[1] == {"deploy.gpu": "RTX", "mcr": 256, "mc": 256}


def test_zip_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        _expand_zip({"a": [1, 2, 3], "b": [10, 20]})


def test_zip_with_nested_cross_mismatched_raises():
    """cross inside zip that produces different length raises."""
    with pytest.raises(ValueError, match="same length"):
        _expand_zip(
            {
                "x": [1, 2],
                "cross": {"a": ["p", "q"], "b": ["r", "s"]},
            }
        )


def test_zip_with_nested_cross():
    """cross inside zip becomes one zip axis."""
    result = _expand_zip(
        {
            "x": [1, 2],
            "cross": {"a": ["p", "q"]},
        }
    )
    assert len(result) == 2
    assert result[0] == {"x": 1, "a": "p"}
    assert result[1] == {"x": 2, "a": "q"}


# ── expand_matrix (entry point) ──────────────────────────────────


def test_expand_matrix_with_cross_key():
    matrices = {
        "cross": {
            "a": 1,
            "b": [10, 20],
        },
    }
    result = expand_matrix(matrices)
    assert len(result) == 2
    assert result == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


def test_expand_matrix_with_zip_key():
    matrices = {
        "zip": {
            "a": "x",
            "b": [1, 2],
            "c": [3, 4],
        },
    }
    result = expand_matrix(matrices)
    assert len(result) == 2
    assert result[0] == {"a": "x", "b": 1, "c": 3}


def test_expand_matrix_implicit_zip():
    """Plain dict without cross/zip key → implicit zip."""
    matrices = {
        "deploy.gpu": "RTX 5090",
        "deploy.gpu_count": 1,
    }
    result = expand_matrix(matrices)
    assert result == [{"deploy.gpu": "RTX 5090", "deploy.gpu_count": 1}]


def test_expand_matrix_legacy_list():
    """List of dicts → legacy experiment format, each entry is implicit zip."""
    matrices = [
        {"deploy.gpu": "RTX 5090", "deploy.gpu_count": 1, "mcr": [8, 16]},
        {"deploy.gpu": "H200", "deploy.gpu_count": 1},
    ]
    result = expand_matrix(matrices)
    assert len(result) == 3  # 2 from first entry + 1 from second
    assert result[0] == {"deploy.gpu": "RTX 5090", "deploy.gpu_count": 1, "mcr": 8}
    assert result[1] == {"deploy.gpu": "RTX 5090", "deploy.gpu_count": 1, "mcr": 16}
    assert result[2] == {"deploy.gpu": "H200", "deploy.gpu_count": 1}


def test_expand_matrix_invalid_type_raises():
    with pytest.raises(TypeError, match="dict or list"):
        expand_matrix("not a dict")


# ── filter_combinations ──────────────────────────────────────────


def test_filter_exact_match():
    combos = [
        {"deploy.gpu": "RTX 5090", "x": 1},
        {"deploy.gpu": "H200", "x": 2},
    ]
    result = filter_combinations(combos, [("deploy.gpu", "RTX 5090")])
    assert len(result) == 1
    assert result[0]["deploy.gpu"] == "RTX 5090"


def test_filter_glob_pattern():
    combos = [
        {"deploy.gpu": "NVIDIA GeForce RTX 5090", "x": 1},
        {"deploy.gpu": "NVIDIA H200 141GB", "x": 2},
    ]
    result = filter_combinations(combos, [("deploy.gpu", "*5090*")])
    assert len(result) == 1
    assert result[0]["x"] == 1


def test_filter_multiple_and():
    combos = [
        {"gpu": "A", "size": "1024"},
        {"gpu": "A", "size": "2048"},
        {"gpu": "B", "size": "1024"},
    ]
    result = filter_combinations(combos, [("gpu", "A"), ("size", "1024")])
    assert len(result) == 1
    assert result[0] == {"gpu": "A", "size": "1024"}


def test_filter_no_match():
    combos = [{"gpu": "A"}, {"gpu": "B"}]
    result = filter_combinations(combos, [("gpu", "C")])
    assert result == []


def test_filter_missing_key():
    """Missing key defaults to empty string — only matches empty-string patterns."""
    combos = [{"gpu": "A"}, {"gpu": "B"}]
    result = filter_combinations(combos, [("nonexistent", "something")])
    assert result == []


def test_filter_numeric_value():
    combos = [{"x": 1}, {"x": 2}]
    result = filter_combinations(combos, [("x", "1")])
    assert len(result) == 1


# ── Integration: sgemm cross-product recipe ──────────────────────


def test_sgemm_cross_product():
    """The sgemm recipe cross-product should produce 6 combinations (3 GPUs × 2 configs)."""
    matrices = {
        "cross": {
            "deploy.gpu_count": 1,
            "deploy.driver_version": "595.58.03",
            "deploy.cuda_version": "13.2",
            "strategy": "adaptive",
            "git_rev": "feature/mini-gpu-compiler@local",
            "deploy.gpu": [
                "NVIDIA GeForce RTX 5090",
                "NVIDIA H200 141GB",
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            ],
            "zip": {
                "sizes": ["1024,2048,4096,8192,16384", "256,512,1024,2048,4096,8192"],
                "batches": ["1", "4,8,16"],
                "iterations": [30, 20],
            },
        },
    }
    result = expand_matrix(matrices)
    assert len(result) == 6

    # All should have the broadcast scalars
    for combo in result:
        assert combo["deploy.gpu_count"] == 1
        assert combo["deploy.driver_version"] == "595.58.03"
        assert combo["strategy"] == "adaptive"

    # Check specific combos
    rtx5090_single = [c for c in result if c["deploy.gpu"] == "NVIDIA GeForce RTX 5090" and c["batches"] == "1"]
    assert len(rtx5090_single) == 1
    assert rtx5090_single[0]["sizes"] == "1024,2048,4096,8192,16384"
    assert rtx5090_single[0]["iterations"] == 30

    h200_batched = [c for c in result if c["deploy.gpu"] == "NVIDIA H200 141GB" and c["batches"] == "4,8,16"]
    assert len(h200_batched) == 1
    assert h200_batched[0]["sizes"] == "256,512,1024,2048,4096,8192"
    assert h200_batched[0]["iterations"] == 20


# ── build_override ─────────────────────────────────────────────────


def test_build_override_simple():
    combo = {"deploy.gpu": "NVIDIA RTX 5090", "deploy.gpu_count": 1}
    result = build_override(combo)
    assert result == {"deploy": {"gpu": "NVIDIA RTX 5090", "gpu_count": 1}}


def test_build_override_deep():
    combo = {
        "deploy.gpu": "NVIDIA RTX 5090",
        "engine.llm.max_concurrent_requests": 256,
        "benchmark.max_concurrency": 128,
    }
    result = build_override(combo)
    assert result["deploy"]["gpu"] == "NVIDIA RTX 5090"
    assert result["engine"]["llm"]["max_concurrent_requests"] == 256
    assert result["benchmark"]["max_concurrency"] == 128


def test_build_override_merges_same_prefix():
    combo = {
        "engine.llm.max_concurrent_requests": 256,
        "engine.llm.context_length": 8192,
    }
    result = build_override(combo)
    assert result == {"engine": {"llm": {"max_concurrent_requests": 256, "context_length": 8192}}}
