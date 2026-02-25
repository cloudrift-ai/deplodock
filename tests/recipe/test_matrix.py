"""Unit tests for matrix expansion logic."""

import pytest

from deplodock.recipe.matrix import (
    PARAM_ABBREVIATIONS,
    build_override,
    dot_to_nested,
    expand_matrix_entry,
    matrix_label,
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


# ── expand_matrix_entry ────────────────────────────────────────────


def test_expand_all_scalars():
    entry = {"deploy.gpu": "NVIDIA RTX 5090", "deploy.gpu_count": 1}
    result = expand_matrix_entry(entry)
    assert result == [{"deploy.gpu": "NVIDIA RTX 5090", "deploy.gpu_count": 1}]


def test_expand_single_list():
    entry = {"deploy.gpu": "NVIDIA RTX 5090", "benchmark.max_concurrency": [1, 2, 4]}
    result = expand_matrix_entry(entry)
    assert len(result) == 3
    assert result[0] == {"deploy.gpu": "NVIDIA RTX 5090", "benchmark.max_concurrency": 1}
    assert result[1] == {"deploy.gpu": "NVIDIA RTX 5090", "benchmark.max_concurrency": 2}
    assert result[2] == {"deploy.gpu": "NVIDIA RTX 5090", "benchmark.max_concurrency": 4}


def test_expand_multiple_lists_zipped():
    entry = {
        "deploy.gpu": "NVIDIA RTX 5090",
        "engine.llm.max_concurrent_requests": [128, 256],
        "benchmark.max_concurrency": [128, 256],
    }
    result = expand_matrix_entry(entry)
    assert len(result) == 2
    assert result[0]["engine.llm.max_concurrent_requests"] == 128
    assert result[0]["benchmark.max_concurrency"] == 128
    assert result[1]["engine.llm.max_concurrent_requests"] == 256
    assert result[1]["benchmark.max_concurrency"] == 256


def test_expand_mismatched_list_lengths_raises():
    entry = {
        "benchmark.max_concurrency": [1, 2, 4],
        "benchmark.num_prompts": [100, 200],
    }
    with pytest.raises(ValueError, match="same length"):
        expand_matrix_entry(entry)


# ── matrix_label ───────────────────────────────────────────────────


def test_matrix_label_no_variables():
    combo = {"deploy.gpu": "NVIDIA RTX 5090", "deploy.gpu_count": 1}
    assert matrix_label(combo, set()) == ""


def test_matrix_label_single_variable():
    combo = {"deploy.gpu": "NVIDIA RTX 5090", "benchmark.max_concurrency": 128}
    label = matrix_label(combo, {"benchmark.max_concurrency"})
    assert label == "c128"


def test_matrix_label_multiple_variables():
    combo = {
        "deploy.gpu": "NVIDIA RTX 5090",
        "engine.llm.max_concurrent_requests": 256,
        "benchmark.max_concurrency": 128,
    }
    label = matrix_label(combo, {"engine.llm.max_concurrent_requests", "benchmark.max_concurrency"})
    assert "c128" in label
    assert "mcr256" in label


def test_matrix_label_unknown_key_uses_last_segment():
    combo = {"engine.llm.some_new_param": 42}
    label = matrix_label(combo, {"engine.llm.some_new_param"})
    assert label == "some_new_param42"


def test_param_abbreviations_coverage():
    """Known params all have abbreviations."""
    assert "max_concurrency" in PARAM_ABBREVIATIONS
    assert "num_prompts" in PARAM_ABBREVIATIONS
    assert "random_input_len" in PARAM_ABBREVIATIONS
    assert "random_output_len" in PARAM_ABBREVIATIONS
    assert "max_concurrent_requests" in PARAM_ABBREVIATIONS
    assert "context_length" in PARAM_ABBREVIATIONS


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
