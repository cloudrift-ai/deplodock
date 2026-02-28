"""Unit tests for the Variant class."""

from deplodock.planner.variant import Variant, _abbreviate, _compact_value

# ── _abbreviate ──────────────────────────────────────────────────


def test_abbreviate_single_word():
    assert _abbreviate("prompts") == "p"


def test_abbreviate_two_words():
    assert _abbreviate("num_prompts") == "np"


def test_abbreviate_three_words():
    assert _abbreviate("max_concurrent_requests") == "mcr"


def test_abbreviate_max_concurrency():
    assert _abbreviate("max_concurrency") == "mc"


def test_abbreviate_random_input_len():
    assert _abbreviate("random_input_len") == "ril"


def test_abbreviate_context_length():
    assert _abbreviate("context_length") == "cl"


# ── _compact_value ───────────────────────────────────────────────


def test_compact_value_plain_number():
    assert _compact_value(128) == "128"


def test_compact_value_docker_image():
    assert _compact_value("lmsysorg/sglang:latest") == "lms-sglang-latest"


def test_compact_value_cli_flags():
    assert _compact_value("--quantization moe_wna16") == "quant-moe-wna16"


def test_compact_value_preserves_version():
    assert _compact_value("v0.6.5") == "v0.6.5"


def test_compact_value_keeps_digit_segments():
    """Segments containing digits are kept intact."""
    assert _compact_value("foo-bar16-baz") == "f-bar16-b"


def test_compact_value_known_abbreviations():
    """Known words use their dictionary abbreviation."""
    assert _compact_value("vllm/vllm-openai:latest") == "vllm-vllm-oai-latest"


def test_compact_value_kv_cache_dtype():
    assert _compact_value("--kv-cache-dtype fp8") == "kv-cache-dtype-fp8"


def test_compact_value_expert_parallel():
    assert _compact_value("--kv-cache-dtype fp8 --enable-expert-parallel") == "kv-cache-dtype-fp8-e-exp-par"


# ── Variant.__str__ ──────────────────────────────────────────────


def test_str_deploy_only():
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1})
    assert str(v) == "rtx5090x1"


def test_str_deploy_multi_gpu():
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 4})
    assert str(v) == "rtx5090x4"


def test_str_with_one_non_deploy_param():
    v = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "benchmark.max_concurrency": 128,
        }
    )
    assert str(v) == "rtx5090x1_mc128"


def test_str_with_multiple_non_deploy_params():
    v = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "benchmark.max_concurrency": 8,
            "engine.llm.max_concurrent_requests": 8,
            "benchmark.num_prompts": 80,
        }
    )
    result = str(v)
    assert result.startswith("rtx5090x1_")
    assert "mc8" in result
    assert "mcr8" in result
    assert "np80" in result


def test_str_params_sorted():
    v = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "benchmark.num_prompts": 80,
            "benchmark.max_concurrency": 8,
        }
    )
    result = str(v)
    # benchmark.max_concurrency sorts before benchmark.num_prompts
    assert result == "rtx5090x1_mc8_np80"


def test_str_default_gpu_count():
    """gpu_count defaults to 1 when not in params."""
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090"})
    assert str(v) == "rtx5090x1"


def test_str_compacts_slashes_and_spaces():
    """Values with / and spaces are compacted to short filesystem-safe labels."""
    v = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "engine.llm.sglang.image": "lmsysorg/sglang:latest",
            "engine.llm.sglang.extra_args": "--quantization moe_wna16",
        }
    )
    result = str(v)
    assert "/" not in result
    assert " " not in result
    assert ":" not in result
    assert result == "rtx5090x1_eaquant-moe-wna16_ilms-sglang-latest"


def test_str_compacts_docker_image():
    """Docker image refs like vllm/vllm-openai:latest are compacted."""
    v = Variant(
        params={
            "deploy.gpu": "NVIDIA GeForce RTX 5090",
            "deploy.gpu_count": 1,
            "engine.llm.vllm.image": "vllm/vllm-openai:latest",
        }
    )
    result = str(v)
    assert "/" not in result
    assert result == "rtx5090x1_ivllm-vllm-oai-latest"


# ── Variant.gpu_short ────────────────────────────────────────────


def test_gpu_short_known():
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090"})
    assert v.gpu_short == "rtx5090"


def test_gpu_short_h100():
    v = Variant(params={"deploy.gpu": "NVIDIA H100 80GB"})
    assert v.gpu_short == "h100"


# ── Variant.gpu_count ────────────────────────────────────────────


def test_gpu_count_explicit():
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 4})
    assert v.gpu_count == 4


def test_gpu_count_default():
    v = Variant(params={"deploy.gpu": "NVIDIA GeForce RTX 5090"})
    assert v.gpu_count == 1


# ── Variant.__eq__ ───────────────────────────────────────────────


def test_eq_same_params():
    v1 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    assert v1 == v2


def test_eq_different_params():
    v1 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 2})
    assert v1 != v2


def test_eq_not_implemented_for_str():
    v = Variant(params={"deploy.gpu": "GPU_A"})
    assert v != "some_string"


# ── Variant.__hash__ ─────────────────────────────────────────────


def test_hash_consistent():
    v1 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    assert hash(v1) == hash(v2)


def test_hash_different_for_different_params():
    v1 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 2})
    assert hash(v1) != hash(v2)


def test_usable_in_set():
    v1 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v2 = Variant(params={"deploy.gpu": "GPU_A", "deploy.gpu_count": 1})
    v3 = Variant(params={"deploy.gpu": "GPU_B", "deploy.gpu_count": 1})
    s = {v1, v2, v3}
    assert len(s) == 2
