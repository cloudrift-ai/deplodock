"""Unit tests for engine flag mapping and CLI argument building."""

from deplodock.recipe import LLMConfig, SglangConfig, VllmConfig, build_engine_args
from deplodock.recipe.engines import (
    _HARDCODED_FLAGS,
    SGLANG_FLAG_MAP,
    VLLM_FLAG_MAP,
    banned_extra_arg_flags,
)

# ── banned_extra_arg_flags ────────────────────────────────────────


def test_banned_flags_vllm_includes_flag_map():
    banned = banned_extra_arg_flags("vllm")
    for flag in VLLM_FLAG_MAP.values():
        assert flag in banned


def test_banned_flags_sglang_includes_flag_map():
    banned = banned_extra_arg_flags("sglang")
    for flag in SGLANG_FLAG_MAP.values():
        assert flag in banned


def test_banned_flags_include_hardcoded():
    banned = banned_extra_arg_flags("vllm")
    for flag in _HARDCODED_FLAGS:
        assert flag in banned


def test_banned_flags_vllm_excludes_sglang():
    banned = banned_extra_arg_flags("vllm")
    # SGLang-specific flags like --tp should not be in vLLM banned set
    assert "--tp" not in banned


def test_banned_flags_sglang_excludes_vllm():
    banned = banned_extra_arg_flags("sglang")
    assert "--tensor-parallel-size" not in banned


# ── build_engine_args (vLLM) ──────────────────────────────────────


def test_build_args_vllm_basic():
    llm = LLMConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        vllm=VllmConfig(),
    )
    args = build_engine_args(llm, "org/model")
    assert "--trust-remote-code" in args
    assert "--host 0.0.0.0" in args
    assert "--port 8000" in args
    assert "--tensor-parallel-size 2" in args
    assert "--pipeline-parallel-size 1" in args
    assert "--model org/model" in args
    assert "--served-model-name org/model" in args


def test_build_args_vllm_gpu_memory_uses_equals():
    llm = LLMConfig(gpu_memory_utilization=0.95, vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert "--gpu-memory-utilization=0.95" in args


def test_build_args_vllm_context_length():
    llm = LLMConfig(context_length=16384, vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert "--max-model-len 16384" in args


def test_build_args_vllm_omits_none_context_length():
    llm = LLMConfig(vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert not any("--max-model-len" in a for a in args)


def test_build_args_vllm_max_concurrent():
    llm = LLMConfig(max_concurrent_requests=256, vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert "--max-num-seqs 256" in args


def test_build_args_vllm_omits_none_max_concurrent():
    llm = LLMConfig(vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert not any("--max-num-seqs" in a for a in args)


def test_build_args_vllm_extra_args():
    llm = LLMConfig(vllm=VllmConfig(extra_args="--kv-cache-dtype fp8 --enable-expert-parallel"))
    args = build_engine_args(llm, "org/model")
    assert "--kv-cache-dtype fp8 --enable-expert-parallel" in args


def test_build_args_vllm_no_extra_args():
    llm = LLMConfig(vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert "" not in args


# ── build_engine_args (SGLang) ────────────────────────────────────


def test_build_args_sglang_basic():
    llm = LLMConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.85,
        sglang=SglangConfig(),
    )
    args = build_engine_args(llm, "org/model")
    assert "--tp 4" in args
    assert "--dp 2" in args
    assert "--mem-fraction-static 0.85" in args
    assert "--trust-remote-code" in args


def test_build_args_sglang_uses_model_path():
    llm = LLMConfig(sglang=SglangConfig())
    args = build_engine_args(llm, "org/model")
    assert "--model-path org/model" in args
    assert "--model org/model" not in args


def test_build_args_vllm_uses_model_not_model_path():
    llm = LLMConfig(vllm=VllmConfig())
    args = build_engine_args(llm, "org/model")
    assert "--model org/model" in args
    assert "--model-path org/model" not in args


def test_build_args_sglang_context_length():
    llm = LLMConfig(context_length=8192, sglang=SglangConfig())
    args = build_engine_args(llm, "org/model")
    assert "--context-length 8192" in args


def test_build_args_sglang_max_concurrent():
    llm = LLMConfig(max_concurrent_requests=128, sglang=SglangConfig())
    args = build_engine_args(llm, "org/model")
    assert "--max-running-requests 128" in args


def test_banned_flags_include_model_path():
    """--model-path must be banned for both engines."""
    assert "--model-path" in banned_extra_arg_flags("vllm")
    assert "--model-path" in banned_extra_arg_flags("sglang")
