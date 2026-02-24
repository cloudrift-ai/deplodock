"""Unit tests for recipe dataclass types."""

from deplodock.recipe import (
    LLMConfig,
    Recipe,
    SglangConfig,
    VllmConfig,
)

# ── VllmConfig / SglangConfig ────────────────────────────────────


def test_vllm_config_defaults():
    cfg = VllmConfig()
    assert cfg.image == "vllm/vllm-openai:latest"
    assert cfg.extra_args == ""


def test_sglang_config_defaults():
    cfg = SglangConfig()
    assert cfg.image == "lmsysorg/sglang:latest"
    assert cfg.extra_args == ""


# ── LLMConfig properties ─────────────────────────────────────────


def test_llm_engine_name_defaults_to_vllm():
    llm = LLMConfig()
    assert llm.engine_name == "vllm"


def test_llm_engine_name_sglang():
    llm = LLMConfig(sglang=SglangConfig())
    assert llm.engine_name == "sglang"


def test_llm_gpus_per_instance():
    llm = LLMConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
    assert llm.gpus_per_instance == 8


def test_llm_gpus_per_instance_defaults():
    llm = LLMConfig()
    assert llm.gpus_per_instance == 1


def test_llm_image_vllm():
    llm = LLMConfig(vllm=VllmConfig(image="custom/vllm:v2"))
    assert llm.image == "custom/vllm:v2"


def test_llm_image_sglang():
    llm = LLMConfig(sglang=SglangConfig(image="custom/sglang:v3"))
    assert llm.image == "custom/sglang:v3"


def test_llm_image_fallback():
    llm = LLMConfig()
    assert llm.image == "vllm/vllm-openai:latest"


def test_llm_extra_args_vllm():
    llm = LLMConfig(vllm=VllmConfig(extra_args="--kv-cache-dtype fp8"))
    assert llm.extra_args == "--kv-cache-dtype fp8"


def test_llm_extra_args_sglang():
    llm = LLMConfig(sglang=SglangConfig(extra_args="--chunked-prefill-size 4096"))
    assert llm.extra_args == "--chunked-prefill-size 4096"


def test_llm_extra_args_empty_default():
    llm = LLMConfig()
    assert llm.extra_args == ""


def test_llm_optional_fields_default_none():
    llm = LLMConfig()
    assert llm.context_length is None
    assert llm.max_concurrent_requests is None


# ── Recipe.from_dict ──────────────────────────────────────────────


def test_from_dict_minimal():
    d = {"model": {"huggingface": "org/model"}}
    recipe = Recipe.from_dict(d)
    assert recipe.model.huggingface == "org/model"
    assert recipe.model_name == "org/model"
    assert recipe.engine.llm.tensor_parallel_size == 1
    assert recipe.gpu is None
    assert recipe.gpu_count == 1


def test_from_dict_full():
    d = {
        "model": {"huggingface": "org/model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.95,
                "context_length": 16384,
                "max_concurrent_requests": 512,
                "vllm": {
                    "image": "custom/vllm:v2",
                    "extra_args": "--kv-cache-dtype fp8",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 64,
            "num_prompts": 128,
            "random_input_len": 2000,
            "random_output_len": 3000,
        },
        "gpu": "NVIDIA H200",
        "gpu_count": 8,
    }
    recipe = Recipe.from_dict(d)
    assert recipe.engine.llm.tensor_parallel_size == 8
    assert recipe.engine.llm.gpu_memory_utilization == 0.95
    assert recipe.engine.llm.context_length == 16384
    assert recipe.engine.llm.max_concurrent_requests == 512
    assert recipe.engine.llm.vllm.image == "custom/vllm:v2"
    assert recipe.engine.llm.vllm.extra_args == "--kv-cache-dtype fp8"
    assert recipe.benchmark.max_concurrency == 64
    assert recipe.benchmark.num_prompts == 128
    assert recipe.gpu == "NVIDIA H200"
    assert recipe.gpu_count == 8


def test_from_dict_sglang():
    d = {
        "model": {"huggingface": "org/model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 2,
                "sglang": {
                    "image": "lmsysorg/sglang:v0.5",
                    "extra_args": "--chunked-prefill-size 4096",
                },
            }
        },
    }
    recipe = Recipe.from_dict(d)
    assert recipe.engine.llm.engine_name == "sglang"
    assert recipe.engine.llm.sglang.image == "lmsysorg/sglang:v0.5"
    assert recipe.engine.llm.vllm is None


def test_from_dict_no_engine_section():
    d = {"model": {"huggingface": "org/model"}}
    recipe = Recipe.from_dict(d)
    assert recipe.engine.llm.engine_name == "vllm"
    assert recipe.engine.llm.vllm is None
    assert recipe.engine.llm.image == "vllm/vllm-openai:latest"


def test_from_dict_benchmark_defaults():
    d = {"model": {"huggingface": "org/model"}}
    recipe = Recipe.from_dict(d)
    assert recipe.benchmark.max_concurrency == 128
    assert recipe.benchmark.num_prompts == 256
    assert recipe.benchmark.random_input_len == 8000
    assert recipe.benchmark.random_output_len == 8000
