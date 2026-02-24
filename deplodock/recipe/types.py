"""Recipe dataclass types."""

from dataclasses import dataclass, field


@dataclass
class VllmConfig:
    """vLLM engine-specific configuration."""

    image: str = "vllm/vllm-openai:latest"
    extra_args: str = ""


@dataclass
class SglangConfig:
    """SGLang engine-specific configuration."""

    image: str = "lmsysorg/sglang:latest"
    extra_args: str = ""


@dataclass
class LLMConfig:
    """Engine-agnostic LLM serving configuration."""

    context_length: int | None = None
    max_concurrent_requests: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm: VllmConfig | None = None
    sglang: SglangConfig | None = None

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs consumed by one model instance."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def engine_name(self) -> str:
        """Active engine name: 'vllm' or 'sglang'."""
        if self.sglang is not None:
            return "sglang"
        return "vllm"

    @property
    def image(self) -> str:
        """Docker image for the active engine."""
        if self.sglang is not None:
            return self.sglang.image
        if self.vllm is not None:
            return self.vllm.image
        return VllmConfig().image

    @property
    def entrypoint(self) -> str | None:
        """Docker entrypoint override for the active engine.

        vLLM images have a built-in entrypoint; SGLang images do not,
        so we must provide one explicitly.
        """
        if self.sglang is not None:
            return "python3 -m sglang.launch_server"
        return None

    @property
    def extra_args(self) -> str:
        """Extra CLI flags for the active engine."""
        if self.sglang is not None:
            return self.sglang.extra_args
        if self.vllm is not None:
            return self.vllm.extra_args
        return ""


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class ModelConfig:
    """Model configuration."""

    huggingface: str = ""


@dataclass
class BenchmarkConfig:
    """Benchmark workload configuration."""

    max_concurrency: int = 128
    num_prompts: int = 256
    random_input_len: int = 8000
    random_output_len: int = 8000


@dataclass
class Recipe:
    """Complete recipe configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    gpu: str | None = None
    gpu_count: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        """Build a Recipe from a (post-merge, post-migrate) config dict."""
        model_dict = d.get("model", {})
        model = ModelConfig(huggingface=model_dict.get("huggingface", ""))

        engine_dict = d.get("engine", {})
        llm_dict = engine_dict.get("llm", {})

        vllm_dict = llm_dict.get("vllm")
        vllm = VllmConfig(**vllm_dict) if vllm_dict is not None else None

        sglang_dict = llm_dict.get("sglang")
        sglang = SglangConfig(**sglang_dict) if sglang_dict is not None else None

        llm = LLMConfig(
            context_length=llm_dict.get("context_length"),
            max_concurrent_requests=llm_dict.get("max_concurrent_requests"),
            tensor_parallel_size=llm_dict.get("tensor_parallel_size", 1),
            pipeline_parallel_size=llm_dict.get("pipeline_parallel_size", 1),
            gpu_memory_utilization=llm_dict.get("gpu_memory_utilization", 0.9),
            vllm=vllm,
            sglang=sglang,
        )

        bench_dict = d.get("benchmark", {})
        benchmark = BenchmarkConfig(
            max_concurrency=bench_dict.get("max_concurrency", 128),
            num_prompts=bench_dict.get("num_prompts", 256),
            random_input_len=bench_dict.get("random_input_len", 8000),
            random_output_len=bench_dict.get("random_output_len", 8000),
        )

        return cls(
            model=model,
            engine=EngineConfig(llm=llm),
            benchmark=benchmark,
            gpu=d.get("gpu"),
            gpu_count=d.get("gpu_count", 1),
        )

    @property
    def model_name(self) -> str:
        """Shortcut for model.huggingface."""
        return self.model.huggingface
