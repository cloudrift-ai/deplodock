"""Recipe dataclass types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VllmConfig:
    """vLLM engine-specific configuration."""

    image: str = "vllm/vllm-openai:v0.17.0"
    extra_args: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class SglangConfig:
    """SGLang engine-specific configuration."""

    image: str = "lmsysorg/sglang:v0.5.9"
    extra_args: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Engine-agnostic LLM serving configuration."""

    context_length: int | None = None
    max_concurrent_requests: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    vllm: VllmConfig | None = None
    sglang: SglangConfig | None = None
    docker_options: dict[str, Any] = field(default_factory=dict)

    @property
    def gpus_per_instance(self) -> int:
        """Number of GPUs consumed by one model instance."""
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size

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

    @property
    def extra_env(self) -> dict[str, str]:
        """Extra environment variables for the active engine."""
        if self.sglang is not None:
            return self.sglang.extra_env
        if self.vllm is not None:
            return self.vllm.extra_env
        return {}


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
class CommandConfig:
    """Generic command workload configuration.

    Used by command-style recipes that run an arbitrary tool on the
    provisioned VM (instead of deploying an inference server).

    Fields:
        stage: Repo paths to stage to the remote VM (via git ls-files +
            tar). Empty list = no staging; $repo_dir is unavailable.
        run: Shell command template (string.Template $-syntax). Substituted
            with variant params (flattened to leaf names) plus injected
            $task_dir, $repo_dir, $gpu_device_ids.
        result_files: List of result file names or shell globs (e.g.
            "result.json", "*.log"). Globs expand on the remote; each
            matched file is pulled back as {variant}_{basename}.
        timeout: Per-task command timeout in seconds.
        env: Extra environment variables to set on the remote command.
    """

    stage: list[str] = field(default_factory=list)
    run: str = ""
    result_files: list[str] = field(default_factory=list)
    timeout: int = 1800
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class AggregateConfig:
    """Post-processing step that runs locally after all variants complete.

    The ``run`` template receives ``$run_dir`` (the local directory containing
    all pulled-back result files). Runs on the orchestrator, not on a GPU VM.
    """

    run: str = ""
    timeout: int = 300


@dataclass
class DeployConfig:
    """Optional deploy section — GPU info for cloud provisioning."""

    gpu: str | None = None
    gpu_count: int = 1
    driver_version: str | None = None
    cuda_version: str | None = None


@dataclass
class Recipe:
    """Complete recipe configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    command: CommandConfig | None = None
    aggregate: AggregateConfig | None = None

    @property
    def kind(self) -> str:
        """Recipe kind: 'command' if a command block is set, else 'inference'."""
        return "command" if self.command is not None else "inference"

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
            data_parallel_size=llm_dict.get("data_parallel_size", 1),
            gpu_memory_utilization=llm_dict.get("gpu_memory_utilization", 0.9),
            vllm=vllm,
            sglang=sglang,
            docker_options=llm_dict.get("docker_options", {}),
        )

        bench_dict = d.get("benchmark", {})
        benchmark = BenchmarkConfig(
            max_concurrency=bench_dict.get("max_concurrency", 128),
            num_prompts=bench_dict.get("num_prompts", 256),
            random_input_len=bench_dict.get("random_input_len", 8000),
            random_output_len=bench_dict.get("random_output_len", 8000),
        )

        deploy_dict = d.get("deploy", {})
        deploy = DeployConfig(
            gpu=deploy_dict.get("gpu"),
            gpu_count=deploy_dict.get("gpu_count", 1),
            driver_version=deploy_dict.get("driver_version"),
            cuda_version=deploy_dict.get("cuda_version"),
        )

        command = None
        cmd_dict = d.get("command")
        if cmd_dict is not None:
            command = CommandConfig(
                stage=list(cmd_dict.get("stage", [])),
                run=cmd_dict.get("run", ""),
                result_files=list(cmd_dict.get("result_files", [])),
                timeout=cmd_dict.get("timeout", 1800),
                env=dict(cmd_dict.get("env", {})),
            )

        aggregate = None
        agg_dict = d.get("aggregate")
        if agg_dict is not None:
            aggregate = AggregateConfig(
                run=agg_dict.get("run", ""),
                timeout=agg_dict.get("timeout", 300),
            )

        return cls(
            model=model,
            engine=EngineConfig(llm=llm),
            benchmark=benchmark,
            deploy=deploy,
            command=command,
            aggregate=aggregate,
        )

    @property
    def model_name(self) -> str:
        """Shortcut for model.huggingface."""
        return self.model.huggingface
