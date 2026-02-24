"""Deploy parameters dataclass."""

from dataclasses import dataclass, field

from deplodock.recipe.types import Recipe


@dataclass
class DeployParams:
    """All parameters needed for a single deployment. Serializable for future API use."""

    server: str  # user@host or IP
    ssh_key: str  # path to SSH private key
    ssh_port: int = 22
    recipe: Recipe = field(default_factory=Recipe)
    model_dir: str = "/hf_models"
    hf_token: str = ""
    dry_run: bool = False
    gpu_device_ids: list[int] | None = None
