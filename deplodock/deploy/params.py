"""Deploy parameters dataclass."""

from dataclasses import dataclass, field


@dataclass
class DeployParams:
    """All parameters needed for a single deployment. Serializable for future API use."""

    server: str  # user@host or IP
    ssh_key: str  # path to SSH private key
    ssh_port: int = 22
    recipe_config: dict = field(default_factory=dict)  # pre-loaded recipe
    model_dir: str = "/hf_models"
    hf_token: str = ""
    dry_run: bool = False
