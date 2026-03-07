"""Deploy library: compose generation, deploy orchestration, scale-out strategies."""

from deplodock.deploy.compose import (
    calculate_num_instances,
    generate_compose,
    generate_nginx_conf,
)
from deplodock.deploy.orchestrate import (
    deploy,
    run_deploy,
    run_teardown,
    teardown,
)
from deplodock.deploy.params import DeployParams
from deplodock.deploy.scale_out import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    DataParallelismScaleOutStrategy,
    ReplicaParallelismScaleOutStrategy,
    ScaleOutStrategy,
)

__all__ = [
    "DEFAULT_STRATEGY",
    "DataParallelismScaleOutStrategy",
    "DeployParams",
    "ReplicaParallelismScaleOutStrategy",
    "STRATEGIES",
    "ScaleOutStrategy",
    "calculate_num_instances",
    "deploy",
    "generate_compose",
    "generate_nginx_conf",
    "run_deploy",
    "run_teardown",
    "teardown",
]
