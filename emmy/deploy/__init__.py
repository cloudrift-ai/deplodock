"""Deploy library: compose generation, deploy orchestration, scale-out strategies."""

from emmy.deploy.compose import (
    calculate_num_instances,
    generate_compose,
    generate_nginx_conf,
)
from emmy.deploy.orchestrate import (
    deploy,
    run_deploy,
    run_teardown,
    teardown,
)
from emmy.deploy.params import DeployParams
from emmy.deploy.scale_out import (
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
