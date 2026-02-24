"""Deploy library: recipe loading, compose generation, deploy orchestration."""

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
from deplodock.deploy.recipe import deep_merge, load_recipe

__all__ = [
    "DeployParams",
    "deep_merge",
    "load_recipe",
    "calculate_num_instances",
    "generate_compose",
    "generate_nginx_conf",
    "run_deploy",
    "run_teardown",
    "deploy",
    "teardown",
]
