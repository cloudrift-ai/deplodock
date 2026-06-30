#!/usr/bin/env python3
"""Server benchmark tools — CLI entrypoint."""

import argparse

from emmy.commands.bench import register_bench_command
from emmy.commands.compare import register_compare_command
from emmy.commands.compile import register_compile_command
from emmy.commands.deploy.cloud import register_cloud_target
from emmy.commands.deploy.local import register_local_target
from emmy.commands.deploy.ssh import register_ssh_target
from emmy.commands.eval import register_eval_command
from emmy.commands.generate import register_generate_command
from emmy.commands.inspect_graph import register_inspect_command
from emmy.commands.pull import register_pull_command
from emmy.commands.run import register_run_command
from emmy.commands.serve import register_serve_command
from emmy.commands.teardown import register_teardown_command
from emmy.commands.trace import register_trace_command
from emmy.commands.tune import register_tune_command
from emmy.commands.vm import register_vm_command
from emmy.logging_setup import setup_cli_logging


def main():
    parser = argparse.ArgumentParser(description="Server benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deploy subcommand with target sub-subcommands
    deploy_parser = subparsers.add_parser("deploy", help="Deploy LLM models")
    deploy_subparsers = deploy_parser.add_subparsers(dest="target", required=True)

    register_local_target(deploy_subparsers)
    register_ssh_target(deploy_subparsers)
    register_cloud_target(deploy_subparsers)

    # bench, serve, teardown, vm subcommands
    register_bench_command(subparsers)
    register_serve_command(subparsers)
    register_teardown_command(subparsers)
    register_vm_command(subparsers)

    # compiler workflow commands
    register_pull_command(subparsers)
    register_trace_command(subparsers)
    register_compile_command(subparsers)
    register_tune_command(subparsers)
    register_run_command(subparsers)
    register_generate_command(subparsers)
    register_inspect_command(subparsers)
    register_eval_command(subparsers)
    register_compare_command(subparsers)

    args = parser.parse_args()
    setup_cli_logging()
    args.func(args)


if __name__ == "__main__":
    main()
