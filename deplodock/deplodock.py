#!/usr/bin/env python3
"""Server benchmark tools — CLI entrypoint."""

import argparse

from deplodock.commands.bench import register_bench_command
from deplodock.commands.compile import register_compile_command
from deplodock.commands.deploy.cloud import register_cloud_target
from deplodock.commands.deploy.local import register_local_target
from deplodock.commands.deploy.ssh import register_ssh_target
from deplodock.commands.inspect_graph import register_inspect_command
from deplodock.commands.pull import register_pull_command
from deplodock.commands.run import register_run_command
from deplodock.commands.teardown import register_teardown_command
from deplodock.commands.trace import register_trace_command
from deplodock.commands.vm import register_vm_command
from deplodock.logging_setup import setup_cli_logging


def main():
    parser = argparse.ArgumentParser(description="Server benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deploy subcommand with target sub-subcommands
    deploy_parser = subparsers.add_parser("deploy", help="Deploy LLM models")
    deploy_subparsers = deploy_parser.add_subparsers(dest="target", required=True)

    register_local_target(deploy_subparsers)
    register_ssh_target(deploy_subparsers)
    register_cloud_target(deploy_subparsers)

    # bench, teardown, vm subcommands
    register_bench_command(subparsers)
    register_teardown_command(subparsers)
    register_vm_command(subparsers)

    # compiler workflow commands
    register_pull_command(subparsers)
    register_trace_command(subparsers)
    register_compile_command(subparsers)
    register_inspect_command(subparsers)
    register_run_command(subparsers)

    args = parser.parse_args()
    setup_cli_logging()
    args.func(args)


if __name__ == "__main__":
    main()
