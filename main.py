#!/usr/bin/env python3
"""Server benchmark tools — CLI entrypoint."""

import argparse
import sys

from commands.deploy.local import register_local_target
from commands.deploy.ssh import register_ssh_target


def main():
    parser = argparse.ArgumentParser(description="Server benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deploy subcommand with target sub-subcommands
    deploy_parser = subparsers.add_parser("deploy", help="Deploy LLM models")
    deploy_subparsers = deploy_parser.add_subparsers(dest="target", required=True)

    register_local_target(deploy_subparsers)
    register_ssh_target(deploy_subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
