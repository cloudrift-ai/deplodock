"""Download a model from HuggingFace to the standard HF cache."""

import logging
import sys

logger = logging.getLogger(__name__)


def register_pull_command(subparsers):
    parser = subparsers.add_parser("pull", help="Download a model from HuggingFace")
    parser.add_argument("model", help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)")
    parser.set_defaults(func=handle_pull)


def handle_pull(args):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required: pip install huggingface_hub")
        sys.exit(1)

    logger.info("Pulling %s...", args.model)
    path = snapshot_download(args.model)
    logger.info("Cached at: %s", path)
