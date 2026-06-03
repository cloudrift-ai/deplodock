"""``deplodock tune-data`` subcommand group: status / pull / publish / clean-cache.

These are the human-driven counterparts to autofetch: publish the local tuning
DB as a release asset, pull a published DB into the local cache without
running tuning, inspect what's resolved for this process, or clear the cache.
"""

from __future__ import annotations

import logging
import sys

from deplodock import config

logger = logging.getLogger(__name__)


def register_tune_data_command(subparsers):
    parser = subparsers.add_parser(
        "tune-data",
        help="Publish / pull / inspect published tuning-data releases.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    s = sub.add_parser("status", help="Show installed version, detected GPU, resolved DB, cache state.")
    s.set_defaults(func=_handle_status)

    p = sub.add_parser("pull", help="Pull the published DB matching this GPU into the local cache.")
    p.add_argument("--gpu", default=None, help="Override GPU name (default: detect locally).")
    p.add_argument("--release", default=None, help="Override release tag (default: installed version).")
    p.set_defaults(func=_handle_pull)

    u = sub.add_parser("publish", help="Compress + upload the local tuning DB as a release asset; update the manifest.")
    u.add_argument("--gpu", default=None, help="Override GPU name (default: detect locally).")
    u.add_argument("--release", default=None, help="Override release tag (default: installed version).")
    u.add_argument("--contributor", default="", help="Optional contributor handle (e.g. @username).")
    u.add_argument("--db", default=None, help="Path to the tuning DB to publish (default: DEPLODOCK_TUNE_DB).")
    u.set_defaults(func=_handle_publish)

    c = sub.add_parser("clean-cache", help="Wipe the cached published manifests + DBs.")
    c.add_argument("--release", default=None, help="Only clean this release (default: all).")
    c.set_defaults(func=_handle_clean)


def _handle_status(args):  # noqa: ARG001
    from deplodock.publish.publish_flow import status_text

    print(status_text())


def _detect_gpu(override: str | None) -> tuple[str, int]:
    from deplodock.detect import detect_local_gpus

    if override:
        return override, 0
    return detect_local_gpus()


def _handle_pull(args):
    from deplodock.publish.publish_flow import pull_published_db

    gpu_name, _ = _detect_gpu(args.gpu)
    path = pull_published_db(gpu_name, release_tag=args.release)
    if path is None:
        logger.error("no published DB available for %s", gpu_name)
        sys.exit(1)
    print(str(path))


def _handle_publish(args):
    from pathlib import Path

    from deplodock.publish.publish_flow import publish_local_db

    db_path = Path(args.db) if args.db else config.tune_db_path()
    gpu_name, _ = _detect_gpu(args.gpu)
    entry = publish_local_db(
        db_path,
        gpu_name,
        contributor=args.contributor,
        release_tag=args.release,
    )
    print(f"published {entry.url}")


def _handle_clean(args):
    import shutil

    from deplodock.publish import cache

    if args.release:
        cache.clear_release(args.release)
    else:
        root = cache.cache_root()
        if root.exists():
            shutil.rmtree(root)
            logger.info("cleared cache root %s", root)
