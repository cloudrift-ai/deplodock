#!/usr/bin/env python
"""Merge the ``node`` table from another deplodock autotune DB into the local one.

The autotune ``node`` table is a **cross-hardware** dataset — its key folds the GPU
identity (``digest(context_key, gpu, op_sig, knobs)``) — so node rows measured on a
rented GPU can be merged into the single canonical local DB without colliding with
other cards (keep-min collapses rows only within a card). This is the copy-back step
of the ``collect-node-data`` skill: it wraps the tested core
:meth:`SearchDB.merge_nodes`.

Two input modes:

    # merge an already-local snapshot
    ./venv/bin/python scripts/merge_node_db.py --src /tmp/remote_autotune.db

    # fetch a WAL-safe snapshot from a remote host over SSH, then merge
    ./venv/bin/python scripts/merge_node_db.py --remote user@host \
        [--ssh-key ~/.ssh/id_ed25519] [--port 22] [--remote-db ~/.cache/deplodock/autotune.db]

The destination defaults to the local tune DB (``DEPLODOCK_TUNE_DB`` or
``~/.cache/deplodock/autotune.db``); override with ``--db``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from deplodock.commands.compile import resolve_tune_db
from deplodock.compiler.pipeline.search.db import SearchDB

# Same hardening as deplodock/provisioning/ssh_transport.py::ssh_base_args (kept
# inline because that helper appends the server and uses ssh's ``-p`` — scp needs ``-P``).
_SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=20",
]

# WAL-safe single-file snapshot: VACUUM INTO applies the WAL and writes a clean copy,
# so we never scp a half-written DB or its -wal/-shm sidecars. Piped to the remote
# ``python3 -`` over stdin so there is no shell quoting to get wrong.
_REMOTE_SNAPSHOT_PY = """
import os, sqlite3
src = os.path.expanduser({remote_db!r})
dst = {snapshot!r}
if os.path.exists(dst):
    os.remove(dst)  # VACUUM INTO refuses to overwrite
con = sqlite3.connect(src)
con.execute("VACUUM INTO ?", (dst,))
con.close()
snap = sqlite3.connect(dst)
try:
    n = snap.execute("SELECT count(*) FROM node").fetchone()[0]
except sqlite3.OperationalError:
    n = "(no node table)"
snap.close()
print("remote snapshot node rows:", n)
"""


def _key_opts(ssh_key: str | None) -> list[str]:
    return ["-i", os.path.expanduser(ssh_key)] if ssh_key else []


def _fetch_remote_snapshot(server: str, *, ssh_key: str | None, port: int | None, remote_db: str) -> Path:
    """Snapshot the remote autotune DB (VACUUM INTO) and scp it back to a local temp
    file, returning that local path. The remote snapshot path is per-process (pid) so
    two merges against one host don't clobber each other, and is removed after the fetch."""
    key = _key_opts(ssh_key)
    remote_snap = f"/tmp/deplodock_nodes_snapshot_{os.getpid()}.db"
    port_ssh = ["-p", str(port)] if port else []
    ssh_cmd = ["ssh", *_SSH_OPTS, *key, *port_ssh, server, "python3 -"]
    code = _REMOTE_SNAPSHOT_PY.format(remote_db=remote_db, snapshot=remote_snap)
    print(f"[merge_node_db] snapshotting {server}:{remote_db} ...")
    subprocess.run(ssh_cmd, input=code, text=True, check=True)

    fd, local = tempfile.mkstemp(prefix="deplodock_nodes_", suffix=".db")
    os.close(fd)
    scp_cmd = ["scp", *_SSH_OPTS, *key, *(["-P", str(port)] if port else []), f"{server}:{remote_snap}", local]
    print(f"[merge_node_db] fetching snapshot -> {local}")
    subprocess.run(scp_cmd, check=True)
    # Drop the remote throwaway copy (best-effort — the VM is usually torn down anyway).
    subprocess.run(["ssh", *_SSH_OPTS, *key, *port_ssh, server, f"rm -f {remote_snap}"], check=False)
    return Path(local)


def _merge_and_report(src_path: Path | str, dest: str | None) -> int:
    """Keep-min merge ``src_path``'s node rows into the destination DB (default: the
    local tune DB) and print the per-card receipt. Returns the rows merged."""
    dest_path = Path(dest).expanduser() if dest else resolve_tune_db()
    db = SearchDB(dest_path)  # creates/migrates the node table + gpu column on the destination
    try:
        merged = db.merge_nodes(src_path)
        print(f"[merge_node_db] merged {merged} node rows from {src_path} into {dest_path}")
        counts = Counter(n.gpu for n in db.iter_nodes())
        print("[merge_node_db] node rows per card now:")
        for gpu, count in sorted(counts.items()):
            print(f"    {gpu or '(unknown card)'}: {count}")
        return merged
    finally:
        db.close()


def fetch_and_merge(
    remote: str,
    *,
    ssh_key: str | None = None,
    port: int | None = None,
    remote_db: str = "~/.cache/deplodock/autotune.db",
    dest: str | None = None,
) -> int:
    """Snapshot the remote autotune DB, scp it back, keep-min merge its node rows into
    ``dest`` (default: the local tune DB), and print the per-card receipt. Returns the
    rows merged. The reusable entry point shared by this CLI and the folded-in merge
    step of ``remote_node_tune.py``."""
    src = _fetch_remote_snapshot(remote, ssh_key=ssh_key, port=port, remote_db=remote_db)
    try:
        return _merge_and_report(src, dest)
    finally:
        src.unlink(missing_ok=True)  # drop the ~40MB local snapshot copy once merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="Local path to an autotune DB whose node rows to merge in.")
    g.add_argument("--remote", help="SSH target (user@host) to snapshot + fetch the node rows from.")
    p.add_argument("--ssh-key", help="SSH private key for --remote (default: ssh's own default).")
    p.add_argument("--port", type=int, help="SSH port for --remote (default 22).")
    p.add_argument(
        "--remote-db",
        default="~/.cache/deplodock/autotune.db",
        help="Remote autotune DB path for --remote (default ~/.cache/deplodock/autotune.db).",
    )
    p.add_argument("--db", help="Destination DB (default: DEPLODOCK_TUNE_DB or ~/.cache/deplodock/autotune.db).")
    args = p.parse_args()

    if args.remote:
        fetch_and_merge(args.remote, ssh_key=args.ssh_key, port=args.port, remote_db=args.remote_db, dest=args.db)
    else:
        src = Path(args.src).expanduser()
        if not src.exists():
            p.error(f"--src not found: {src}")
        _merge_and_report(src, args.db)


if __name__ == "__main__":
    sys.exit(main())
