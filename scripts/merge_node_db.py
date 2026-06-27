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

_REMOTE_SNAPSHOT_PATH = "/tmp/deplodock_nodes_snapshot.db"


def _key_opts(ssh_key: str | None) -> list[str]:
    return ["-i", os.path.expanduser(ssh_key)] if ssh_key else []


def _fetch_remote_snapshot(server: str, *, ssh_key: str | None, port: int | None, remote_db: str) -> Path:
    """Snapshot the remote autotune DB (VACUUM INTO) and scp it back to a local temp
    file, returning that local path."""
    key = _key_opts(ssh_key)
    ssh_cmd = ["ssh", *_SSH_OPTS, *key, *(["-p", str(port)] if port else []), server, "python3 -"]
    code = _REMOTE_SNAPSHOT_PY.format(remote_db=remote_db, snapshot=_REMOTE_SNAPSHOT_PATH)
    print(f"[merge_node_db] snapshotting {server}:{remote_db} ...")
    subprocess.run(ssh_cmd, input=code, text=True, check=True)

    fd, local = tempfile.mkstemp(prefix="deplodock_nodes_", suffix=".db")
    os.close(fd)
    scp_cmd = ["scp", *_SSH_OPTS, *key, *(["-P", str(port)] if port else []), f"{server}:{_REMOTE_SNAPSHOT_PATH}", local]
    print(f"[merge_node_db] fetching snapshot -> {local}")
    subprocess.run(scp_cmd, check=True)
    return Path(local)


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
        src = _fetch_remote_snapshot(args.remote, ssh_key=args.ssh_key, port=args.port, remote_db=args.remote_db)
    else:
        src = Path(args.src).expanduser()
        if not src.exists():
            p.error(f"--src not found: {src}")

    dest = Path(args.db).expanduser() if args.db else resolve_tune_db()
    db = SearchDB(dest)  # creates/migrates the node table + gpu column on the destination
    merged = db.merge_nodes(src)

    print(f"[merge_node_db] merged {merged} node rows from {src} into {dest}")
    counts = Counter(n.gpu for n in db.iter_nodes())
    print("[merge_node_db] node rows per card now:")
    for gpu, count in sorted(counts.items()):
        print(f"    {gpu or '(unknown card)'}: {count}")
    db.close()


if __name__ == "__main__":
    sys.exit(main())
