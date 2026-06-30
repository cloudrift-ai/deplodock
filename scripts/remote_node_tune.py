#!/usr/bin/env python
"""Drive the setup + golden tune on a remote GPU host, then wait for it — the
token-heavy middle of the ``collect-node-data`` skill, extracted so the agent makes
ONE backgrounded tool call instead of ~20 verbose ssh polls.

Given an SSH target it: ensures the Python 3.12 venv/dev packages + ``nvcc``, rsyncs
the local working tree, runs ``make setup`` (output to a remote logfile — only a tail
comes back on failure), launches ``deplodock tune --dataset golden`` detached, then
**polls the remote log internally** until the tune finishes. All the per-poll ssh
chatter happens inside this process, so it never reaches the agent's context.

Run it from the agent in the BACKGROUND (Bash ``run_in_background: true``) — the tune
takes ~30–45 min, well past a foreground tool timeout; the harness re-invokes the
agent with just the final summary when this exits.

    ./venv/bin/python scripts/remote_node_tune.py --remote user@host \
        [--ssh-key ~/.ssh/id_ed25519] [--port 57006] [--repo /path/to/deplodock] \
        [--poll 60] [--timeout 7200]

Then (separately) merge the node rows home with ``scripts/merge_node_db.py --remote``.

Robustness baked in (the four traps the first run hit): all ssh goes through argv
lists (no shell word-splitting / zsh quirks); liveness uses the ``[d]eplodock tune``
bracket-pgrep so it never self-matches; each poll is its own short ssh (no long-held
session that broken-pipes); and venv/dev are always installed before ``make setup``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from deplodock.provisioning.ssh_transport import REMOTE_DEPLOY_DIR

# Matches deplodock/provisioning/ssh_transport.py::ssh_base_args.
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

_CUDA_EXPORT = "export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda"
# Nest under the repo's established remote layout (REMOTE_DEPLOY_DIR = ~/.local/share/deplodock) —
# the same base the deploy/bench paths use (they already clone a repo under it).
_BASE = f"{REMOTE_DEPLOY_DIR}/node-tune"
_REMOTE_DIR = f"{_BASE}/repo"  # rsync target — the deplodock checkout we tune from
_SETUP_LOG = f"{_BASE}/setup.log"
_TUNE_LOG = f"{_BASE}/tune.log"
_DONE_RE = re.compile(r"done: (\d+)/(\d+) shape")


def _opts(ssh_key: str | None, port: int | None) -> list[str]:
    out = list(_SSH_OPTS)
    if ssh_key:
        out += ["-i", ssh_key]
    if port:
        out += ["-p", str(port)]  # rsync/ssh use -p (scp would use -P, not relevant here)
    return out


def _run(remote: str, ssh_key: str | None, port: int | None, command: str, *, timeout: int = 600, detach: bool = False) -> tuple[int, str]:
    """Run one remote command over a fresh ssh connection; return (rc, combined output).

    ``detach=True`` adds ssh ``-n`` (stdin from /dev/null) — needed when launching a
    remote background process: without it (plus a ``< /dev/null`` redirect on the
    command itself) ssh holds the channel open past a ``nohup … &``, so this call
    times out (rc 124) long *after* the process already started successfully."""
    argv = ["ssh", *(["-n"] if detach else []), *_opts(ssh_key, port), remote, command]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "(ssh command timed out)"


def _log(msg: str) -> None:
    print(f"[remote_node_tune] {msg}", flush=True)


def _fail(remote: str, ssh_key: str | None, port: int | None, why: str, logfile: str) -> int:
    _, tail = _run(remote, ssh_key, port, f"tail -n 40 {logfile} 2>/dev/null")
    print("\n=== remote_node_tune summary ===", flush=True)
    print(f"status: FAILED ({why})", flush=True)
    print(f"--- last 40 lines of {remote}:{logfile} ---", flush=True)
    print(tail.strip() or "(empty / not found)", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", required=True, help="SSH target, user@host")
    ap.add_argument("--ssh-key", help="SSH private key (default: ssh's own default)")
    ap.add_argument("--port", type=int, help="SSH port (default 22)")
    ap.add_argument("--repo", help="Local repo root to rsync (default: git toplevel)")
    ap.add_argument("--poll", type=int, default=60, help="Seconds between completion polls (default 60)")
    ap.add_argument("--timeout", type=int, default=7200, help="Max seconds to wait for the tune (default 7200)")
    ap.add_argument("--no-merge", action="store_true", help="Skip the folded-in node-row merge after a successful tune")
    args = ap.parse_args()

    remote, key, port = args.remote, args.ssh_key, args.port
    repo = args.repo
    if not repo:
        gp = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if gp.returncode != 0:
            print("error: not inside a git repo and --repo not given; pass --repo <path>", flush=True)
            return 2
        repo = gp.stdout.strip()

    # 1. make the remote base dir (so the logs + rsync target exist), then deps: always
    #    install the venv/dev packages (the image ships 3.12 without them); install the CUDA
    #    toolkit only if nvcc is genuinely absent. All noisy output → the setup log.
    _log(f"ensuring deps on {remote} (apt; output to {_SETUP_LOG}) ...")
    deps = (
        f"mkdir -p {_BASE}; "
        f"sudo apt-get update -qq >> {_SETUP_LOG} 2>&1; "
        f"sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >> {_SETUP_LOG} 2>&1; "
        "if ! command -v nvcc >/dev/null 2>&1 && ! ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; then "
        f"sudo apt-get install -y -qq cuda-toolkit-12-9 >> {_SETUP_LOG} 2>&1; fi; "
        "if command -v nvcc >/dev/null 2>&1 || ls /usr/local/cuda*/bin/nvcc >/dev/null 2>&1; "
        "then echo NVCC_OK; else echo NVCC_MISSING; fi"
    )
    rc, out = _run(remote, key, port, deps, timeout=1800)
    if rc != 0 or "NVCC_OK" not in out:
        # rc==0 with NVCC_OK absent means the apt steps ran but nvcc is still missing
        # (the chain ends in an echo, so rc reflects only that last command); rc!=0 means
        # the ssh/apt command itself failed. The setup-log tail below carries the detail.
        reason = "nvcc not found after toolkit install" if rc == 0 else f"ssh/apt rc={rc}"
        return _fail(remote, key, port, f"deps/nvcc ({reason})", _SETUP_LOG)

    # 2. rsync the working tree (exact local code, incl. uncommitted changes).
    _log(f"rsyncing {repo} -> {remote}:{_REMOTE_DIR} ...")
    e = "ssh " + " ".join(_opts(key, port))
    rsync = [
        "rsync",
        "-az",
        "-e",
        e,
        "--exclude",
        "venv",
        "--exclude",
        ".git",
        "--exclude",
        "recipes",
        "--exclude",
        "_tune",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.pyc",
        f"{repo}/",
        f"{remote}:{_REMOTE_DIR}/",
    ]
    rp = subprocess.run(rsync, capture_output=True, text=True)
    if rp.returncode != 0:
        _log(f"rsync failed: {(rp.stderr or rp.stdout).strip()}")
        return 1

    # 3. make setup (output → the setup log); on failure rebuild a clean venv once.
    _log(f"running make setup (output to {_SETUP_LOG}) ...")
    setup = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && make setup >> {_SETUP_LOG} 2>&1 && echo SETUP_OK || echo SETUP_FAIL"
    rc, out = _run(remote, key, port, setup, timeout=2400)
    if "SETUP_OK" not in out:
        _log("make setup failed; wiping venv and retrying once ...")
        setup2 = f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && rm -rf venv && make setup >> {_SETUP_LOG} 2>&1 && echo SETUP_OK || echo SETUP_FAIL"
        rc, out = _run(remote, key, port, setup2, timeout=2400)
        if "SETUP_OK" not in out:
            return _fail(remote, key, port, "make setup", _SETUP_LOG)

    # 4. launch the tune detached. Redirect ALL three std streams away from the ssh
    #    channel (`< /dev/null` + `> <tune log> 2>&1`) and use ssh -n (detach=True),
    #    else ssh holds the channel open past the `&` and this call times out *after*
    #    a successful launch.
    _log(f"launching `deplodock tune --dataset golden` (detached -> {_TUNE_LOG}) ...")
    launch = (
        f"cd {_REMOTE_DIR} && {_CUDA_EXPORT} && "
        f"nohup ./venv/bin/deplodock tune --dataset golden > {_TUNE_LOG} 2>&1 < /dev/null & echo tune_launched"
    )
    rc, out = _run(remote, key, port, launch, timeout=60, detach=True)
    if "tune_launched" not in out:
        # Belt-and-suspenders: if the launch ssh still didn't confirm, the tune may
        # have started anyway — verify the process before declaring failure.
        time.sleep(5)
        _, chk = _run(remote, key, port, "pgrep -f '[d]eplodock tune' >/dev/null 2>&1 && echo ALIVE || echo DEAD")
        if "ALIVE" not in chk:
            return _fail(remote, key, port, f"launch (rc={rc})", _TUNE_LOG)
        _log("launch ssh did not confirm but the tune process is alive — continuing")

    # 5. poll the remote log internally until done / dead / timeout. The `[d]eplodock tune`
    #    bracket pattern is what stops pgrep from matching this very poll command's own argv.
    poll = (
        f"echo \"DONE_MARK=$(grep -aoE 'done: [0-9]+/[0-9]+ shape' {_TUNE_LOG} 2>/dev/null | tail -1)\"; "
        "pgrep -f '[d]eplodock tune' >/dev/null 2>&1 && echo PROC=ALIVE || echo PROC=DEAD"
    )
    start = time.monotonic()
    dead_streak = 0
    while True:
        if time.monotonic() - start > args.timeout:
            return _fail(remote, key, port, f"timeout after {args.timeout}s", _TUNE_LOG)
        time.sleep(args.poll)
        rc, out = _run(remote, key, port, poll)
        # The poll always echoes a ``PROC=`` marker when it actually ran on the remote;
        # its absence means the ssh itself failed (timeout / connection reset), which is
        # NOT evidence the tune died — only a confirmed ``PROC=DEAD`` is.
        reachable = "PROC=" in out
        done_mark = next((ln[len("DONE_MARK=") :] for ln in out.splitlines() if ln.startswith("DONE_MARK=")), "")
        alive = "PROC=ALIVE" in out
        m = _DONE_RE.search(done_mark)
        if m:
            shapes = f"{m.group(1)}/{m.group(2)}"
            elapsed = int(time.monotonic() - start)
            _, bf = _run(remote, key, port, f"grep -aEc 'bench_fail|bench worker exceeded' {_TUNE_LOG} 2>/dev/null || echo 0")
            bench_fails = (bf.strip().splitlines() or ["0"])[0]
            print("\n=== remote_node_tune summary ===", flush=True)
            print("status: ok", flush=True)
            print(f"shapes: {shapes}", flush=True)
            print(f"bench_fails: {bench_fails}", flush=True)
            print(f"elapsed (wait only): {elapsed}s", flush=True)
            print(f"remote log: {remote}:{_TUNE_LOG}", flush=True)
            manual = (
                f"./venv/bin/python scripts/merge_node_db.py --remote {remote}"
                + (f" --ssh-key {key}" if key else "")
                + (f" --port {port}" if port else "")
            )
            if args.no_merge:
                print(f"merge: skipped (--no-merge); run `{manual}`", flush=True)
                return 0
            # Fold the merge in so the single backgrounded call ends with the node rows
            # already in the local DB and the per-card receipt printed — no separate step
            # for the caller to remember to launch or to report on.
            print("\n--- merging node rows into the local DB ---", flush=True)
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import merge_node_db  # noqa: PLC0415  (sibling script in scripts/)

                merged = merge_node_db.fetch_and_merge(remote, ssh_key=key, port=port)
            except Exception as exc:  # noqa: BLE001 — report any merge failure, don't crash the run
                print(f"merge FAILED: {exc!r}; tune data is safe on {remote} — re-run `{manual}`", flush=True)
                return 1
            if merged == 0:
                # The tune finished but nothing came home — the remote wrote its node store
                # somewhere other than the snapshot's default path (e.g. a remote
                # DEPLODOCK_TUNE_DB), so don't claim success.
                print(
                    f"merge brought back 0 node rows — the remote node store wasn't at the expected "
                    f"path; tune data is on {remote}. Check the remote DEPLODOCK_TUNE_DB and re-run "
                    f"`{manual} --remote-db <path>`.",
                    flush=True,
                )
                return 1
            print("\nstatus: COMPLETE (tune + merge done)", flush=True)
            return 0
        if not reachable:
            # Transient ssh/network failure this cycle — keep waiting (still bounded by
            # --timeout); don't mistake an unreachable host for a crashed tune.
            _log(f"poll ssh failed (rc={rc}) — host unreachable this cycle, retrying")
            continue
        if not alive:
            dead_streak += 1
            if dead_streak >= 2:  # two consecutive confirmed DEADs without a done marker ⇒ it crashed
                return _fail(remote, key, port, "tune process died without a done marker", _TUNE_LOG)
        else:
            dead_streak = 0
            _log(f"tuning... ({int(time.monotonic() - start)}s elapsed)")


if __name__ == "__main__":
    sys.exit(main())
