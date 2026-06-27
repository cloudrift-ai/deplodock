---
name: collect-node-data
description: Use this skill when the user wants to populate or refresh the autotune DB's node table (the cross-hardware search-tree node store) with data measured on a SPECIFIC GPU — e.g. "collect node data for an H200", "tune the goldens on a rented <GPU> and merge the nodes back", "gather cross-hardware node-store data", "populate / update the node table from <hardware>", "run the golden node sweep on a remote <GPU>". Rents a fresh single-GPU server (via start-remote-server; --billing-exempt for CloudRift), rsyncs + sets up deplodock there, runs `deplodock tune --dataset golden`, then merges the remote node rows into the local `~/.cache/deplodock/autotune.db` (nodes table only, keep-min, GPU-keyed so cards never collide), and tears the server down.
version: 0.1.0
---

# Collect node-store data from specific hardware

The autotune `node` table (`SearchDB`, default `~/.cache/deplodock/autotune.db`) is a **cross-hardware** dataset of
search-tree value-of-position rows — every partial branch + leaf of each per-kernel search, with the full feature dict
the prior sees. It is read by `deplodock eval prior --dataset nodes` (per-card fork sibling-ranking + leaf reachability)
and feeds prior diagnostics. Because your dev box (and most of the fleet) has no local CUDA GPU, the data for any given
card must be **measured on that card** and brought back.

This skill does exactly that for one GPU: rent it → set up deplodock → `deplodock tune --dataset golden` → merge the new
card's node rows into the local DB → tear the server down.

**Why merging into the single local DB is safe (read this once).** The node store is keyed by GPU:
`node_key = digest(context_key, gpu, op_sig, tunable-knobs)` and the `node` table carries a `gpu` column
(`Context.hardware_id()` — the canonicalized PCIe product name). Different cards therefore **never collide** — keep-min
only collapses rows *within one card*. So accumulating many GPUs' node rows in the one canonical
`~/.cache/deplodock/autotune.db` is the intended design (that is what makes it a cross-hardware dataset). Do **not** keep
per-GPU DB files.

**Scope: nodes table only.** This skill copies back the `node` table and nothing else (not `perf`, `cuda_op`,
`lowering`, or the learned `prior.json`). Those are needed only for `--dataset db` / greedy replay, out of scope here.

The golden tune is ~30–45 min (every recorded golden shape; the matmul/reduce/pointwise snippets are pure `torch.randn`,
hardware-independent, so they yield valid node rows for whatever card is rented). It needs **no `HF_TOKEN` and downloads
no models** — but it **does** need `nvcc` (it compiles CUDA kernels). Budget ~45–60 min total including env build.

## Inputs to confirm

Ask only for what the user hasn't already given:

1. **GPU model** — must map to a key in `deplodock/hardware.py::GPU_INSTANCE_TYPES` (e.g. "H200" → `"NVIDIA H200
   141GB"`). If it isn't in the table, stop and say so — don't guess. This card's identity is what the node rows are
   keyed by.
2. **Provider** — only ask if the GPU is offered by more than one (e.g. H200 is on CloudRift and GCP). A user-named
   provider is **binding** (never silently substitute).
3. **Env file** (CloudRift creds) — default `.env`; if the user named an overlay or wants a non-default cluster, follow
   the `start-remote-server` sourcing rules (base first, overlay second, same Bash call). H200 on CloudRift often needs
   a non-default cluster.
4. **GPU count is fixed at 1.** Node data is per-card; one GPU is all the tune needs. Don't rent more.

`--billing-exempt`: for **CloudRift** rentals, pass it — the user has standing authorization for this workflow. It is
silently dropped for GCP candidates. (This is the one place the usual "ask before billing-exempt" rule is pre-answered;
honor an explicit override if the user gives one this run.)

## Step 1 — Provision the server (delegate to `start-remote-server`)

Provision exactly one GPU using the orchestrator, following the full `start-remote-server` skill (credential sourcing,
candidate fallback, capacity handling, the binding `--provider` rule). The command shape:

```bash
[ -f .env ] && set -a && . ./.env && set +a && \
deplodock vm create gpu --gpu "<full GPU name>" --gpu-count 1 [--provider cloudrift|gcp] [--billing-exempt]
```

Capture from the final `VM ready at <user@host[:port]>` line:

- `REMOTE` — `user@host` (and the port, if any)
- the teardown handle (`--instance-id <id>` for CloudRift, `--instance <name> --zone <zone>` for GCP)

Do not wrap the command in a retry loop — the orchestrator handles fallback itself.

## Step 2 — Set up deplodock on the remote

Use the SSH options the codebase uses (`deplodock/provisioning/ssh_transport.py`). Define once:

```bash
REMOTE="<user@host>"
KEY="$HOME/.ssh/id_ed25519"
# The harness shell is zsh, which does NOT word-split a plain string var — so pass the SSH options as an
# ARRAY (expands correctly in both bash and zsh). A plain SSHOPTS="-o …" string fails under zsh with
# `keyword stricthostkeychecking extra arguments at end of line`.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -i "$KEY")
# Non-22 port: append `-p <PORT>` to SSH_OPTS for ssh; for scp/rsync pass `-P <PORT>` (scp's flag) separately.
# zsh gotcha: never name a helper var `status` or `path` (read-only/reserved in zsh) in any poll/waiter loop.
```

Invoke ssh as `ssh "${SSH_OPTS[@]}" "$REMOTE" …` and rsync as `rsync … -e "ssh ${SSH_OPTS[*]}" …` below.

**1. Sanity-check the toolchain** (driver + nvcc + python):

```bash
ssh "${SSH_OPTS[@]}" "$REMOTE" 'nvidia-smi -L; (command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null); python3 --version'
```

- **`nvcc` missing?** CloudRift / GCP images ship the NVIDIA driver + CUDA 12.9, but the CloudRift image may carry only
  the runtime. If `nvcc` isn't found, either it's under `/usr/local/cuda/bin` (just export the PATH, below) or install
  the toolkit matching the driver: `sudo apt-get update -qq && sudo apt-get install -y -qq cuda-toolkit-12-9`.
- **Always install the Python 3.12 venv + dev packages before `make setup`** — do NOT gate this on the version check.
  Even when `python3` *is* already 3.12, the CloudRift image ships it **without** the venv module (so `make setup` dies
  with `ensurepip is not available`, exit 127) and **without** the dev headers (so the cppyy wheel build fails with
  `fatal error: Python.h: No such file or directory`, exit 1). Install both up front (add `python3.12` itself only if
  `python3 --version` isn't 3.12):
  ```bash
  ssh "${SSH_OPTS[@]}" "$REMOTE" 'sudo apt-get update -qq && sudo apt-get install -y -qq python3.12-venv python3.12-dev'
  ```
  If a previous `make setup` already created a half-built `venv/`, `rm -rf ~/deplodock/venv` before re-running it.

**2. Rsync the working tree and set up the venv.** Rsync (not clone) so the remote runs the **exact local code**,
including any uncommitted changes:

```bash
REPO="$(git rev-parse --show-toplevel)"   # repo root, wherever this skill runs from
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  --exclude venv --exclude .git --exclude recipes --exclude _tune --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO/" "$REMOTE:~/deplodock/"
ssh "${SSH_OPTS[@]}" "$REMOTE" 'cd ~/deplodock && export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda && make setup'
```

`make setup` is a no-op if `venv/` already exists and installs `.[dev]` (CUDA torch, cupy-cuda12x, cppyy, catboost, …).
It installs **no** system packages — that's why Step 2.1 ensures the `python3.12-venv`/`-dev` packages + `nvcc` first.

## Step 3 — Run the golden tune (detached) and wait

Run it detached with a logfile so it survives a shaky SSH link:

```bash
ssh "${SSH_OPTS[@]}" "$REMOTE" '
  cd ~/deplodock
  export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  nohup ./venv/bin/deplodock tune --dataset golden > ~/tune.log 2>&1 &
  echo "tune_pid=$!"
'
```

Notes:

- **No `--clean`** — the VM is fresh, nothing to clean.
- **No `DEPLODOCK_TUNE_DB`** — let it write the default `~/.cache/deplodock/autotune.db`; Step 4 reads that path.
- tune compiles at `-Xcicc -O1` (a fast *ranking* pass) and re-benches near-best configs at `-O3`; both are recorded as
  node rows. We want the whole search tree, not just deployable winners.

**Poll until it finishes** (~30–45 min). Each poll is its **own short ssh** — do NOT hold a single ssh session open
running a multi-minute `for`/`sleep` loop: long-lived sessions drop with `client_loop: send disconnect: Broken pipe`
(exit 255). Re-poll every few minutes from the parent loop instead:

```bash
# Note the `[d]` bracket trick: a plain `pgrep -f "deplodock tune"` ALSO matches the pgrep wrapper's own
# command line (it contains the string "deplodock tune"), so it reports the tune as alive forever — even long
# after it finished. `"[d]eplodock tune"` matches the real process but not the literal pattern string.
ssh "${SSH_OPTS[@]}" "$REMOTE" 'tail -n 25 ~/tune.log; echo "---"; pgrep -af "[d]eplodock tune" || echo "TUNE DONE"'
```

Re-poll every few minutes. Done = `TUNE DONE` **and** the log ends with the tune summary (no traceback). If in doubt,
cross-check with `ssh "${SSH_OPTS[@]}" "$REMOTE" 'ps -C python3.12 -o pid,etime,cmd; nvidia-smi --query-compute-apps=pid
--format=csv,noheader'` — an empty process list + idle GPU confirms the tune actually exited. An `nvcc`/`CUDA_HOME`
error means the toolchain export from Step 2 didn't take — fix and re-run Step 3.

## Step 4 — Copy the node rows back and merge into the local DB

Use the merge helper — it snapshots the remote DB (WAL-safe `VACUUM INTO`), scps it back, and merges **only** the `node`
table into the local canonical DB with keep-min semantics (a remote row wins only when strictly faster for the *same*
`(gpu, op, knobs)` — which, cross-card, never happens, so other cards' rows are untouched):

```bash
./venv/bin/python scripts/merge_node_db.py --remote "$REMOTE"   # add --ssh-key / --port if non-default
```

(For an already-fetched local snapshot, use `--src <file.db>` instead of `--remote`. Override the destination with
`--db <path>` if not the default tune DB.)

It prints a **per-card receipt** (`node rows per card now: …`): the rented card should appear as its own line (e.g.
`NVIDIA H200 141GB: <n>`) and the counts of cards already present should be unchanged.

## Step 5 — Verify

```bash
./venv/bin/deplodock eval prior --dataset nodes
```

`node_report` groups by card — confirm a block headed with the rented GPU's name appears (`[<gpu>] <n> nodes`, with fork
sibling-ranking + leaf reachability for that card). `--kernel matmul` / `reduce` / `pointwise` narrows to one op family.

## Step 6 — Tear down the server

The VM bills until deleted, and this skill's job is done once the merge verifies. **Confirm with the user, then delete**
(never delete a VM without an explicit go-ahead):

```bash
deplodock vm delete cloudrift --instance-id <id>           # CloudRift
deplodock vm delete gcp --instance <name> --zone <zone>    # GCP
```

If the user wants to keep the box for more tuning, leave it up and report the SSH target + the exact teardown command.

## Verification checklist

Before reporting success:

- [ ] `vm create gpu` exited 0 and an SSH target was captured (and `--billing-exempt` was passed for CloudRift).
- [ ] Remote `nvcc` resolved and `make setup` completed without error.
- [ ] `~/tune.log` ended with the tune summary (no traceback); the `deplodock tune` process exited.
- [ ] The merge printed the rented card as its own per-card line; counts for other cards are unchanged.
- [ ] `eval prior --dataset nodes` shows a block for the rented GPU.
- [ ] The VM was deleted (or the user explicitly chose to keep it, and has the teardown command).

If any check fails, report the failure + raw output instead of claiming success.

## Common mistakes to avoid

- **Don't keep per-GPU DB files.** The node key includes `gpu`, so the single `~/.cache/deplodock/autotune.db` is the
  correct cross-hardware accumulator. Splitting it defeats the design and breaks the per-card `eval` views.
- **Don't `scp` over the local DB / merge the whole DB.** That clobbers other cards' node rows (and the unrelated
  `perf`/`cuda_op`/`lowering` tables). Use `scripts/merge_node_db.py` — keep-min, `node` table only.
- **Don't forget the `nvcc` PATH/CUDA_HOME export** in both `make setup` and the tune invocation — without it kernel
  compiles hard-fail (there is no NVRTC fallback).
- **Don't set `HF_TOKEN` or expect model downloads** — the golden tune is pure torch snippets.
- **Don't rent more than 1 GPU** — node data is per-card; extra GPUs just burn money.
- **Don't auto-delete the VM** — confirm teardown with the user first (and never modify a CloudRift server beyond the
  tune we explicitly started).
- **Don't trust a plain `pgrep -f "deplodock tune"`** — it self-matches the wrapper's own argv and reports the tune
  alive forever. Use the `[d]eplodock tune` bracket trick (Step 3) and cross-check with `ps -C python3.12` + an idle GPU.
- **Don't pass SSH options as an unquoted string var** — the harness shell is zsh; use the `SSH_OPTS=(…)` array, and
  never name a helper var `status`/`path` (read-only in zsh — a waiter that does `status=$(…)` crashes outright).
