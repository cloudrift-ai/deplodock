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

## Step 2 — Set up + tune on the remote (one backgrounded script)

`scripts/remote_node_tune.py` does the whole token-heavy middle in **one process**: ensures the Python 3.12 venv/dev
packages + `nvcc`, rsyncs your working tree (exact local code, incl. uncommitted changes), runs `make setup` (output to
the remote `~/setup.log` — only a tail returns on failure), launches `deplodock tune --dataset golden` detached, then
**polls the remote log internally** until it finishes. The four traps from the first run are baked in: argv-list ssh (no
zsh word-split), `[d]eplodock tune` bracket-pgrep (no self-match), one short ssh per poll (no broken-pipe), and venv/dev
always installed before `make setup`.

Run it in the **background** (Bash `run_in_background: true`) — the tune is ~30–45 min, past a foreground tool timeout,
and you only want the final summary in context, not ~20 ssh polls:

```bash
./venv/bin/python scripts/remote_node_tune.py --remote "<user@host>" --ssh-key ~/.ssh/id_ed25519 [--port <PORT>]
```

When it exits you get one compact summary:
- **success** → `status: ok`, `shapes: N/N`, `bench_fails: K`, elapsed, and the remote log path. (A `bench_fail` on a big
  shape like `square.4096` is expected — an 8 s bench-wall guard, non-fatal; `K` small is fine.) The tune wrote the
  default `~/.cache/deplodock/autotune.db` on the remote, ready for Step 3.
- **failure** → `status: FAILED (<phase>)` plus the last 40 lines of the relevant remote log (`~/setup.log` or
  `~/tune.log`). Fix and re-run — the script is idempotent (rsync + `make setup` no-op when already done).

**Manual debugging** (only if the script fails and you need to poke the box): the harness shell is zsh, so pass ssh
options as an **array** — `SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -i
"$HOME/.ssh/id_ed25519")` (a plain string var fails: zsh doesn't word-split it). Then `ssh "${SSH_OPTS[@]}" "$REMOTE"
'tail -n 40 ~/tune.log'`; check liveness with `pgrep -f "[d]eplodock tune"` (bracket trick — a plain pattern self-matches
the poll's own argv); never run a multi-minute loop inside one ssh session.

## Step 3 — Copy the node rows back and merge into the local DB

Use the merge helper — it snapshots the remote DB (WAL-safe `VACUUM INTO`), scps it back, and merges **only** the `node`
table into the local canonical DB with keep-min semantics (a remote row wins only when strictly faster for the *same*
`(gpu, op, knobs)` — which, cross-card, never happens, so other cards' rows are untouched):

```bash
./venv/bin/python scripts/merge_node_db.py --remote "<user@host>"   # add --ssh-key / --port if non-default
```

(For an already-fetched local snapshot, use `--src <file.db>` instead of `--remote`. Override the destination with
`--db <path>` if not the default tune DB.)

It prints a **per-card receipt** (`node rows per card now: …`): the rented card should appear as its own line (e.g.
`NVIDIA H200 141GB: <n>`) and the counts of cards already present should be unchanged.

## Step 4 — Verify

```bash
./venv/bin/deplodock eval prior --dataset nodes
```

`node_report` groups by card — confirm a block headed with the rented GPU's name appears (`[<gpu>] <n> nodes`, with fork
sibling-ranking + leaf reachability for that card). `--kernel matmul` / `reduce` / `pointwise` narrows to one op family.

## Step 5 — Tear down the server

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
- [ ] `remote_node_tune.py` printed `status: ok` with `shapes: N/N` (not `FAILED`).
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
- **Don't hand-roll the remote setup/poll loop** — `scripts/remote_node_tune.py` (Step 2) already handles it correctly:
  argv-list ssh (zsh doesn't word-split a string var), the `[d]eplodock tune` bracket-pgrep (a plain pattern self-matches
  the poll's own argv and reports the tune alive forever), one short ssh per poll, and venv/dev install. Only drop to
  manual ssh for debugging — and then keep the same precautions (array ssh opts; never name a var `status`/`path`, which
  are read-only in zsh).
