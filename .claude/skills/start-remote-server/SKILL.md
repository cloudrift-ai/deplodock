---
name: start-remote-server
description: This skill should be used when the user asks to "start a remote server", "spin up a GPU VM", "provision a cloud GPU", "create a remote server with X GPU", "give me a server with Y GPUs", or otherwise wants a fresh cloud GPU VM provisioned (without immediately deploying a model) so it stays available for later benchmarks or inference work.
version: 0.2.0
---

# Start Remote Server

Provision a fresh cloud GPU VM on the right provider (GCP or CloudRift) for the requested hardware, leave it running, and report back the SSH connection details so the user can run `deplodock deploy ssh ...` or `deplodock bench --ssh ...` against it later.

This skill **does not** deploy a recipe. To deploy + bench + teardown in one shot, use `deplodock deploy cloud` or `deplodock bench` directly — those handle their own VM lifecycle.

## Preferred Path: `deplodock vm create gpu`

For almost every request, use the GPU-name orchestrator command:

```bash
deplodock vm create gpu \
  --gpu "<full GPU name>" \
  --gpu-count <N> \
  [--provider cloudrift|gcp] \
  [--name <prefix>] \
  [--ssh-key ~/.ssh/id_ed25519] \
  [--config config.yaml] \
  [--billing-exempt]    # CloudRift admin-only, see below
```

What this command does that the provider-specific commands do **not**:

- Resolves the GPU name to candidates from `deplodock/hardware.py::GPU_INSTANCE_TYPES` automatically — you do not pass an instance type.
- Iterates candidates in preference order (every `(provider, base)` pair under the GPU key, all GCP zones from `GPU_GCP_ZONES`).
- Handles capacity fallback: on `CapacityExhausted` (HTTP 5xx, 409, "no capacity", "out of stock", "Service Unavailable") it advances to the next candidate. On transient errors it retries the same candidate.
- Cleans up orphans from failed attempts.
- Reads `config.yaml` for provider defaults (GCP image/disk/service account, CloudRift image override, etc.).
- Stops on terminal errors (auth 401/403, malformed 400, unknown instance type 404) without burning the candidate list.

**`--ssh-key` is the PRIVATE key path** for this command (default `~/.ssh/id_ed25519`). The orchestrator appends `.pub` internally and also uses the private key to verify SSH reachability. This is different from `vm create cloudrift` (see Manual Overrides below), which requires the `.pub` path explicitly. Do not pass `.pub` to `vm create gpu`.

**`--provider` is binding.** If the user named a provider, pass it. If the GPU isn't offered by that provider, the orchestrator raises `ValueError` with the available providers — surface that to the user; do not silently substitute.

## Inputs to Confirm

Before running anything, confirm these with the user (ask only the ones not already given):

1. **GPU model** — must match a key in `deplodock/hardware.py::GPU_INSTANCE_TYPES`. If the user gives a short name ("5090", "H200"), map it to the full name (`"NVIDIA GeForce RTX 5090"`, `"NVIDIA H200 141GB"`, etc.). If the requested GPU is not in `GPU_INSTANCE_TYPES`, stop and tell the user — don't guess.
2. **GPU count** — integer (1, 2, 4, 8). Default to 1 if not specified and the user is just experimenting.
3. **Provider** — only ask if the GPU is offered by more than one provider (e.g. H200 is on both CloudRift and GCP). Otherwise omit `--provider` and let the hardware-table preference order apply.
   - **If the user explicitly named a provider, it is binding.** Pass `--provider` and never silently fall back to another provider. If the GPU isn't available on that provider, report the mismatch — let the user decide whether to switch GPU, switch provider, or abort.
4. **Server name** — optional `--name <prefix>`. Embedded into the VM hostname; only matters for GCP labelling. Skip unless the user has a preference.
5. **SSH private key path** — default `~/.ssh/id_ed25519`. Confirm the `.pub` counterpart exists (the orchestrator reads it for upload).
6. **Billing-exempt rental?** (CloudRift only) — explicitly ask the user whether to add `--billing-exempt`. This flag skips CloudRift billing and is admin-only; it is not a default. Only pass it if the user confirms they have an admin/no-cost agreement. Do not infer from "free", "test", or "cheap" — require an explicit yes. The flag is silently dropped when GCP candidates are chosen.

## Sourcing Provider Credentials

`deplodock vm create gpu` reads `CLOUDRIFT_API_KEY` / `CLOUDRIFT_API_URL` from the process environment. The repo does **not** auto-load env files (no `python-dotenv` dependency). The repo root typically has a base `.env`; the user may also keep `.env.*` files that point at different clusters / accounts.

Pick the file based on what the user said:

- **Default** → source plain `.env` only.
- **User specified an additional env file** (e.g. they named one or pointed at `.env.<something>`) → source `.env` first as the base, then overlay the user-specified file so its values win on conflict. Use the filename the user gave verbatim — never guess one from context.

If the user wants a non-default cluster but didn't say which file, **list candidates** (`ls .env.* 2>/dev/null`) and ask. Don't guess.

Source in the **same Bash call** as `deplodock vm create gpu` (Bash sessions don't preserve env across calls). Order matters: base first, overlay second.

Default (just `.env`):

```bash
[ -f .env ] && set -a && . ./.env && set +a && deplodock vm create gpu --gpu "<name>" --gpu-count <N>
```

With a user-specified overlay file `<extra>`:

```bash
set -a && [ -f .env ] && . ./.env; . ./<extra> && set +a && deplodock vm create gpu --gpu "<name>" --gpu-count <N>
```

The `[ -f .env ] && . ./.env;` part lets the base file be optional but fails loudly if the overlay is missing — that's intentional, since a missing user-specified file usually means you're about to hit the wrong target.

**Sanity check** before running the command — never echo the secret value. Use a present/absent check, not parameter expansion of the value:

```bash
[ -n "$CLOUDRIFT_API_URL" ] && echo "CLOUDRIFT_API_URL: set (override)" || echo "CLOUDRIFT_API_URL: default"
[ -n "$CLOUDRIFT_API_KEY" ] && echo "CLOUDRIFT_API_KEY: set" || echo "CLOUDRIFT_API_KEY: MISSING"
```

Do **not** use `${CLOUDRIFT_API_KEY:+set}${CLOUDRIFT_API_KEY:-MISSING}` — when the var is set, `${VAR:-MISSING}` still expands to the **value** (the default only applies when unset), so you get `set<actual-secret>` printed.

If a user-specified overlay was meant to redirect the cluster but `CLOUDRIFT_API_URL` shows as default, you sourced the wrong file (or didn't source the overlay at all) — fix and re-check before running. `.env*` files are gitignored — never `git add` them.

**H200 on CloudRift typically requires a non-default cluster.** The public `api.cloudrift.ai` may not return H200 capacity. If the user asks for H200 and the chosen candidate is CloudRift, confirm which env file to use before running.

For GCP, the project comes from `gcloud config` (no `--project` flag). If `gcloud auth` is not active the command will fail; suggest the user run `! gcloud auth login` (the `!` prefix runs it in their session).

## Capacity Failures

The orchestrator handles capacity fallback automatically: it advances to the next base in `GPU_INSTANCE_TYPES`, then the next zone (GCP) or provider (only if the user did not pin `--provider`), retries transient errors on the same candidate, and surfaces a single end-of-run error if every candidate is exhausted.

You should **not** wrap the command in a manual retry loop. If it exits non-zero, read the final log line — it lists what was tried and the last error. Concrete next steps to offer the user:

- Wait and retry later (capacity often returns within minutes).
- Drop `--billing-exempt` (CloudRift) in case the rejection is account-side, not capacity.
- Switch GPU model or provider.

Terminal errors (auth 401/403, malformed 400, "Unknown GPU" `ValueError`, missing instance type 404) are surfaced immediately as `TerminalProvisionError` and won't be retried — don't suggest waiting; fix the input or credentials.

## Manual Overrides

Use the provider-specific commands **only** when the user explicitly asks for a specific instance type / machine type that doesn't match a hardware-table entry, or when debugging a provisioning bug. For normal "give me a server" requests, always prefer `vm create gpu`.

### CloudRift (single-shot, exact instance type)

```bash
deplodock vm create cloudrift \
  --instance-type <base>.<gpu_count> \
  --ssh-key ~/.ssh/id_ed25519.pub \
  [--billing-exempt]
```

This command requires the **public** key path (`.pub`), unlike `vm create gpu`. It does not retry across bases/zones — one attempt, then exit.

Image is auto-selected from the instance type: `mi*` (AMD Instinct) → ROCm image; everything else → NVIDIA driver image. Override with `--image-url` only when the user explicitly asks — the wrong image leaves the GPU unusable.

### GCP (single-shot, exact machine type)

```bash
deplodock vm create gcp \
  --instance <name> \
  --zone <zone> \
  --machine-type <resolved_type> \
  --provisioning-model <FLEX_START|SPOT|STANDARD> \
  --wait-ssh
```

Always pass `--wait-ssh` so the VM is genuinely ready when the command returns. Resolve the zone from `GPU_GCP_ZONES` (fall back to `DEFAULT_GCP_ZONE = us-central1-b`) and the provisioning model from `GPU_GCP_PROVISIONING_MODEL` (default `FLEX_START`; Pro 6000 Server Edition uses `SPOT`).

## After Creation

When the command succeeds (last log line `VM ready at <user@host>:<port>` for `vm create gpu`):

1. Capture the SSH connection target from the command output. For `vm create gpu` and `vm create cloudrift`, it's printed as `user@ip` / `user@host:port`. For `vm create gcp` manual mode, use `gcloud compute ssh <instance> --zone <zone>` or pull the external IP from `gcloud compute instances describe`.
2. Report to the user: provider, instance ID/name, zone (GCP), instance type, SSH target.
3. Remind the user that the VM is **billed until deleted**. Provide the matching teardown command:
   - `deplodock vm delete cloudrift --instance-id <id>`
   - `deplodock vm delete gcp --instance <name> --zone <zone>`
4. Offer next steps:
   - Deploy a recipe: `deplodock deploy ssh --recipe <path> --ssh <user@host>`
   - Run benchmarks against it: `deplodock bench <recipes> --ssh <user@host>`

Do **not** automatically schedule a teardown — let the user decide when to release the VM. If the user explicitly asks for a deadline, offer to `/schedule` a delete agent for that time.

## Verification Checklist

Before reporting success, verify:

- [ ] The create command exited 0 (not just printed something).
- [ ] An SSH connection target was reported (CloudRift / `vm create gpu`) or `--wait-ssh` confirmed reachability (GCP manual).
- [ ] If the user named a provider, the chosen provider matches it exactly (no fallback substitution).
- [ ] If `--billing-exempt` was passed, the user explicitly confirmed it.

If any check fails, report the failure and the raw output instead of claiming success.

## Common Mistakes to Avoid

- Don't pass `.pub` to `vm create gpu --ssh-key` — that command takes the **private** key path and appends `.pub` itself. (The provider-specific `vm create cloudrift` is the opposite: it requires `.pub`.)
- Don't pass `--instance-type` / `--machine-type` to `vm create gpu` — it derives them from the hardware table.
- Don't wrap `vm create gpu` in a manual retry loop or try other bases yourself — the orchestrator already does both, and a second wrapper just hides the real error.
- Don't override a user-specified provider. If the GPU isn't available on that provider, abort and ask — never quietly run on a different provider "because that's where it's available."
- Don't set `--dry-run` when the user asked for a real server. Use it only when the user asks to preview.
- Don't run `deplodock deploy cloud` for this task — that bundles deploy+teardown and won't leave a server idle for the user.
- Don't `git add` `.env*` files — they're gitignored for a reason.
