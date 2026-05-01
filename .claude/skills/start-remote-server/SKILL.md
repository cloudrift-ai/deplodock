---
name: start-remote-server
description: This skill should be used when the user asks to "start a remote server", "spin up a GPU VM", "provision a cloud GPU", "create a remote server with X GPU", "give me a server with Y GPUs", or otherwise wants a fresh cloud GPU VM provisioned (without immediately deploying a model) so it stays available for later benchmarks or inference work.
version: 0.1.0
---

# Start Remote Server

Provision a fresh cloud GPU VM on the right provider (GCP or CloudRift) for the requested hardware, leave it running, and report back the SSH connection details so the user can run `deplodock deploy ssh ...` or `deplodock bench --ssh ...` against it later.

This skill **does not** deploy a recipe. To deploy + bench + teardown in one shot, use `deplodock deploy cloud` or `deplodock bench` directly — those handle their own VM lifecycle.

## Inputs to Confirm

Before running anything, confirm these with the user (ask only the ones not already given):

1. **GPU model** — must match a key in `deplodock/hardware.py::GPU_INSTANCE_TYPES`. If the user gives a short name ("5090", "H200"), map it to the full name (`"NVIDIA GeForce RTX 5090"`, `"NVIDIA H200 141GB"`, etc.).
2. **GPU count** — integer (1, 2, 4, 8). Default to 1 if not specified and the user is just experimenting.
3. **Provider** — only ask if the GPU is offered by more than one provider (e.g. H200 is on both CloudRift and GCP). Otherwise pick the only provider.
   - **If the user explicitly named a provider, it is binding.** If the requested GPU is not listed under that provider in `GPU_INSTANCE_TYPES`, **stop and report the mismatch** — do not silently fall back to another provider. List the providers that actually offer the GPU and let the user decide whether to switch GPU, switch provider, or abort. Never substitute providers on the user's behalf.
4. **Instance name** (GCP only) — propose `deplodock-<gpu-short>-<count>g` if not given.
5. **SSH public key path** (CloudRift only) — default `~/.ssh/id_ed25519.pub`. Confirm it exists.
6. **Billing-exempt rental?** (CloudRift only) — explicitly ask the user whether to add `--billing-exempt`. This flag skips CloudRift billing and is admin-only; it is not a default. Only pass it if the user confirms they have an admin/no-cost agreement. Do not infer from "free", "test", or "cheap" — require an explicit yes. The flag does not exist on GCP, so never ask for it when the provider is GCP.

If the requested GPU is not in `GPU_INSTANCE_TYPES`, stop and tell the user — don't guess an instance type.

## Resolving the Instance Type

Read `deplodock/hardware.py` to confirm current values; do not trust this skill's snippets if they conflict with the source.

- **CloudRift**: `{base}.{gpu_count}` — e.g. `rtx59-7-50-400-ec.1`. Pick the first `(provider, base)` tuple under the GPU key unless the user requested otherwise. Multiple bases exist for some GPUs (different CPU/RAM/disk profiles); if the user has a preference, honor it.
- **GCP**: `{base}-{gpu_count}g` (most), `g4-standard-{gpu_count*48}` (Pro 6000 Server Edition), or the smallest available count ≥ requested for entries in `GCP_AVAILABLE_GPU_COUNTS` (e.g. `a4-highgpu` only offers 8). The `resolve_instance_type` function encodes this — mirror it, don't reinvent.
- **GCP zone**: look up in `GPU_GCP_ZONES`. Use the first listed zone. Fall back to `DEFAULT_GCP_ZONE` (`us-central1-b`) if the GPU is not listed.
- **GCP provisioning model**: look up in `GPU_GCP_PROVISIONING_MODEL`. Default is `FLEX_START`. (Pro 6000 Server Edition uses `SPOT`.)

## Capacity Fallback: Try Each Base In Order

When a `(GPU, provider)` key in `GPU_INSTANCE_TYPES` lists **multiple bases** with the same provider, treat the list as a capacity-fallback chain — different bases are different CPU/RAM/disk profiles, and CloudRift capacity often varies across them. The selected base may simply not be available right now even when the GPU is listed.

Procedure when the create command fails with a capacity-style error (HTTP 5xx, 409 Conflict, "no capacity", "out of stock", or "Service Unavailable") on the chosen base:

1. Move to the **next base** under the same `(GPU, provider)` key in `GPU_INSTANCE_TYPES` and retry with the same `gpu_count`.
2. Continue down the list until one base succeeds or all bases are exhausted.
3. Do **not** retry the same base immediately — capacity failures clear on a different base, not on the same one in the same second.
4. Do **not** silently switch to a different *provider* — the user-specified provider is still binding (see Inputs to Confirm #3).
5. Do **not** silently switch to a different *GPU model* (e.g. Workstation Edition → Max-Q) without asking — those are different hardware.

If every base in the list fails:

1. **Stop. Do not retry indefinitely.**
2. Report to the user: each base attempted, the exact error each returned, and the order they were tried.
3. Offer concrete next steps: wait and retry later, drop `--billing-exempt` (CloudRift) in case the rejection is account-side, switch GPU model, or switch provider.

Treat clearly non-capacity errors (auth: 401/403, malformed request: 400, missing instance type: 404 with a "not found" body) as terminal — those won't be fixed by trying another base. Stop and report immediately.

## Commands

### CloudRift

```bash
deplodock vm create cloudrift \
  --instance-type <base>.<gpu_count> \
  --ssh-key ~/.ssh/id_ed25519.pub \
  [--billing-exempt]    # ONLY if the user explicitly confirmed admin/no-cost
```

The command requires the **public** key path (`.pub`), not the private key. `CLOUDRIFT_API_KEY` must be set in the environment (or pass `--api-key`). The command waits up to 600s for `Active` status and prints the SSH connection on success.

**`--billing-exempt`** is an admin-only flag that skips CloudRift billing. Always confirm with the user before adding it; never set it implicitly. It is silently accepted by the API only for accounts authorized for no-cost rentals — passing it on a regular account will likely fail or be billed normally. The flag is CloudRift-specific and does not exist for GCP.

**Sourcing the API key and URL from env files:** deplodock does **not** auto-load env files (no `python-dotenv` dependency). The repo root typically has a base `.env`. The user may also keep additional `.env.*` files that point at different clusters / accounts.

Pick the file based on what the user said:

- **Default** → source plain `.env` only.
- **User specified an additional env file** (e.g. they named one or pointed at `.env.<something>`) → source `.env` first as the base, then overlay the user-specified file so its values win on conflict. Use the filename the user gave verbatim — never guess one from context.

If the user wants a non-default cluster but didn't say which file, **list candidates** (`ls .env.* 2>/dev/null`) and ask. Don't guess.

Source in the **same Bash call** as `deplodock vm create` (Bash sessions don't preserve env across calls). Order matters when overlaying: base first, overlay second.

Default (just `.env`):

```bash
[ -f .env ] && set -a && . ./.env && set +a && deplodock vm create cloudrift --instance-type <base>.<gpu_count> --ssh-key ~/.ssh/id_ed25519.pub
```

With a user-specified overlay file `<extra>` (e.g. `.env.local`, or whatever they named):

```bash
set -a && [ -f .env ] && . ./.env; . ./<extra> && set +a && deplodock vm create cloudrift --instance-type <base>.<gpu_count> --ssh-key ~/.ssh/id_ed25519.pub
```

The `[ -f .env ] && . ./.env;` part lets the base file be optional but fails loudly if the overlay is missing — that's intentional, since a missing user-specified file usually means you're about to hit the wrong target.

**Sanity check** before running `vm create` — never echo the secret values. Use a present/absent check, not parameter expansion of the value:

```bash
[ -n "$CLOUDRIFT_API_URL" ] && echo "CLOUDRIFT_API_URL: set (override)" || echo "CLOUDRIFT_API_URL: default"
[ -n "$CLOUDRIFT_API_KEY" ] && echo "CLOUDRIFT_API_KEY: set" || echo "CLOUDRIFT_API_KEY: MISSING"
```

Do **not** use `${CLOUDRIFT_API_KEY:+set}${CLOUDRIFT_API_KEY:-MISSING}` — when the var is set, `${VAR:-MISSING}` still expands to the **value** (the default only applies when unset), so you get `set<actual-secret>` printed.

If a user-specified overlay was meant to redirect the cluster but `CLOUDRIFT_API_URL` shows as default, you sourced the wrong file (or didn't source the overlay at all) — fix and re-check before running. `.env*` files are gitignored — never `git add` them.

**H200 on CloudRift typically requires a non-default cluster.** The public `api.cloudrift.ai` may not return H200 capacity. If the user asks for H200, confirm which env file to use (or which cluster to target) before running — don't fall back to the public endpoint silently.

### GCP

```bash
deplodock vm create gcp \
  --instance <name> \
  --zone <zone> \
  --machine-type <resolved_type> \
  --provisioning-model <FLEX_START|SPOT|STANDARD> \
  --wait-ssh
```

Always pass `--wait-ssh` for the start-server use case so the VM is genuinely ready when the command returns. GCP project comes from `gcloud config` — do not pass `--project`. If `gcloud auth` is not active, the command will fail; in that case suggest the user run `! gcloud auth login` (the `!` prefix runs it in their session).

For long-running servers the user wants to keep, consider increasing `--max-run-duration` past the default `7d` only if the user explicitly asks — otherwise let the default stand.

## After Creation

When the create command succeeds:

1. Capture the SSH connection string from the command output (CloudRift prints `user@ip`; GCP needs `gcloud compute ssh <instance> --zone <zone>` or the external IP from `gcloud compute instances describe`).
2. Report to the user: provider, instance ID/name, zone (GCP), instance type, SSH command.
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
- [ ] An SSH connection target was reported (CloudRift) or `--wait-ssh` confirmed reachability (GCP).
- [ ] The instance type and zone match what `deplodock/hardware.py` resolves for the requested GPU.
- [ ] If the user named a provider, the chosen provider matches it exactly (no fallback substitution).

If any check fails, report the failure and the raw output instead of claiming success.

## Common Mistakes to Avoid

- Don't pass private key paths to `--ssh-key` for CloudRift — it requires `.pub`.
- Don't invent instance types; always derive them from `GPU_INSTANCE_TYPES` + `resolve_instance_type`.
- Don't pick the GCP zone arbitrarily — use `GPU_GCP_ZONES` so the GPU is actually available there.
- Don't set `--dry-run` when the user asked for a real server. Use it only when the user asks to preview.
- Don't run `deplodock deploy cloud` for this task — that bundles deploy+teardown and won't leave a server idle for the user.
- Don't override a user-specified provider. If the GPU isn't available on that provider, abort and ask — never quietly run on a different provider "because that's where it's available."
