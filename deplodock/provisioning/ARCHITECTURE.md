# Provisioning Layer

This package owns *all* VM lifecycle logic for the CLI:

* `deplodock deploy cloud` → `provision_cloud_vm()`
* `deplodock bench` → `provision_cloud_vm()` (via `benchmark/execution.py`)
* `deplodock vm create gpu` → `provision_cloud_vm()`
* `deplodock vm create gcp / cloudrift` → provider `create_instance()` directly (single-shot manual)

`deploy ssh` and fixed-host `bench` go through `host.py` / `remote.py` and don't touch this orchestrator.

## Layout

```
provisioning/
  cloud.py        # orchestrator: provision_cloud_vm + provider dispatch
  candidates.py   # iter_candidates: ordered list of allocation attempts
  errors.py       # CapacityExhausted, TerminalProvisionError
  cloudrift.py    # CloudRift API wrapper (create/delete/wait)
  gcp.py          # gcloud-compute wrapper
  ssh.py          # generic wait_for_ssh
  host.py         # RemoteHost abstraction over an existing SSH target
  remote.py       # bare-VM bootstrap (driver/CUDA install)
  staging.py      # tar-and-scp helpers used by the deploy layer
  shell.py        # async shell-out helper
  types.py        # VMConnectionInfo dataclass
```

## Allocation model

`provision_cloud_vm()` enumerates *candidates* via `iter_candidates()`. A candidate is one concrete `(provider, instance_type, zone?)` tuple. Order:

1. Filter `hardware.GPU_INSTANCE_TYPES[gpu_name]` by `--provider` if set; else use the full preference-ordered list.
2. For each `(provider, base_type)` entry:
   * **CloudRift** → one candidate per base type.
   * **GCP** → one candidate per zone in `hardware.GPU_GCP_ZONES[gpu_name]` (falls back to `DEFAULT_GCP_ZONE`); all zones for the current base type before advancing to the next entry.

The orchestrator tries candidates in this order until one succeeds or all are exhausted. **Fallback never crosses the provider boundary of the initial selection** — H200 callers who passed `--provider cloudrift` won't be silently relocated to GCP.

For each candidate, the orchestrator makes up to `SAME_CANDIDATE_RETRIES` (= 2) attempts on transient errors. On the contracted exceptions it short-circuits:

| Provider raises | Orchestrator does |
|---|---|
| `CapacityExhausted` | Advance to next candidate immediately. |
| `TerminalProvisionError` | Abort. Propagate to caller. |
| Any other `Exception` | Treat as transient: retry same candidate up to `SAME_CANDIDATE_RETRIES`, then advance. |
| Returns `None` | Treat as soft capacity exhaustion: advance. |

When every candidate is exhausted, the orchestrator returns `None` and logs the last error.

## Error contract

`errors.py` defines two exceptions that providers raise to communicate intent:

* **`CapacityExhausted`** — the current candidate has no capacity. Raised for CloudRift HTTP 503/429 on rent, CloudRift `Inactive` terminal status / readiness timeout, GCP `ZONE_RESOURCE_POOL_EXHAUSTED` / `QUOTA_EXCEEDED` / `STOCKOUT` in `gcloud create` stderr, and GCP RUNNING-status timeout. Recoverable by trying a different candidate.
* **`TerminalProvisionError`** — auth, malformed request, anything that will recur on any candidate. Raised for CloudRift HTTP 4xx (other than 429) and unrecognized non-zero `gcloud create` exit. Not recoverable; propagated to caller.

Anything else a provider raises is treated as transient by the orchestrator.

## Orphan cleanup invariant

**A provider's `create_instance` must never leave a VM behind that the caller can't see.** If `create_instance` returns a `VMConnectionInfo`, the caller owns the VM and is responsible for delete. If it raises, *no VM exists*: the provider must have already terminated any partially-provisioned instance before re-raising.

This invariant lets the orchestrator iterate candidates without leaking orphans:

* `cloudrift.create_instance` wraps `wait_for_status` in try/except and calls `_terminate_instance` on any exception or `None` return.
* `gcp.create_instance` wraps the post-`gcloud create` flow (`wait_for_status`, external-IP fetch, SSH wait) and runs `gcloud delete` on any exception. The pre-create classification branch (`_classify_create_failure`) only fires when nothing was provisioned, so no cleanup is needed there.

Both providers swallow termination errors and log them — the original failure is what the orchestrator needs to classify the next move, not a cascade from cleanup.

## Adding a new provider

See `commands/ARCHITECTURE.md` § *Adding a New VM Provider*. The provider's `create_instance` must:

1. Raise `CapacityExhausted` on no-capacity / no-stock signals.
2. Raise `TerminalProvisionError` on credentials / bad-request errors.
3. Terminate any orphan it created before re-raising or returning `None`.
4. Return `VMConnectionInfo` (with `delete_info` set) on success.

Then add it to `_provision_candidate` in `cloud.py` and to `iter_candidates` if it has zone- or region-style fan-out.
