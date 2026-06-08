# Move golden configs from Python code to per-GPU YAML files

## Context

Golden configs are the ground-truth tuning records — per matmul/reduce/pointwise shape, the autotuned knob set plus
deplodock-vs-cuBLAS latencies — that pin what the search **prior** should land on for canonical shapes. Today they live
as ~50 hand/script-generated `@dataclass` instances inside one 600-line region of
`deplodock/compiler/pipeline/search/golden.py` (a machine-generated `BEGIN/END GENERATED` matmul block plus a
hand-maintained `REDUCE_POINTWISE_GOLDENS` list). Every config currently carries `gpu_name="NVIDIA GeForce RTX 5090"` /
`compute_cap=(12, 0)` — the data is already GPU-tagged, but there is no way to keep more than one GPU's goldens, and
regenerating them means rewriting Python source via a text-splice.

Goal: store the data as **one YAML file per GPU** (`search/goldens/*.yaml`), loaded into the same `GOLDEN_CONFIGS`
list at import. This makes goldens editable as data, keeps `golden.py` import-light (no torch/cupy), and opens the door
to multiple GPUs without touching code. Per the user's decisions: **concatenate all GPU files** into one flat
`GOLDEN_CONFIGS`, files live under **`deplodock/compiler/pipeline/search/goldens/`**, and **matmul + reduce + pointwise
all share the per-GPU file** discriminated by a `kernel:` field.

## YAML schema (one file per GPU)

`deplodock/compiler/pipeline/search/goldens/rtx5090_sm120.yaml`:

```yaml
gpu_name: NVIDIA GeForce RTX 5090     # file-level header — not repeated per config
compute_cap: [12, 0]                  # parsed to tuple(int, int)
configs:
  - kernel: matmul                    # discriminator -> MatmulGoldenConfig
    name: square.512
    M: 512
    N: 512
    K: 512
    dtype: fp32                       # optional, default fp32 (omit for fp32 to stay terse)
    knobs: {BN: 32, BM: 8, FM: 4, FN: 2, BK: 64, SPLITK: 1, BR: 1, FK: 1, STAGE: '11', RING: 2, WARPSPEC: false}
    deplodock_us: 12.3
    cublas_us: 14.5
  - kernel: reduce                    # -> ReduceGoldenConfig (M, K)
    name: reduce.2048x2048
    M: 2048
    K: 2048
    knobs: {BN: 1, BM: 1, BR: 16, BK: 64, FM: 1, FN: 1, FK: 1, SPLITK: 1}
    deplodock_us: 4.12
    cublas_us: 6.13
  - kernel: pointwise                 # -> PointwiseGoldenConfig (M, N)
    ...
```

Notes on round-trip fidelity (load-bearing):
- `gpu_name` / `compute_cap` live in the file header and are stamped onto every config by the loader — no per-config
  repetition. `compute_cap` is a YAML list, converted to a `tuple` so it stays hashable for the frozen dataclass.
- Knob values keep their Python types. `STAGE: '11'` and MMA atom names must stay **strings** (quoted in YAML);
  ints/bools (`WARPSPEC: false`) round-trip natively. Writing via `yaml.safe_dump` preserves this automatically; reading
  via `yaml.safe_load` restores it. A test asserts the loaded `GOLDEN_CONFIGS` equals the previously hardcoded set.

## Changes

### 1. New data files — `deplodock/compiler/pipeline/search/goldens/rtx5090_sm120.yaml`
One file holding all 50 current configs (45 matmul from the generated block + 5 reduce/pointwise from
`REDUCE_POINTWISE_GOLDENS`), translated 1:1 from the current Python. This is the only GPU file for now.

### 2. Loader in `golden.py` (replaces the generated block + hand list)
Keep everything except the data: the module docstring, `matmul_snippet`, `_knobs_env`, the four dataclasses
(`GoldenConfig`, `MatmulGoldenConfig`, `ReduceGoldenConfig`, `PointwiseGoldenConfig`), and `goldens_by_name` are
unchanged. Delete the `BEGIN/END GENERATED` block and `REDUCE_POINTWISE_GOLDENS`; replace with:

```python
import yaml
_GOLDENS_DIR = Path(__file__).parent / "goldens"
_KERNEL_CLASSES = {"matmul": MatmulGoldenConfig, "reduce": ReduceGoldenConfig, "pointwise": PointwiseGoldenConfig}

def _load_goldens() -> list[GoldenConfig]:
    out: list[GoldenConfig] = []
    for path in sorted(_GOLDENS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        gpu_name, cap = doc["gpu_name"], tuple(doc["compute_cap"])
        for c in doc["configs"]:
            cls = _KERNEL_CLASSES[c.pop("kernel")]
            out.append(cls(gpu_name=gpu_name, compute_cap=cap, **c))
    return out

GOLDEN_CONFIGS: list[GoldenConfig] = _load_goldens()
```

- `yaml` is already a dependency (used by `deplodock/recipe/recipe.py`, `deplodock/benchmark/config.py` via
  `yaml.safe_load`), so the import stays light — no torch/cupy, preserving the module's import-light invariant that
  passes/tests rely on.
- All current consumers — `data/dataset.py` (`Dataset.from_golden`), `data/sample.py` (`Sample.from_golden`),
  `prior/diagnostics.py`, `scripts/golden_knob_heuristics.py`, `commands/compile.py|eval.py|run.py`, `goldens_by_name` —
  import the `GOLDEN_CONFIGS` symbol and need **no change**; the list is built the same shape, just from YAML.

### 3. Delete the regenerator — `scripts/find_golden_configs.py`
This script existed only to autotune matmul shapes and **text-splice the Python `GOLDEN_CONFIGS` block**. It is no
longer used (confirmed: no code imports it — only doc/comment mentions remain). Once the data lives in YAML, the splice
target is gone, so **delete the script** rather than convert it. The per-GPU YAML becomes the hand-maintained source of
truth, produced/updated via the existing CLI golden workflow (`tune --golden NAME --bench`, then record the winning
knobs/latencies into the GPU's YAML; `eval golden` to validate).

Clean up its now-stale references (comments/docs, no code):
- `golden.py` module docstring + `matmul_snippet` docstring + the deleted `# --- BEGIN GENERATED ... ---` marker.
- `prior/diagnostics.py:167` (the "tune the golden shapes (`scripts/find_golden_configs.py` …)" hint).
- `tests/compiler/test_golden_configs.py:6` docstring.
- `plans/golden-knob-heuristics.md:65` mention (light touch — historical plan doc).

### 4. Tests — `tests/compiler/test_golden_configs.py`
- Keep schema/invariant tests (`test_golden_configs_set_is_well_formed`, ratio/golden, snippet, repro,
  `goldens_by_name`). They operate on the loaded list and stay green.
- Add `test_goldens_load_from_yaml`: assert `goldens/` has ≥1 file, every config maps to a known `kernel` class, and the
  loaded set is non-empty with the expected count (50) — guards the YAML↔loader contract.
- `tests/compiler/pipeline/search/test_data.py` (`from_golden`) and `tests/compiler/test_analytic.py` need no change
  (they consume `GOLDEN_CONFIGS`).

### 5. Packaging / docs
- Ensure the YAML ships as package data so non-editable installs find it: check `pyproject.toml` / `setup.cfg` for a
  `package-data` / `include` entry and add `compiler/pipeline/search/goldens/*.yaml` if package data is declared
  (editable installs and source runs already resolve via `Path(__file__).parent`).
- Update `golden.py`'s module docstring (it describes the `BEGIN/END GENERATED` splice) to describe the YAML loader.
- Update `deplodock/compiler/pipeline/search/ARCHITECTURE.md` if it documents the golden-config storage location.

## Critical files
- `deplodock/compiler/pipeline/search/golden.py` — loader replaces the data block (dataclasses/helpers kept).
- `deplodock/compiler/pipeline/search/goldens/rtx5090_sm120.yaml` — NEW data file (50 configs).
- `scripts/find_golden_configs.py` — **deleted** (its only job was splicing the Python block).
- `tests/compiler/test_golden_configs.py` — add YAML-load test, keep invariants.
- `pyproject.toml`/`setup.*` — package-data include (only if declared).

## Verification
1. `./venv/bin/python -c "from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS; print(len(GOLDEN_CONFIGS))"`
   → 50, and types/knobs match the pre-migration values (spot-check `square.512`, a `.fp16`, a `reduce.*`).
2. `./venv/bin/pytest tests/compiler/test_golden_configs.py tests/compiler/pipeline/search/test_data.py tests/compiler/test_analytic.py -v`
   → all green.
3. `./venv/bin/python -m deplodock eval golden` and `eval analytic` → same tables as before the migration (the prior/
   diagnostics read the identical config set).
4. `./venv/bin/python -m deplodock run --golden square.512 --bench` → resolves the name, compiles, benches (confirms the
   name-lookup path through the YAML-loaded list).
5. `grep -rn find_golden_configs .` → no remaining references after the deletion + comment cleanup.
6. `make test` and `make lint`.
