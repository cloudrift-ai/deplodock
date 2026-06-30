# Style Guide

## Python Code Style

### Naming

- `snake_case` for functions, variables, and modules.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for module-level constants.
- Prefix private/internal helpers with underscore (e.g., `_ssh_base_args`).

### Logging

All output goes through Python's `logging` module — never use `print()`.

Each module gets a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Level mapping:

| Pattern | Level |
|---|---|
| Normal output | `logger.info(...)` |
| Warnings | `logger.warning(...)` |
| Errors | `logger.error(...)` |

Two logging configurations:

- **Standalone CLI** (`setup_cli_logging()` in `logging_setup.py`): `%(message)s` format — output identical to `print()`.
- **Bench** (`setup_logging()` in `bench_logging.py`): `[%(name)s] %(message)s` — prefixed output with module/group names.

The bench formatter shows short module names for library loggers (`emmy.deploy.orchestrate` → `[orchestrate]`) and split group loggers (`rtx5090_x_1.ModelName` → `[rtx5090_x_1] [ModelName]`).

### Error Handling

Log errors and return failure for operational errors:

```python
logger.error("Failed to pull images")
return False
```

Raise exceptions for programming errors and invalid internal state:

```python
raise ValueError(f"Model config must have 'name' field: {model}")
```

### Docstrings

Use triple-quote docstrings for modules and public functions. Keep them
to one line when the purpose is obvious:

```python
def load_recipe(recipe_dir):
    """Load recipe.yaml and return base Recipe (no matrix expansion)."""
```

Use `Args:` / `Returns:` sections only when parameters or return values
are non-obvious.

### Module Structure

- `__init__.py` files contain only re-exports. No classes, functions, interfaces, or business logic.
- ABCs and interfaces go in explicitly named files (e.g., `backend/base.py`, not `backend/__init__.py`).
- Business logic goes in named modules (e.g., `recipe.py`, `compose.py`).
- `commands/` layer: CLI code only (argparse registration + `handle_*` handlers). Reusable logic lives in top-level domain packages (`emmy/deploy/`, `emmy/provisioning/`, `emmy/benchmark/`).

### Imports

Group imports in this order, separated by blank lines:

1. Standard library (`os`, `sys`, `subprocess`, etc.)
2. Third-party packages (`yaml`, `pandas`, `pytest`)
3. Local imports (`from emmy.deploy import ...`)

### Formatting

- 4 spaces for indentation.
- Keep lines under ~140 characters (enforced by Ruff).
- Double quotes for strings.

### Tooling

Style rules are enforced by [Ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. Run `make lint` to check and `make format` to auto-fix. Enabled rule sets: `E` (pycodestyle), `F` (pyflakes), `W` (warnings), `I` (isort), `UP` (pyupgrade), `B` (bugbear).

### IR statements must be frozen dataclasses

Every concrete `Stmt` subclass — Loop-IR (`Loop`, `StridedLoop`, `Cond`, leaves), Tile-IR (`GridTile`, `ThreadTile`,
`RegisterTile`, `SerialTile`, `StridedTile`, `Stage`, `StageBundle`, `AsyncWait`), Kernel-IR (`Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, `TmaLoad`, `MbarrierInit`, …) — must be declared `@dataclass(frozen=True)`. `Body` is
already a `tuple[Stmt, ...]` subclass, so freezing every Stmt makes the entire body tree hashable end-to-end.

Why: structural caches (`Body.structural_key()` and any future bodies-as-cache-keys work) traverse the body and hash
every Stmt. A single mutable Stmt anywhere in the tree poisons every cache that keys on the surrounding Body — and
the surrounding code can't degrade gracefully without losing the optimization.

If you need to "edit" a frozen Stmt, return a new instance via `dataclasses.replace(stmt, field=value)`. If a
`__post_init__` needs to coerce a field (e.g. `tuple → Body`), use `object.__setattr__(self, "field", coerced)` —
that's the standard pattern for frozen dataclasses that still need light normalization at construction time. Don't
add `try/except TypeError` fallbacks around structural caches to tolerate unhashable stmts; fix the unhashable stmt
instead.

Op subclasses don't have to be frozen (the engine mutates `op.source` / `op.knobs` / `op.inputs` / `op.outputs` post-
construction). Just make sure no Op ends up as a *field value* of a Stmt — `Assign.op` / `Accum.op` / `Select.op` take
an `ElementwiseImpl` (the lightweight value object, already hashable), never an `ElementwiseOp` wrapper.

### Dependency Injection for Testability

Shared logic accepts callable parameters (`run_cmd`, `write_file`) so
that local and SSH targets can provide their own implementations and
tests can use dry-run or mock versions:

```python
def run_deploy(run_cmd, write_file, recipe, model_dir, ...):
```

### Concurrency

The codebase is fully async. All subprocess and network I/O uses native
`asyncio` APIs (`asyncio.create_subprocess_exec/shell`, `httpx.AsyncClient`).
Use `asyncio.Semaphore` to limit concurrency. CLI entry points use
`asyncio.run()`:

```python
def handle_foo(args):
    asyncio.run(_handle_foo(args))

async def _handle_foo(args):
    await ...
```

All `run_cmd` callables are `async def` with a `timeout` parameter.
Timeouts use `asyncio.wait_for()` around `proc.communicate()`:

```python
async def run_cmd(command, stream=True, timeout=600):
    proc = await asyncio.create_subprocess_shell(command, ...)
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
```

On `TimeoutError`: kill the process, await termination, log, return `(1, "", "")`.

## Commit Messages

- Keep the subject line short (under ~72 characters).
- Use imperative mood: "Add feature", not "Added feature".
- No multi-line descriptions unless truly necessary — if the change needs
  explanation, put it in the PR description.

Good:
```
Add CLI deploy tool with local and SSH targets
Fix variant resolution for single-GPU models
Remove unused benchmark script
```

Bad:
```
This commit adds a new CLI deploy tool that supports both local and SSH
deployment targets with dry-run mode and variant resolution for different
GPU configurations.
```
