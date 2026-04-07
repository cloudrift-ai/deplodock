"""Generic command workload: render and run a templated shell command on a remote VM.

Used by recipes whose `command` block declares a `run` template and a list of
`result_files` to pull back. The harness flattens variant params to leaf names,
substitutes them into the template via `string.Template`, runs the rendered
command on the provisioned VM, then scp's matching result files back to the
local run directory.
"""

import logging
import shlex
from pathlib import Path
from string import Template

from deplodock.planner import BenchmarkTask
from deplodock.planner.variant import Variant

logger = logging.getLogger(__name__)


def _leaf_name(key: str) -> str:
    """Take the last dot-separated segment of a matrix key (e.g. 'deploy.gpu' → 'gpu')."""
    return key.rsplit(".", 1)[-1]


def build_substitution_map(
    variant: Variant,
    gpu_device_ids: list[int],
    repo_dir: str | None,
    task_dir: str,
) -> dict[str, str]:
    """Build the substitution map for a command template.

    Variant params are flattened to their leaf names; conflicts (two keys
    sharing a leaf) raise ValueError. Harness keys $task_dir, $gpu_device_ids,
    and $repo_dir (when staging is configured) are injected.
    """
    subs: dict[str, str] = {}
    seen_sources: dict[str, str] = {}
    for key, value in variant.params.items():
        leaf = _leaf_name(key)
        if leaf in seen_sources:
            raise ValueError(
                f"variant params have conflicting leaf name '{leaf}': '{seen_sources[leaf]}' and '{key}' both flatten to the same name."
            )
        seen_sources[leaf] = key
        subs[leaf] = str(value)

    subs["task_dir"] = task_dir
    subs["gpu_device_ids"] = ",".join(str(i) for i in gpu_device_ids)
    if repo_dir is not None:
        subs["repo_dir"] = repo_dir
    return subs


def render_command(template: str, subs: dict[str, str]) -> str:
    """Render a string.Template command, raising a friendly error on missing vars.

    Uses ``safe_substitute`` so that shell metacharacters like ``$(...)``,
    ``$1``, ``$$``, and ``${VAR:-default}`` are passed through to the shell
    untouched. Missing identifiers referenced by the template (anything that
    matches Template's idpattern but isn't in ``subs``) still raise a friendly
    ValueError.
    """
    tmpl = Template(template)
    referenced = {m.group("named") or m.group("braced") for m in tmpl.pattern.finditer(template) if (m.group("named") or m.group("braced"))}
    missing = sorted(referenced - subs.keys())
    if missing:
        raise ValueError(f"command template references undefined variable: ${missing[0]}")
    return tmpl.safe_substitute(subs)


async def _expand_remote_glob(run_cmd, task_dir: str, pattern: str) -> list[str]:
    """List files matching `pattern` inside `task_dir` on the remote VM.

    Returns absolute remote paths. Empty list when nothing matches (the
    `|| true` swallows the ls failure).
    """
    safe_pattern = shlex.quote(pattern)
    # task_dir may start with `~/`; pass it unquoted so the remote shell
    # expands tilde. It is internally composed and free of shell metachars.
    cmd = (
        f'sh -c \'cd {task_dir} && for f in {pattern}; do [ -e "$f" ] && printf "%s/%s\\n" {task_dir} "$f"; done\' # pattern={safe_pattern}'
    )
    rc, stdout, _ = await run_cmd(cmd, stream=False)
    if rc != 0 or not stdout:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


async def run_command_workload(
    task: BenchmarkTask,
    run_cmd,
    repo_dir: str | None,
    task_dir: str,
    gpu_device_ids: list[int],
    server: str,
    ssh_key: str,
    ssh_port: int,
    dry_run: bool = False,
) -> tuple[bool, dict]:
    """Run one command-recipe task on the remote VM.

    Returns (success, info) where info contains the rendered command and a
    list of locally-pulled result paths.
    """
    from deplodock.provisioning.ssh_transport import scp_from_remote

    cmd_cfg = task.recipe.command
    assert cmd_cfg is not None, "run_command_workload requires a command recipe"

    subs = build_substitution_map(task.variant, gpu_device_ids, repo_dir, task_dir)
    rendered = render_command(cmd_cfg.run, subs)

    # Prepend env exports if any.
    if cmd_cfg.env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in cmd_cfg.env.items())
        rendered_with_env = f"{env_prefix} {rendered}"
    else:
        rendered_with_env = rendered

    # Ensure task_dir exists. task_dir is internally composed from
    # REMOTE_DEPLOY_DIR/group_label/variant and may begin with `~/`, so we
    # interpolate it unquoted to preserve tilde expansion. Both group_label
    # and variant come from sanitized internal sources (no shell metachars).
    await run_cmd(f"mkdir -p {task_dir}")

    logger.info(f"Running command for {task.variant}:\n{rendered}")
    rc, _, _ = await run_cmd(rendered_with_env, log_output=True, timeout=cmd_cfg.timeout)
    success = rc == 0
    info: dict = {"rendered_command": rendered, "result_paths": []}

    if not success:
        logger.error(f"Command failed (rc={rc}) for {task.variant}")
        return False, info

    if dry_run:
        return True, info

    # Pull back result files (with glob expansion on the remote).
    for pattern in cmd_cfg.result_files:
        # If the pattern contains no glob metachars, treat it literally; else expand.
        if any(c in pattern for c in "*?["):
            remote_paths = await _expand_remote_glob(run_cmd, task_dir, pattern)
            if not remote_paths:
                logger.warning(f"result_files pattern '{pattern}' matched no files in {task_dir}")
                continue
        else:
            remote_paths = [f"{task_dir}/{pattern}"]

        for remote_path in remote_paths:
            basename = Path(remote_path).name
            local_path = task.run_dir / f"{task.variant}_{basename}"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            rc_scp, stderr = await scp_from_remote(server, ssh_key, ssh_port, remote_path, str(local_path))
            if rc_scp != 0:
                logger.warning(f"scp_from_remote failed for {remote_path}: {stderr}")
                continue
            info["result_paths"].append(str(local_path))

    return True, info
