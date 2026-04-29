"""Count effective lines of Python code under a directory.

Skips: blank lines, ``#`` comments, docstrings (first-stmt string
expressions on module / class / function), ``import`` / ``from … import``
lines, and ``__all__ = [...]`` re-export blocks.

Usage:
    python scripts/count_loc.py <root>             # nested tree, all levels
    python scripts/count_loc.py <root> --depth 2   # collapse below depth 2
    python scripts/count_loc.py <root> --files     # also list individual .py files
"""

import argparse
import ast
from pathlib import Path


def count_file(path: Path) -> int:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0

    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                skip.add(ln)
        # __all__ = [...]  /  __all__: list[str] = [...]  /  __all__ += [...]
        is_all = False
        if isinstance(node, ast.Assign):
            is_all = any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        elif isinstance(node, ast.AnnAssign):
            is_all = isinstance(node.target, ast.Name) and node.target.id == "__all__"
        elif isinstance(node, ast.AugAssign):
            is_all = isinstance(node.target, ast.Name) and node.target.id == "__all__"
        if is_all:
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                skip.add(ln)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ds = body[0]
                for ln in range(ds.lineno, (ds.end_lineno or ds.lineno) + 1):
                    skip.add(ln)

    count = 0
    for i, line in enumerate(src.splitlines(), start=1):
        if i in skip:
            continue
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        count += 1
    return count


# --------------------------------------------------------------------------- #
# Tree model: each Node holds direct file LOC + child subdirs + child files.
# --------------------------------------------------------------------------- #


class Node:
    __slots__ = ("name", "subdirs", "files", "self_loc")

    def __init__(self, name: str):
        self.name = name
        self.subdirs: dict[str, Node] = {}
        self.files: dict[str, int] = {}
        self.self_loc = 0

    def total(self) -> int:
        return self.self_loc + sum(c.total() for c in self.subdirs.values())


def build_tree(root: Path) -> Node:
    tree = Node(root.name)
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        n = count_file(p)
        rel = p.relative_to(root)
        cur = tree
        for part in rel.parts[:-1]:
            cur = cur.subdirs.setdefault(part, Node(part))
        cur.files[rel.parts[-1]] = n
        cur.self_loc += n
    return tree


def render_tree(
    node: Node,
    *,
    show_files: bool,
    max_depth: int | None,
    cur_depth: int = 0,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True,
    total: int | None = None,
) -> list[str]:
    lines: list[str] = []
    grand = node.total() if total is None else total

    if is_root:
        head = f"{node.name}/"
        new_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        head = f"{prefix}{connector}{node.name}/"
        new_prefix = prefix + ("    " if is_last else "│   ")

    loc = node.total()
    pct = 100 * loc / grand if grand else 0
    lines.append(f"{head:<48}{loc:>6}  {pct:>4.1f}%")

    # max_depth caps how deep we descend. cur_depth is the depth of `node`
    # (root = 0). At cur_depth == max_depth, stop expanding children — the
    # node's total above already includes everything beneath it.
    if max_depth is not None and cur_depth >= max_depth:
        return lines

    subs = sorted(node.subdirs.values(), key=lambda c: -c.total())
    files = sorted(node.files.items(), key=lambda kv: -kv[1]) if show_files else []

    n_children = len(subs) + len(files)
    for i, sub in enumerate(subs):
        last = i == n_children - 1
        lines.extend(
            render_tree(
                sub,
                show_files=show_files,
                max_depth=max_depth,
                cur_depth=cur_depth + 1,
                prefix=new_prefix,
                is_last=last,
                is_root=False,
                total=grand,
            )
        )

    for j, (fname, floc) in enumerate(files):
        last = len(subs) + j == n_children - 1
        connector = "└── " if last else "├── "
        pct = 100 * floc / grand if grand else 0
        head = f"{new_prefix}{connector}{fname}"
        lines.append(f"{head:<48}{floc:>6}  {pct:>4.1f}%")

    return lines


def _parse_depth(value: str) -> tuple[str, int | None]:
    """Return ``(mode, max_depth)`` from the ``--depth`` value.

    - ``"folder"`` (default): per-folder, unlimited depth.
    - ``"file"``: per-folder + per-file leaves, unlimited depth.
    - integer N: per-folder, capped at depth N.
    """
    if value == "folder":
        return ("folder", None)
    if value == "file":
        return ("file", None)
    try:
        return ("folder", int(value))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"--depth must be 'folder', 'file', or an integer (got {value!r})") from err


def main():
    ap = argparse.ArgumentParser(description="Count effective Python LOC.")
    ap.add_argument("root", type=Path)
    ap.add_argument(
        "--depth",
        type=_parse_depth,
        default=("folder", None),
        help="'folder' (default, all folder levels), 'file' (also show per-file rows), or an integer N (folder levels capped at N).",
    )
    args = ap.parse_args()
    mode, max_depth = args.depth

    tree = build_tree(args.root)
    for line in render_tree(tree, show_files=(mode == "file"), max_depth=max_depth):
        print(line)


if __name__ == "__main__":
    main()
