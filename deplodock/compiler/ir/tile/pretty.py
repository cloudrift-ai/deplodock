"""Tile IR pretty-printer — structural view of a ``TileOp``.

Used by ``deplodock compile --ir tile`` so the pre-materialization
form is inspectable. Handles the Tile-IR schedule vocabulary
(``Block`` / ``BoundLoop`` / ``Combine``) plus Loop-IR leaves
(``Load`` / ``Assign`` / ``Accum`` / ``Write`` / ``Cond`` / ``Loop``).
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import render as render_expr
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Select, Write
from deplodock.compiler.ir.tile.ir import (
    Block,
    BoundLoop,
    Combine,
    Stmt,
    TileOp,
)


def pretty_print(tile_op: TileOp) -> str:
    """Render a ``TileOp`` as an indented structural listing."""
    lines: list[str] = []
    sig_in = ", ".join(tile_op.inputs) or "-"
    sig_out = ", ".join(tile_op.outputs) or "-"
    lines.append(f"kernel {tile_op.name or '<unnamed>'}  inputs: {sig_in}  outputs: {sig_out}")
    _render_body(tile_op.body, "    ", lines)
    return "\n".join(lines)


def _render_body(stmts: tuple[Stmt, ...], indent: str, lines: list[str]) -> None:
    for stmt in stmts:
        _render_stmt(stmt, indent, lines)


def _render_stmt(stmt: Stmt, indent: str, lines: list[str]) -> None:
    if isinstance(stmt, Block):
        axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in stmt.axes) or "-"
        lines.append(f"{indent}Block(axes=({axes})):")
        _render_body(stmt.body, indent + "    ", lines)
        return
    if isinstance(stmt, BoundLoop):
        kind = "reduce" if any(isinstance(s, Accum) for s in stmt.body) else "free"
        lines.append(f"{indent}BoundLoop({stmt.axis.name}:{stmt.axis.extent}={stmt.bind}):  # {kind}")
        _render_body(stmt.body, indent + "    ", lines)
        return
    if isinstance(stmt, Combine):
        lines.append(f"{indent}Combine({stmt.name}, op={stmt.op.name}, via={stmt.via})")
        return

    # Loop-IR leaves
    if isinstance(stmt, Load):
        idx = ", ".join(render_expr(e) for e in stmt.index)
        lines.append(f"{indent}{stmt.name} = load {stmt.input}[{idx}]")
        return
    if isinstance(stmt, Assign):
        args = ", ".join(stmt.args)
        lines.append(f"{indent}{stmt.name} = {stmt.op.name}({args})")
        return
    if isinstance(stmt, Accum):
        lines.append(f"{indent}{stmt.name} <- {stmt.op.name}({stmt.name}, {stmt.value})")
        return
    if isinstance(stmt, Write):
        idx = ", ".join(render_expr(e) for e in stmt.index)
        lines.append(f"{indent}{stmt.output}[{idx}] = {stmt.value}")
        return
    if isinstance(stmt, Select):
        for bi, br in enumerate(stmt.branches):
            prefix = f"{stmt.name} =" if bi == 0 else f"{' ' * len(stmt.name)}  "
            lines.append(f"{indent}{prefix} {br.value} when ({render_expr(br.select)})")
        return
    if isinstance(stmt, Loop):
        kind = "reduce" if any(isinstance(s, Accum) for s in stmt.body) else "free"
        lines.append(f"{indent}Loop({stmt.axis.name} in 0..{stmt.axis.extent}):  # {kind}")
        _render_body(stmt.body, indent + "    ", lines)
        return
    if isinstance(stmt, Cond):
        lines.append(f"{indent}if ({render_expr(stmt.cond)}):")
        _render_body(stmt.body, indent + "    ", lines)
        if stmt.else_body:
            lines.append(f"{indent}else:")
            _render_body(stmt.else_body, indent + "    ", lines)
        return

    lines.append(f"{indent}<unrecognized {type(stmt).__name__}>")


__all__ = ["pretty_print"]
