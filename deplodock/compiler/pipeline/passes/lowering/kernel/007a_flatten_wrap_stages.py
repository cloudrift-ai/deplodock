"""Flatten wrap-body Stages just before materialization.

A wrap-body ``Stage(sources, body=[consumer])`` carries its consumer
subtree nested inside ``Stage.body``. The materializer wants the legacy
flat shape — ``[Stage(sources, body=()), *consumer_stmts]`` — so it can
emit the producer cooperative-load scaffolding, then walk the consumer
stmts as siblings. This pass performs that flatten as a discrete
Tile-IR → Tile-IR rewrite, so ``008_materialize_tile`` receives an
already-flat body and commits no structural rewrites of its own.

Runs as ``007a`` (immediately before ``008``): nothing else observes the
flattened shape. Idempotent — once every ``Stage.body`` is empty there is
nothing left to flatten, so a re-run is skipped.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Body, Cond, Stmt
from deplodock.compiler.ir.tile.ir import (
    ComputeStage,
    GridTile,
    RegisterTile,
    SerialTile,
    Stage,
    StridedTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def flatten_wrap_stages(body) -> tuple[Stmt, ...]:
    """Pre-flatten wrap-body Stages: ``Stage(sources, body=[consumer])`` becomes
    ``[Stage(sources, body=Body(())), *consumer_stmts]`` so the materializer's
    flat walker can emit producer scaffolding then process the consumer stmts
    as siblings.

    Recurses into Loop / StridedLoop / Cond / Tile (Grid / Thread / Register)
    bodies so nested Stages flatten too. ComputeStage's ``compute`` body is
    kept attached to the stage; the materializer emits it specially.
    """
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Stage):
            inner = flatten_wrap_stages(s.body)
            # Recursively flatten ComputeStage.compute too (in case future
            # passes nest stages inside it).
            if isinstance(s, ComputeStage):
                compute_inner = flatten_wrap_stages(s.compute)
                out.append(replace(s, body=Body(()), compute=Body(compute_inner)))
            else:
                out.append(replace(s, body=Body(())))
            out.extend(inner)
        elif isinstance(s, (GridTile, ThreadTile, SerialTile, StridedTile, RegisterTile)):
            new_body = flatten_wrap_stages(s.body)
            out.append(s.with_bodies((Body(new_body),)))
        elif isinstance(s, Cond):
            new_body = flatten_wrap_stages(s.body)
            new_else = flatten_wrap_stages(s.else_body) if s.else_body else ()
            out.append(replace(s, body=Body(new_body), else_body=Body(new_else)))
        else:
            out.append(s)
    return tuple(out)


def rewrite(root: Node) -> Graph | None:
    top: TileOp = root.op
    flat = flatten_wrap_stages(top.body)
    if flat == tuple(top.body):
        raise RuleSkipped("no wrap-body Stage to flatten")
    return TileOp(body=Body(flat), name=top.name, knobs=dict(top.knobs))
