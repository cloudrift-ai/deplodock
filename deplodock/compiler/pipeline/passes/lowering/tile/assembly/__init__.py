"""assemble(TileGraph) -> the materialized TileOp tower."""

from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block

__all__ = ["assemble_block"]
