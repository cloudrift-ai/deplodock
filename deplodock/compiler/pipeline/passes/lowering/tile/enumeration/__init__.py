"""Move composer (enumeration): build the TileGraph DAG + search the Schedule."""

from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._compose import try_compose

__all__ = ["try_compose"]
