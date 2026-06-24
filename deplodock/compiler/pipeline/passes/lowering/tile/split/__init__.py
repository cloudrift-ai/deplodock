"""Split-decision passes — structural forks that un-fuse a LoopOp into a
producer/consumer kernel set BEFORE the ``enumeration`` build seeds the tile
DAG (``plans/tile-ir-block-dag.md``: the ``010_split_demoted`` cut at the
partition head branches the outer two-level tree). Each rule matches a
``LoopOp`` and may return a ``Graph`` fragment (the kernel-set change) the
engine splices; the deterministic ``enumeration/010_build`` then seeds each
resulting LoopOp. Kept its own pass dir (run before ``enumeration``) so the cut
sees the un-tiled body."""
