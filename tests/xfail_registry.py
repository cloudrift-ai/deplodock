"""Single source of truth for tests expected to fail during the tile-IR rebuild.

The tile IR (``deplodock/compiler/ir/tile/`` + ``.../pipeline/passes/lowering/tile/``) is being
rebuilt from scratch. Until it is restored, some recovery tests cannot pass. Rather than scatter
``@pytest.mark.xfail`` decorators across the suite, every expected failure is registered HERE and
applied centrally by the ``pytest_collection_modifyitems`` hook in ``tests/conftest.py``.

How it works
------------
- Each entry maps a **node-id substring** to a one-line reason. Any collected test whose ``nodeid``
  contains the substring is marked ``xfail(strict=False)``.
- ``strict=False`` means a test that starts passing again shows up as **XPASS** (not a failure) —
  that is the recovery signal. When a capability is restored, delete its entry here and the test
  reverts to a hard requirement.
- A substring matches broadly: ``"test_fused_edge.py"`` xfails the whole file (e.g. when its
  module-level tile imports break collection — see note below); a full nodeid like
  ``"test_fused_edge.py::test_fused_map_matmul_runs_correctly"`` xfails a single case.

An **empty registry means the rebuild is fully recovered.**

Note on collection-time import errors
-------------------------------------
A file whose *module-level* import of a tile symbol breaks will raise at COLLECTION, before any
item exists to mark — pytest reports it as an error, which xfail cannot catch. The known
tile-entangled files (their import lines reference ``...lowering.tile``) are listed in
``TILE_ENTANGLED_FILES`` for visibility; when the rebuild removes/renames those symbols, either
update the import or move the file's reason into ``XFAIL`` after the import is made lazy.
"""

from __future__ import annotations

# nodeid-substring -> reason. Populated as the rebuild breaks tests; emptied as it recovers.
XFAIL: dict[str, str] = {}

# Files whose accuracy tests build their reference through tile-internal helpers (kept so the
# coverage survives, but they couple to the IR being rebuilt). Listed for visibility only — these
# pass today (tile IR is still intact) and only need xfail entries once the rebuild lands.
TILE_ENTANGLED_FILES: tuple[str, ...] = (
    "test_monoid_reduce_kernel.py",  # all 4 tests use deferred_combine_tilegraph / reduce_tilegraphop
    "test_fused_edge.py",  # 4/5 tests hand-build TileGraph / use assemble_block / seed_demoted
    "test_stage_scalar.py",  # test_scalar_matmul_stages_through_pipeline asserts on TileOp / StageBundle
)
