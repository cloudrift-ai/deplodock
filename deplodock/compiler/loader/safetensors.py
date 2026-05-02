"""Read parameters/buffers straight from safetensors shards.

This bypasses the PyTorch ``nn.Module`` round-trip: given a model id (HF
repo) or a local directory of safetensors files, the loader resolves
each ``ConstantOp.source_path`` to a tensor in one of the shards, reads
it as a numpy array, and runs the recorded ``load_ops`` chain via the
NumPy backend.

The function keeps a small key-canonicalization table because HF
checkpoints sometimes carry a ``model.`` prefix that ``torch.export``
strips, and vice versa. We try the original name, then with ``model.``
added, then with ``model.`` removed — that covers every case we've
seen on Llama / Qwen / TinyLlama.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.loader.binder import apply_load_ops

logger = logging.getLogger(__name__)


def _resolve_model_dir(model_id_or_path: str) -> Path:
    """Return a local directory containing the model's safetensors files.

    If the argument is an existing directory, use it as-is. Otherwise
    treat it as an HF repo id and snapshot-download it (cached).
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path))


def _build_index(model_dir: Path) -> dict[str, Path]:
    """Map each tensor name in the model to the safetensors shard it lives in.

    Handles both the single-file (``model.safetensors``) and sharded
    (``model.safetensors.index.json``) layouts.
    """
    from safetensors import safe_open

    index_json = model_dir / "model.safetensors.index.json"
    if index_json.exists():
        weight_map = json.loads(index_json.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="numpy") as f:
            return {name: single for name in f.keys()}

    raise FileNotFoundError(f"No safetensors files found under {model_dir}")


def _candidate_keys(source_path: str) -> list[str]:
    """Generate the names to try in the safetensors index for a constant."""
    cands = [source_path]
    if source_path.startswith("model."):
        cands.append(source_path[len("model.") :])
    else:
        cands.append("model." + source_path)
    return cands


def load_constants_from_safetensors(graph: Graph, model_id_or_path: str) -> dict[str, np.ndarray]:
    """Bind every parameter/buffer ``ConstantOp`` from the model's safetensors.

    Returns a dict keyed by node id, ready to feed into ``Backend.run``
    as ``input_data``. Scalar constants (``value is not None``) and
    constants without a ``source_path`` are skipped — the backend
    materializes them on its own.
    """
    from safetensors import safe_open

    model_dir = _resolve_model_dir(model_id_or_path)
    index = _build_index(model_dir)

    needed: dict[str, list[str]] = {}  # shard path → list of keys
    resolved: dict[str, str] = {}  # node_id → safetensors key
    for nid, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp) or node.op.value is not None or node.op.source_path is None:
            continue
        for cand in _candidate_keys(node.op.source_path):
            if cand in index:
                resolved[nid] = cand
                needed.setdefault(str(index[cand]), []).append(cand)
                break
        else:
            logger.warning("safetensors loader: no key matched for %s (source_path=%r)", nid, node.op.source_path)

    sources: dict[str, np.ndarray] = {}
    for shard_path, keys in needed.items():
        with safe_open(shard_path, framework="numpy") as f:
            for k in set(keys):
                sources[k] = f.get_tensor(k)

    out: dict[str, np.ndarray] = {}
    for nid, node in graph.nodes.items():
        if not isinstance(node.op, ConstantOp) or node.op.value is not None:
            continue
        key = resolved.get(nid)
        if key is None:
            continue
        out[nid] = apply_load_ops(sources[key], node.op.load_ops)
    return out
