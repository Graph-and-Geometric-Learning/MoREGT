"""Cache locations, seeding and CPU thread limits."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def graph_cache_dir(dataset: str, text_encoder: str = "minilm") -> Path:
    """Materialised graph of ``dataset``. Keyed by the cell-text encoder, whose output it stores."""
    root = os.environ.get("MOREGT_GRAPH_CACHE") or REPO / "data" / "graph_cache"
    return Path(root) / text_encoder / dataset


def schema_cache_path(dataset: str, encoder: str) -> Path:
    """Cached schema-name embeddings of ``dataset`` under the name encoder ``encoder``."""
    root = os.environ.get("MOREGT_SCHEMA_CACHE") or REPO / "data" / "schema_cache"
    return Path(root) / dataset / f"name_emb_{encoder.replace('/', '__')}.pt"


def ckpt_dir(run: str = "") -> Path:
    """Pretraining checkpoint directory of ``run``."""
    root = Path(os.environ.get("MOREGT_CKPT_DIR") or REPO / "checkpoints")
    return root / run if run else root


def seed_everything(seed: int = 0) -> int:
    """Seed python, numpy and torch (CPU and CUDA)."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def limit_torch_threads(default: int = 2) -> int:
    """Cap torch's intra-op CPU threads (``$MOREGT_TORCH_THREADS``, default 2).

    The collate runs many small CPU tensor ops per batch; with the default thread count their
    launch overhead dominates and the GPU waits on the main process.
    """
    import torch

    n = int(os.environ.get("MOREGT_TORCH_THREADS") or default)
    if n < 1:
        raise ValueError(f"MOREGT_TORCH_THREADS must be >= 1, got {n}")
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)
    return n
