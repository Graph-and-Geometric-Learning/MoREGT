"""Frozen text encoders and an on-disk embedding cache.

An encoder is any callable ``encode(texts, kind) -> Tensor[N, d]`` with L2-normalised rows. ``kind``
is ``"query"`` or ``"document"``; queries are prefixed with an instruction prompt.

Two encoders are used, for different text:

* schema names (columns, tables, FK roles, category labels): Qwen3-Embedding-4B, embedded once per
  database and cached by :class:`EmbeddingCache`;
* free-text cell values: MiniLM-L12-v2, embedded once when the graph is materialised and stored
  inside the graph cache.

:class:`HashEncoder` is a dependency-free stand-in for tests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

QUERY_INSTRUCTION = (
    "Given the name of a database table and column, represent its relational role and semantic meaning."
)

ENCODER_MODELS = {
    "qwen": "Qwen/Qwen3-Embedding-4B",                     # schema names, d = 2560
    "minilm": "sentence-transformers/all-MiniLM-L12-v2",   # cell values, d = 384
}


def _as_texts(texts) -> list[str]:
    """Plain strings for the encoder; missing values (``None``/NaN) become the empty string."""
    out: list[str] = []
    for t in texts:
        if t is None or (isinstance(t, float) and t != t):
            out.append("")
        else:
            out.append(t if isinstance(t, str) else str(t))
    return out


class HashEncoder:
    """Deterministic pseudo-embeddings, stable per ``(text, kind)``. For tests only."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        out = torch.empty(len(texts), self.dim, dtype=torch.float32)
        for i, t in enumerate(texts):
            h = int(hashlib.sha256(f"{kind}\x01{t}".encode()).hexdigest()[:16], 16)
            g = torch.Generator().manual_seed(h & 0x7FFFFFFFFFFFFFFF)
            out[i] = torch.randn(self.dim, generator=g)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SentenceEncoder:
    """A frozen sentence-transformers model, loaded lazily. Queries get the instruction prompt."""

    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 16,
                 query_instruction: str = QUERY_INSTRUCTION):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        emb = self.model.encode(
            _as_texts(texts),
            prompt=self.query_instruction if kind == "query" else None,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.float().cpu()


class EmbeddingCache:
    """Memoise an encoder per ``(kind, text)`` in a file, so each string is embedded once.

    A cache miss runs the wrapped encoder and rewrites the file atomically.
    """

    def __init__(self, encoder, path: str | Path | None = None):
        self.encoder = encoder
        self.path = Path(path) if path is not None else None
        self._store: dict[str, Tensor] = {}
        if self.path is not None and self.path.exists():
            self._store = torch.load(self.path, map_location="cpu")

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return hashlib.sha256(f"{kind}\x01{text}".encode()).hexdigest()

    def __call__(self, texts: list[str], kind: str = "document") -> Tensor:
        missing = [t for t in texts if self._key(kind, t) not in self._store]
        if missing:
            for t, e in zip(missing, self.encoder(missing, kind=kind), strict=True):
                self._store[self._key(kind, t)] = e.clone()
            self._flush()
        return torch.stack([self._store[self._key(kind, t)] for t in texts], dim=0)

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        torch.save(self._store, tmp)
        os.replace(tmp, self.path)


def name_encoder(dataset: str, encoder: str = "qwen", *, d_text: int = 64):
    """The cached schema-name encoder for ``dataset``. ``encoder="hash"`` gives an uncached
    :class:`HashEncoder` of width ``d_text``."""
    from ..utils import schema_cache_path

    if encoder == "hash":
        return HashEncoder(dim=d_text)
    model = SentenceEncoder(ENCODER_MODELS.get(encoder, encoder))
    return EmbeddingCache(model, schema_cache_path(dataset, encoder))


def value_encoder(encoder: str = "minilm", *, batch_size: int = 1024, device: str | None = None):
    """The frozen encoder for free-text cell values, called by torch_frame at materialisation."""
    return SentenceEncoder(ENCODER_MODELS.get(encoder, encoder), batch_size=batch_size,
                           device=device)
