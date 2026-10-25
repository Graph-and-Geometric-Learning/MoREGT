"""MoREGT: cell encoder + relational signature + two-level substrate + row-token head.

    x          = CellEncoder(cb)              # [B, S, d]      value + frozen column-name token
    z          = RelationalSignature(cb)      # [B, S, d_sig]  value-free routing signature
    h, u, aux  = TwoLevelSubstrate(x, cb, z)  # cells [B, S, d], rows [B, R, d]
    y          = RowTokenHead(u, cb)          # [B, out_dim]   read at the seed row

The cell and row feed-forward layers are mixtures of experts whose routers read only value-free
signatures (what a cell or row *is*), while the experts transform the evolving hidden state.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from torch import nn

from ..text.schema import N_STYPES, SchemaEmbeddings, build_column_modality_ids
from .encoder import CellEncoder
from .heads import RowTokenHead
from .signature import RelationalSignature
from .two_level import TwoLevelSubstrate


@dataclass
class MoREGTConfig:
    """Architecture hyperparameters. The defaults are the 134M-parameter model."""

    d_model: int = 256
    n_blocks: int = 10
    n_heads: int = 8
    d_ff: int = 1024
    d_sig: int = 128
    num_experts: int = 8          # routed cell experts
    num_shared: int = 2           # always-on cell experts
    top_k: int = 2
    row_num_experts: int = 4
    row_num_shared: int = 1
    row_top_k: int = 2
    pool_slots: int = 4
    max_hop: int = 8
    cell_attn_backend: str = "flex"
    grad_checkpoint: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> MoREGTConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> MoREGTConfig:
        """The config stored in ``config.json`` next to a pretraining checkpoint."""
        p = Path(path)
        cfg_path = p if p.suffix == ".json" else p.parent / "config.json"
        return cls.from_dict(json.loads(cfg_path.read_text())["model"])

    def to_dict(self) -> dict:
        return asdict(self)


class MoREGT(nn.Module):
    def __init__(self, bundle, schema: SchemaEmbeddings, config: MoREGTConfig | None = None, *,
                 out_dim: int = 1):
        super().__init__()
        cfg = config or MoREGTConfig()
        self.config = cfg
        self.encoder = CellEncoder(bundle, schema.column, schema.category, d_model=cfg.d_model)
        self.signature = RelationalSignature(schema.column, build_column_modality_ids(bundle),
                                             N_STYPES, d_sig=cfg.d_sig)
        self.substrate = TwoLevelSubstrate(
            cfg.d_model, cfg.d_ff, cfg.d_sig, schema.table, schema.role, schema.column,
            n_blocks=cfg.n_blocks, n_heads=cfg.n_heads, max_hop=cfg.max_hop,
            grad_checkpoint=cfg.grad_checkpoint, cell_attn_backend=cfg.cell_attn_backend,
            num_experts=cfg.num_experts, top_k=cfg.top_k, num_shared=cfg.num_shared,
            pool_slots=cfg.pool_slots, row_num_experts=cfg.row_num_experts,
            row_top_k=cfg.row_top_k, row_num_shared=cfg.row_num_shared,
        )
        self.head = RowTokenHead(cfg.d_model, out_dim)

    def category_table(self):
        """The frozen category-label table that the masked-cell head scores against."""
        return self.encoder.cat_emb

    def forward(self, cb, *, cell_mask=None, return_cells: bool = False):
        """-> ``(logits, aux)``, plus the final cell states when ``return_cells``.

        ``cell_mask`` withholds those cells' values (masked-cell pretraining); ``aux`` is the
        routers' auxiliary loss.
        """
        x = self.encoder(cb, cell_mask)
        z = self.signature(cb)
        cells, rows, aux = self.substrate(x, cb, z)
        logits = self.head(rows, cb)
        return (logits, aux, cells) if return_cells else (logits, aux)
