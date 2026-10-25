"""Schema-free cell encoder.

Each cell becomes the RT cell token

    x_{v,c} = W_v e_{v,c} + W_name n_c,

the sum of a value component and the frozen name embedding ``n_c`` of its column. The value
embedding ``e`` has one shared projection per datatype, not per column:

=============  ===================================================  ============================
stype          value -> e                                           per-database frozen data
=============  ===================================================  ============================
numerical      ``Linear(1, e)`` of ``(x - mu_c) / sigma_c``          column mean and std
categorical    ``Linear(d_text, e)`` of the frozen label embedding   category-label table
text           ``Linear(d_value, e)`` of the frozen MiniLM vector    none
timestamp      year (standardised per database) + cyclic fields     one year mean and std
=============  ===================================================  ============================

All per-database data are non-persistent buffers, so no learned parameter is sized by any schema and
one ``state_dict`` loads on every database. A masked cell keeps its name component and replaces its
value component with a learned per-stype mask vector.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.collate import column_vocab, feature_col_names
from ..text.schema import N_STYPES, build_column_modality_ids, category_index

N_TIME_FIELDS = 7                                  # year, month, day, weekday, hour, minute, second
CYCLIC_PERIODS = (12.0, 31.0, 7.0, 24.0, 60.0, 60.0)


def column_tables(bundle) -> dict:
    """The per-database frozen data the encoder reads, indexed by global column id."""
    from torch_frame.data.stats import StatType

    vocab = column_vocab(bundle)
    C = len(vocab)
    cat_ix, _ = category_index(bundle)
    num_mean = torch.zeros(C)
    num_std = torch.ones(C)
    cat_base = torch.zeros(C, dtype=torch.long)
    years: list[float] = []
    layout: dict[str, dict[str, list[int]]] = {}
    gids: dict[str, Tensor] = {}
    for nt in bundle.node_types:
        tf = bundle.data[nt].tf
        for st, cols in tf.col_names_dict.items():
            kind = str(st).rsplit(".", 1)[-1]
            for c in cols:
                gid = vocab.get((nt, c))
                if gid is None:
                    continue
                stats = bundle.col_stats_dict[nt][c]
                if kind == "numerical":
                    num_mean[gid] = float(stats.get(StatType.MEAN, 0.0))
                    num_std[gid] = float(stats.get(StatType.STD, 1.0)) + 1e-6
                elif kind == "categorical":
                    cat_base[gid] = cat_ix[(nt, c)][0]
                elif kind == "timestamp":
                    yr = stats.get(StatType.YEAR_RANGE)
                    if yr is not None:
                        years.extend(float(y) for y in yr)
        sorted_cols = feature_col_names(tf)
        gids[nt] = torch.tensor([vocab[(nt, c)] for c in sorted_cols], dtype=torch.long)
        layout[nt] = {str(st).rsplit(".", 1)[-1]: [sorted_cols.index(c) for c in cols]
                      for st, cols in tf.col_names_dict.items()}
    if years:
        y = torch.tensor(years, dtype=torch.float32)
        year_stats = torch.tensor([float(y.mean()), float(y.std()) + 1e-6])
    else:
        year_stats = torch.tensor([2000.0, 1.0])
    return dict(num_mean=num_mean, num_std=num_std, cat_base=cat_base, year_stats=year_stats,
                modality_id=build_column_modality_ids(bundle), layout=layout, gids=gids)


def value_text_dim(bundle) -> int | None:
    """Width of the frozen free-text value embeddings, if the database has text columns."""
    from torch_frame.data.stats import StatType

    for nt in bundle.node_types:
        for s in bundle.col_stats_dict[nt].values():
            d = s.get(StatType.EMB_DIM)
            if d is not None:
                return int(d)
    return None


class CellEncoder(nn.Module):
    def __init__(self, bundle, name_emb: Tensor, cat_emb: Tensor, *, d_model: int = 256,
                 time_out_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.time_out_size = time_out_size
        e = self.enc_channels = d_model

        t = column_tables(bundle)
        for key in ("num_mean", "num_std", "cat_base", "year_stats", "modality_id"):
            self.register_buffer(key, t[key], persistent=False)
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("cat_emb", cat_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("cyclic_periods", torch.tensor(CYCLIC_PERIODS), persistent=False)
        self._layout, self._gids = t["layout"], t["gids"]

        d_cat = int(self.cat_emb.shape[-1])
        self.d_value_text = value_text_dim(bundle) or d_cat
        self.proj_num = nn.Linear(1, e)
        self.proj_cat = nn.Linear(d_cat, e)
        self.proj_text = nn.Linear(self.d_value_text, e)
        self.time_weight = nn.Parameter(torch.randn(N_TIME_FIELDS, time_out_size, e) * 0.01)
        self.time_bias = nn.Parameter(torch.zeros(e))
        self.w_v = nn.Linear(e, d_model)
        self.name_proj = nn.Linear(int(name_emb.shape[1]), d_model)
        self.mask_emb = nn.Embedding(N_STYPES, d_model)

    def _encode_numerical(self, feat: Tensor, gids: Tensor) -> Tensor:
        z = (feat - self.num_mean[gids]) / self.num_std[gids]
        return self.proj_num(torch.nan_to_num(z).unsqueeze(-1))

    def _encode_categorical(self, feat: Tensor, gids: Tensor) -> Tensor:
        # class index, -1 for missing; row 0 of the label table is the missing slot
        idx = torch.where(feat < 0, torch.zeros_like(feat), feat + self.cat_base[gids].unsqueeze(0))
        return self.proj_cat(self.cat_emb[idx.clamp(0, self.cat_emb.shape[0] - 1)])

    def _encode_text(self, feat) -> Tensor:
        v = feat.values.view(feat.num_rows, -1, self.d_value_text) if hasattr(feat, "values") else feat
        return self.proj_text(v.to(self.proj_text.weight.dtype))

    def _encode_timestamp(self, feat: Tensor) -> Tensor:
        """``[n, C, 7]`` -> ``[n, C, e]``: the standardised year as a ramp, the other fields as
        sin/cos harmonics."""
        f = feat.to(torch.float32)
        year = ((f[..., :1] - self.year_stats[0]) / self.year_stats[1]).clamp(-8.0, 8.0)
        ang = 2 * torch.pi * f[..., 1:] / self.cyclic_periods
        k = torch.arange(1, self.time_out_size + 1, device=f.device, dtype=f.dtype)
        cyc = torch.where(k.long() % 2 == 0, torch.cos(ang.unsqueeze(-1) * ((k + 1) // 2)),
                          torch.sin(ang.unsqueeze(-1) * ((k + 1) // 2)))
        parts = torch.cat([year.unsqueeze(-1) * k, cyc], dim=-2)          # [n, C, 7, out_size]
        return torch.einsum("ncfo,foe->nce", parts, self.time_weight) + self.time_bias

    def encode_type(self, nt: str, tf) -> Tensor:
        """All rows of one table -> ``[n, C_nt, d_model]`` in ``feature_col_names`` order."""
        import torch_frame

        gids = self._gids[nt].to(self.name_emb.device)
        out = torch.zeros(tf.num_rows, gids.numel(), self.enc_channels, device=self.name_emb.device)
        handlers = {
            "numerical": (torch_frame.numerical, self._encode_numerical, True),
            "categorical": (torch_frame.categorical, self._encode_categorical, True),
            "embedding": (torch_frame.embedding, self._encode_text, False),
            "timestamp": (torch_frame.timestamp, self._encode_timestamp, False),
        }
        for kind, pos in self._layout[nt].items():
            spec = handlers.get(kind)
            if spec is None or not pos:
                continue
            st, fn, needs_gids = spec
            feat = tf.feat_dict.get(st)
            if feat is None:
                continue
            idx = torch.tensor(pos, device=out.device, dtype=torch.long)
            block = fn(feat, gids[idx]) if needs_gids else fn(feat)
            out[:, idx, :] = block.to(out.dtype)
        return self.w_v(out) + self.name_proj(self.name_emb.index_select(0, gids)).unsqueeze(0)

    def masked_token(self, cb) -> Tensor:
        """``[B, S, d_model]``: ``mask_emb[stype] + W_name n_c``."""
        flat = cb.cell_col.clamp_min(0).reshape(-1)
        B, S = cb.cell_col.shape
        name = self.name_emb.index_select(0, flat).view(B, S, -1)
        mod = self.modality_id.index_select(0, flat).view(B, S)
        return self.mask_emb(mod) + self.name_proj(name)

    def forward(self, cb, cell_mask: Tensor | None = None) -> Tensor:
        """-> ``[B, S, d_model]`` cell tokens (zero at padding). ``cell_mask`` withholds values."""
        dev = self.name_emb.device
        h = torch.zeros(cb.num_seeds, cb.seq_len, self.d_model, device=dev)
        for nt, tf in cb.tf_dict.items():
            if nt not in self._layout or nt not in cb.cell_placement:
                continue
            cell = self.encode_type(nt, tf)
            b_idx, s_idx, row_idx, col_idx = (t.to(dev) for t in cb.cell_placement[nt])
            h[b_idx, s_idx] = cell[row_idx, col_idx].to(h.dtype)
        if cell_mask is not None:
            h = torch.where(cell_mask.to(dev).unsqueeze(-1), self.masked_token(cb).to(h.dtype), h)
        return h * (~cb.is_padding).to(dev).unsqueeze(-1)
