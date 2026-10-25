"""The value-free cell signature that the cell-level MoE router reads.

    z_{v,c} = RMSNorm( W_s n_c + E_st[stype(c)] + W_tau [1_timed (sin theta_v; cos theta_v) ; 1_clamped] )

``n_c`` is the frozen name embedding of the cell's column and ``theta_v`` the time ladder of its row.
The signature depends only on the cell's column, modality and row age relative to the seed: never on
the cell's value, any other cell, or which neighbours were sampled. It is computed once per forward
pass and reused by every block.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .time_encoding import TimeLadder


class RelationalSignature(nn.Module):
    def __init__(self, name_emb: Tensor, modality_id: Tensor, n_stypes: int, *, d_sig: int = 128):
        super().__init__()
        self.d_sig = d_sig
        self.register_buffer("name_emb", name_emb.detach().to(torch.float32), persistent=False)
        self.register_buffer("modality_id", modality_id.detach().to(torch.long), persistent=False)
        self.schema_proj = nn.Linear(int(name_emb.shape[1]), d_sig, bias=False)
        self.stype_emb = nn.Embedding(max(int(n_stypes), 1), d_sig)
        self.ladder = TimeLadder()
        self.w_tau = nn.Linear(self.ladder.feat_dim, d_sig, bias=False)
        self.norm = nn.RMSNorm(d_sig)

    def forward(self, cb) -> Tensor:
        col = cb.cell_col.clamp(min=0)
        B, S = col.shape
        flat = col.reshape(-1)
        name = self.name_emb.index_select(0, flat).view(B, S, -1)
        mod = self.modality_id.index_select(0, flat).view(B, S)
        z = self.schema_proj(name) + self.stype_emb(mod)
        seed_t = cb.seed_time.unsqueeze(1)
        tau = self.ladder.tau_from_times(seed_t, cb.cell_time)
        clamped = self.ladder.was_clamped(seed_t, cb.cell_time) & cb.cell_is_timed
        z = z + self.w_tau(self.ladder.feats(tau, cb.cell_is_timed, clamped).to(z.dtype))
        return self.norm(z)                                                  # [B, S, d_sig]
