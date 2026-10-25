"""The two-level relational block and the stack of blocks.

Each block applies six sublayers::

    1. cell attention   all cells of a seed, temporal RoPE, padding-only mask
    2. cell MoE FFN     routed on the cell signature z
    3. RowPool          cells -> row tokens
    4. row attention    over the sampled FK graph, with the FK-role bias
    5. row MoE FFN      routed on the row signature s
    6. Broadcast        row tokens -> cells

Sublayers 1, 2, 4 and 5 are pre-norm residual; RowPool and Broadcast are residual without a norm.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from .flex_cell import (FLEX_MIN_HEAD_DIM, HAS_FLEX, build_cell_block_mask, cell_score_mod,
                        flex_cell_attention)
from .moe import MoEFFN
from .row_level import Broadcast, RMSNorm, RowAttention, RowMoE, RowPool, RowSignature
from .time_encoding import TimeLadder

#: ``flex`` is the training backend; ``sdpa`` is the numerically equivalent reference that also
#: runs on CPU.
CELL_BACKENDS = ("sdpa", "flex")


class CellAttention(nn.Module):
    """Attention over all cells of a seed, with temporal RoPE on the leading dims of every head and
    additive untimed / clamped flags (``time_encoding.TimeLadder.time_bias``)."""

    def __init__(self, d_model: int, n_heads: int, ladder: TimeLadder, *, backend: str = "flex"):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        if backend not in CELL_BACKENDS:
            raise ValueError(f"unknown cell attention backend {backend!r}; expected {CELL_BACKENDS}")
        if backend == "flex" and (not HAS_FLEX or d_model // n_heads < FLEX_MIN_HEAD_DIM):
            raise ValueError(f"the flex backend needs torch >= 2.5 and a head dim >= "
                             f"{FLEX_MIN_HEAD_DIM}; use backend='sdpa'")
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.backend = backend
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cb, block_mask=None) -> Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, S, self.h, self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, S, self.h, self.d_h).transpose(1, 2)

        seed_t = cb.seed_time.unsqueeze(1)
        theta = self.ladder.theta(self.ladder.tau_from_times(seed_t, cb.cell_time), cb.cell_is_timed)
        q = self.ladder.rotate(q, theta.unsqueeze(1), self.n_rot)
        k = self.ladder.rotate(k, theta.unsqueeze(1), self.n_rot)
        clamped = self.ladder.was_clamped(seed_t, cb.cell_time) & cb.cell_is_timed

        if self.backend == "flex":
            if block_mask is None:
                block_mask = build_cell_block_mask(cb.is_padding)
            score_mod = cell_score_mod(cb.cell_is_timed, clamped, self.ladder.b_untimed,
                                       self.ladder.b_clamped, dtype=x.dtype)
            out = flex_cell_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        else:
            bias = self.ladder.time_bias(cb.cell_is_timed, cb.cell_is_timed,
                                         clamped, clamped).unsqueeze(1).to(x.dtype)
            real = ~cb.is_padding
            mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(1)
            attn = bias.masked_fill(~mask, float("-inf"))
            out = torch.nan_to_num(F.scaled_dot_product_attention(q, k, v, attn_mask=attn))
        return self.wo(out.transpose(1, 2).reshape(B, S, -1))


class TwoLevelBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_sig: int, ladder: TimeLadder, col_name_emb: Tensor,
                 role_name_emb: Tensor, *, n_heads: int = 8, cell_attn_backend: str = "flex",
                 num_experts: int = 8, top_k: int = 2, num_shared: int = 2, pool_slots: int = 4,
                 row_num_experts: int = 4, row_top_k: int = 2, row_num_shared: int = 1):
        super().__init__()
        self.cell_norm = RMSNorm(d_model)
        self.cell_attn = CellAttention(d_model, n_heads, ladder, backend=cell_attn_backend)
        self.cell_ffn_norm = RMSNorm(d_model)
        self.cell_ffn = MoEFFN(d_model, d_ff, d_sig, num_experts=num_experts, k=top_k,
                               num_shared=num_shared)
        self.pool = RowPool(d_model, d_sig, col_name_emb, slots=pool_slots)
        self.row_attn = RowAttention(d_model, d_sig, role_name_emb, ladder, n_heads=n_heads)
        self.row_ffn = RowMoE(d_model, d_ff, d_sig, num_experts=row_num_experts, k=row_top_k,
                              num_shared=row_num_shared)
        self.broadcast = Broadcast(d_model)

    def forward(self, h: Tensor, u: Tensor, z: Tensor, s: Tensor, cb, block_mask=None):
        """-> ``(cells, rows, aux_cell, aux_row)``."""
        h = h + self.cell_attn(self.cell_norm(h), cb, block_mask)
        h = h + self.cell_ffn(self.cell_ffn_norm(h), z, valid=~cb.is_padding)
        aux_cell = self.cell_ffn.ortho_loss()
        u = self.pool(h, u, s, cb)
        u = self.row_attn(u, cb)
        u, aux_row = self.row_ffn(u, s, cb)
        h = self.broadcast(h, u, cb)
        return h, u, aux_cell, aux_row


class TwoLevelSubstrate(nn.Module):
    """``n_blocks`` two-level blocks. Row tokens start as ``W_u [mean of the row's cells ; s_r]``.

    ``forward -> (cells [B, S, d], rows [B, R, d], aux)``, where ``aux`` is the cell and row auxiliary
    losses, each averaged over blocks.
    """

    def __init__(self, d_model: int, d_ff: int, d_sig: int, table_name_emb: Tensor,
                 role_name_emb: Tensor, col_name_emb: Tensor, *, n_blocks: int = 10,
                 n_heads: int = 8, max_hop: int = 8, grad_checkpoint: bool = False, **block_kw):
        super().__init__()
        self.ladder = TimeLadder()
        self.grad_checkpoint = grad_checkpoint
        self.row_sig = RowSignature(table_name_emb, role_name_emb, self.ladder, d_sig=d_sig,
                                    max_hop=max_hop)
        self.w_u = nn.Linear(d_model + d_sig, d_model, bias=False)
        self.blocks = nn.ModuleList(
            TwoLevelBlock(d_model, d_ff, d_sig, self.ladder, col_name_emb, role_name_emb,
                          n_heads=n_heads, **block_kw)
            for _ in range(n_blocks))
        self.flex = block_kw.get("cell_attn_backend", "flex") == "flex"

    def _init_rows(self, h: Tensor, s: Tensor, cb) -> Tensor:
        B, R, _ = s.shape
        rows = torch.arange(R, device=h.device).view(1, R, 1)
        member = ((cb.cell_row.unsqueeze(1) == rows) & (~cb.is_padding).unsqueeze(1)).to(h.dtype)
        mean = (member @ h) / member.sum(-1, keepdim=True).clamp_min(1.0)
        return self.w_u(torch.cat([mean, s], dim=-1))

    def forward(self, x: Tensor, cb, z: Tensor):
        s = self.row_sig(cb)
        u = self._init_rows(x, s, cb)
        block_mask = build_cell_block_mask(cb.is_padding) if self.flex else None
        h = x
        aux_cell = aux_row = x.new_zeros(())
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                h, u, a_c, a_r = torch.utils.checkpoint.checkpoint(
                    blk, h, u, z, s, cb, block_mask, use_reentrant=False)
            else:
                h, u, a_c, a_r = blk(h, u, z, s, cb, block_mask)
            aux_cell = aux_cell + a_c
            aux_row = aux_row + a_r
        n = max(len(self.blocks), 1)
        return h, u, aux_cell / n + aux_row / n
