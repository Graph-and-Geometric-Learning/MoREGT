"""Masked-cell reconstruction head.

For the final state ``h_i`` of a masked cell, ``h~ = RMSNorm(h_i)``:

* numerical: ``y_hat = w^T h~ + b`` (zero-initialised), Huber loss (delta 1) on the z-scored value;
* categorical: logits ``(W_c h~ + b_c)^T q_{c,k}`` against the frozen label embeddings ``q`` of the
  cell's own column, cross-entropy.

Neither decoder has a vocabulary-shaped parameter, so the head transfers to unseen schemas. The loss is
``L_num + lambda_cat * L_cat``. The statistics include each modality's no-skill reference (the Huber
loss of predicting each column's in-batch mean, and ``log K`` for a uniform guess), which pretraining
uses to compute its validation skill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.masking import CATEGORICAL_ID, NUMERICAL_ID
from .row_level import RMSNorm


@torch.no_grad()
def _mean_baseline(y: Tensor, gid: Tensor, delta: float) -> Tensor:
    """Huber loss of predicting each column's in-batch mean (columns with >= 2 targets)."""
    y = y.detach().float()
    uniq, inv = torch.unique(gid, return_inverse=True)
    cnt = torch.zeros(uniq.numel(), device=y.device, dtype=y.dtype).index_add_(0, inv, torch.ones_like(y))
    tot = torch.zeros(uniq.numel(), device=y.device, dtype=y.dtype).index_add_(0, inv, y)
    keep = (cnt >= 2)[inv]
    if not bool(keep.any()):
        return y.new_zeros(())
    mu = (tot / cnt.clamp_min(1.0))[inv]
    return F.huber_loss(mu[keep], y[keep], reduction="mean", delta=delta)


class MaskedCellHead(nn.Module):
    def __init__(self, d_model: int, d_cat: int, *, lambda_cat: float = 4.0, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_cat = lambda_cat
        self.huber_delta = huber_delta
        self.norm = RMSNorm(d_model)
        self.num_head = nn.Linear(d_model, 1)
        self.cat_proj = nn.Linear(d_model, d_cat)
        nn.init.zeros_(self.num_head.weight)
        nn.init.zeros_(self.num_head.bias)

    def category_logits(self, hidden: Tensor, gid: Tensor, spec, cat_table: Tensor) -> Tensor:
        """``[M, max_cat]`` logits over each cell's own column's classes (padded with -inf)."""
        dev = hidden.device
        h = self.cat_proj(hidden)
        n_cat = spec.n_cat.to(dev).index_select(0, gid)
        base = spec.cat_base.to(dev).index_select(0, gid)
        ar = torch.arange(int(n_cat.max()), device=dev).unsqueeze(0)
        rows = (base.unsqueeze(1) + ar).clamp(0, cat_table.shape[0] - 1)
        emb = cat_table.index_select(0, rows.reshape(-1)).view(*rows.shape, -1)
        logits = torch.einsum("md,mkd->mk", h, emb)
        return logits.masked_fill(~(ar < n_cat.unsqueeze(1)), float("-inf"))

    def forward(self, cell_states: Tensor, targets, spec, cat_table: Tensor) -> tuple[Tensor, dict]:
        """-> ``(loss, stats)``; a graph-connected zero when nothing was masked."""
        zero = cell_states.sum() * 0.0
        if len(targets) == 0:
            return zero, {"n_num": 0.0, "n_cat": 0.0}
        h = self.norm(cell_states[targets.b, targets.s])
        is_num = targets.stype == NUMERICAL_ID
        is_cat = targets.stype == CATEGORICAL_ID
        stats: dict[str, float] = {"n_num": float(int(is_num.sum())), "n_cat": float(int(is_cat.sum()))}

        loss_num = zero
        if bool(is_num.any()):
            pred = self.num_head(h[is_num]).squeeze(-1)
            loss_num = F.huber_loss(pred, targets.y_num[is_num], delta=self.huber_delta)
            stats["loss_num"] = float(loss_num.detach())
            stats["loss_num_mean"] = float(_mean_baseline(targets.y_num[is_num], targets.gid[is_num],
                                                          self.huber_delta))

        loss_cat = zero
        if bool(is_cat.any()):
            gid_cat = targets.gid[is_cat]
            logits = self.category_logits(h[is_cat], gid_cat, spec, cat_table)
            y_cat = targets.y_cat[is_cat]
            loss_cat = F.cross_entropy(logits, y_cat)
            stats["loss_cat"] = float(loss_cat.detach())
            stats["cat_acc"] = float((logits.argmax(-1) == y_cat).float().mean())
            n_cat = spec.n_cat.to(logits.device).index_select(0, gid_cat).clamp_min(2)
            stats["loss_cat_uniform"] = float(n_cat.float().log().mean())
        return loss_num + self.lambda_cat * loss_cat, stats
