"""Task head: an MLP on the seed row's row token."""
from __future__ import annotations

from torch import Tensor, nn


class RowTokenHead(nn.Module):
    """``y = W_2 GELU(W_1 LayerNorm(u_root) + b_1) + b_2``."""

    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, rows: Tensor, cb) -> Tensor:
        """``rows [B, R, d]`` -> ``[B, out_dim]``."""
        root = cb.row_is_root.to(rows.dtype)
        root = root / root.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.mlp((rows * root.unsqueeze(-1)).sum(dim=1))
