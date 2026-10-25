"""SwiGLU experts and the cell-level mixture-of-experts FFN.

The router reads a value-free routing feature (the relational signature); the experts transform the
hidden state. Gates are a softmax over the top-``k`` router logits. Each expert runs only on the
tokens routed to it, and ``num_shared`` always-on experts are added for every token. Balance comes
from a router-orthogonality penalty rather than a load-balancing loss, so expert usage may follow the
long-tailed frequency of column types.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def top_k_gates(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Softmax over the top-``k`` logits (zero elsewhere), and the selected expert ids."""
    topv, topi = logits.topk(k, dim=-1)
    masked = torch.full_like(logits, float("-inf")).scatter(-1, topi, topv)
    return F.softmax(masked, dim=-1), topi


def sparse_combine(x: Tensor, gates: Tensor, topi: Tensor, experts: nn.ModuleList, *,
                   valid: Tensor | None = None) -> Tensor:
    """``sum_{e in top-k} g_e E_e(x)``, running each expert only on its own tokens.

    Tokens where ``valid`` is False (padding) are skipped and come back as zeros. An expert that gets
    no token has no gradient on that step, so DDP needs ``find_unused_parameters=True``.
    """
    d = x.shape[-1]
    xf = x.reshape(-1, d)
    gf = gates.reshape(-1, gates.shape[-1])
    tf = topi.reshape(-1, topi.shape[-1])
    live = valid.reshape(-1) if valid is not None else None
    y = torch.zeros_like(xf)
    for e, expert in enumerate(experts):
        sel = (tf == e).any(-1)
        if live is not None:
            sel = sel & live
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ge = gf.index_select(0, idx)[:, e:e + 1]
        # under autocast the expert output is bf16 while the accumulator is fp32
        y = y.index_add(0, idx, (ge * expert(xf.index_select(0, idx))).to(y.dtype))
    return y.view_as(x)


def apply_shared(x: Tensor, shared: nn.ModuleList | None, *, valid: Tensor | None = None) -> Tensor:
    """Sum of the always-on experts over the ``valid`` tokens (zero when there are none)."""
    if not shared:
        return torch.zeros_like(x)
    d = x.shape[-1]
    xf = x.reshape(-1, d)
    idx = (valid.reshape(-1) if valid is not None
           else torch.ones(xf.shape[0], dtype=torch.bool, device=x.device)).nonzero(as_tuple=True)[0]
    y = torch.zeros_like(xf)
    if idx.numel():
        xe = xf.index_select(0, idx)
        acc = shared[0](xe)
        for m in shared[1:]:
            acc = acc + m(xe)
        y = y.index_add(0, idx, acc.to(y.dtype))
    return y.view_as(x)


def orthogonality_loss(w: Tensor) -> Tensor:
    """``||W_hat W_hat^T - I||_F^2`` over the L2-normalised rows of a router matrix."""
    W = F.normalize(w, dim=-1)
    eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    return ((W @ W.t() - eye) ** 2).sum()


class MoEFFN(nn.Module):
    """``num_experts`` routed SwiGLU experts behind a linear top-``k`` router on ``route_feat``,
    plus ``num_shared`` always-on experts."""

    def __init__(self, d_model: int, d_ff: int, d_route: int, *, num_experts: int = 8, k: int = 2,
                 num_shared: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        self.shared = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_shared)) or None
        self.router = nn.Linear(d_route, num_experts, bias=False)

    def gates(self, route_feat: Tensor) -> Tensor:
        """``[..., num_experts]`` gate weights: zero off the top-k, rows sum to 1."""
        return top_k_gates(self.router(route_feat), self.k)[0]

    def forward(self, x: Tensor, route_feat: Tensor, *, valid: Tensor | None = None) -> Tensor:
        g, topi = top_k_gates(self.router(route_feat), self.k)
        return sparse_combine(x, g, topi, self.experts, valid=valid) + apply_shared(
            x, self.shared, valid=valid)

    def ortho_loss(self) -> Tensor:
        return orthogonality_loss(self.router.weight)
