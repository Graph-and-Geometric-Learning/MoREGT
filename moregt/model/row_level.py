"""Row-level operators of the two-level block.

* :class:`RowSignature`  value-free row signature ``s_r`` (router input of the row MoE)
* :class:`RowPool`       cells -> row token
* :class:`RowAttention`  row <-> row attention over the sampled FK graph, with an FK-role bias
* :class:`RowMoE`        cosine-routed mixture of experts on ``s_r``
* :class:`Broadcast`     row token -> its cells

No parameter is sized by the number of tables or FK roles of a database: table and role ids only
index frozen name embeddings, registered as non-persistent buffers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .moe import SwiGLU, apply_shared, orthogonality_loss, sparse_combine, top_k_gates
from .time_encoding import TimeLadder

DIR_CHILD, DIR_PARENT, DIR_SELF = 0, 1, 2
N_DIR = 3


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _frozen(module: nn.Module, name: str, t: Tensor) -> None:
    module.register_buffer(name, t.detach().to(torch.float32), persistent=False)


class RowSignature(nn.Module):
    r"""``s_r = RMSNorm(W_tab N_tab[T(r)] + W_rho N_role[rho_in(r)] + E_hop[min(hop_r, 8)]
    + W_tau [ladder(tau_r); 1_clamped])``.

    ``role_name_emb`` is ``[K + 1, d_text]`` with an all-zero row 0, the role of the seed row.
    """

    def __init__(self, table_name_emb: Tensor, role_name_emb: Tensor, ladder: TimeLadder, *,
                 d_sig: int = 128, max_hop: int = 8):
        super().__init__()
        _frozen(self, "table_name_emb", table_name_emb)
        _frozen(self, "role_name_emb", role_name_emb)
        self.ladder = ladder
        self.max_hop = max_hop
        self.w_tab = nn.Linear(table_name_emb.shape[-1], d_sig, bias=False)
        self.w_rho = nn.Linear(role_name_emb.shape[-1], d_sig, bias=False)
        self.hop_emb = nn.Embedding(max_hop + 1, d_sig)
        self.w_tau = nn.Linear(ladder.feat_dim, d_sig, bias=False)
        self.norm = RMSNorm(d_sig)

    def forward(self, cb) -> Tensor:
        """-> ``[B, R, d_sig]``."""
        table = cb.row_table.clamp_min(0)
        role = cb.row_in_role.clamp(0, self.role_name_emb.shape[0] - 1)
        z = (self.w_tab(self.table_name_emb[table]) + self.w_rho(self.role_name_emb[role])
             + self.hop_emb(cb.row_hop.clamp(0, self.max_hop)))
        seed_t = cb.seed_time.unsqueeze(1)
        tau = self.ladder.tau_from_times(seed_t, cb.row_time)
        clamped = self.ladder.was_clamped(seed_t, cb.row_time) & cb.row_is_timed
        z = z + self.w_tau(self.ladder.feats(tau, cb.row_is_timed, clamped).to(z.dtype))
        return self.norm(z)


class RowPool(nn.Module):
    """Cross-attention from a row's own cells into its row token.

    ``slots`` queries come from ``[u_r ; s_r]``, keys are the frozen column-name embeddings (so the
    pool attends by column meaning) and values are the cell states. The softmax runs over the row's
    own non-padding cells, computed as a segment softmax over cells grouped by row; a row without
    cells receives zero.
    """

    def __init__(self, d_model: int, d_sig: int, col_name_emb: Tensor, *, slots: int = 4,
                 d_head: int | None = None):
        super().__init__()
        _frozen(self, "col_name_emb", col_name_emb)
        self.slots = slots
        self.d_h = d_head or d_model // 4
        self.w_q = nn.Linear(d_model + d_sig, self.slots * self.d_h, bias=False)
        self.w_k = nn.Linear(col_name_emb.shape[-1], self.d_h, bias=False)
        self.w_v = nn.Linear(d_model, self.d_h, bias=False)
        self.w_o = nn.Linear(self.slots * self.d_h, d_model, bias=False)

    def forward(self, h: Tensor, u: Tensor, s: Tensor, cb) -> Tensor:
        """``h [B, S, d]``, ``u [B, R, d]``, ``s [B, R, d_sig]`` -> updated ``u``."""
        B, R, _ = u.shape
        S = h.shape[1]
        M, dh = self.slots, self.d_h
        row_of = cb.cell_row
        valid = (~cb.is_padding) & (row_of >= 0) & (row_of < R)
        idx = row_of.clamp(0, R - 1)                                      # [B, S]
        flat_g = (idx + torch.arange(B, device=h.device).unsqueeze(1) * R).reshape(-1)

        q = self.w_q(torch.cat([u, s], dim=-1)).view(B, R, M, dh)
        k = self.w_k(self.col_name_emb[cb.cell_col.clamp_min(0)])         # [B, S, dh]
        v = self.w_v(h)                                                   # [B, S, dh]
        q_cell = q.gather(1, idx.view(B, S, 1, 1).expand(B, S, M, dh))    # each cell's own row query
        scores = torch.matmul(q_cell.reshape(B * S, M, dh),
                              k.reshape(B * S, dh, 1)).view(B, S, M) / dh ** 0.5
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid.unsqueeze(-1), neg)

        n_groups = B * R
        gm = scores.new_full((n_groups, M), neg).scatter_reduce(
            0, flat_g.unsqueeze(-1).expand(-1, M), scores.detach().reshape(-1, M),
            reduce="amax", include_self=True)
        e = torch.exp(scores - gm.index_select(0, flat_g).view(B, S, M))
        e = torch.where(valid.unsqueeze(-1), e, torch.zeros_like(e))
        den = e.new_zeros(n_groups, M).index_add(0, flat_g, e.reshape(-1, M))
        a = e / den.index_select(0, flat_g).view(B, S, M).clamp_min(torch.finfo(e.dtype).tiny)

        contrib = torch.matmul(a.reshape(B * S, M, 1), v.reshape(B * S, 1, dh))
        pooled = contrib.new_zeros(n_groups, M, dh).index_add(0, flat_g, contrib)
        return u + self.w_o(pooled.view(B, R, M * dh))


class _ScalarTableGather(torch.autograd.Function):
    """``table[:, idx]`` for a small table and a large, heavily repeated ``idx``, with an
    ``index_add_`` backward (plain indexing's sort-based backward dominates the step time)."""

    @staticmethod
    def forward(ctx, table: Tensor, idx: Tensor) -> Tensor:
        ctx.save_for_backward(idx)
        ctx.shape = table.shape
        return table[:, idx]

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (idx,) = ctx.saved_tensors
        return grad_out.new_zeros(ctx.shape).index_add_(1, idx, grad_out), None


class RowAttention(nn.Module):
    r"""Row attention over the sampled FK graph (edges plus self loops).

    .. math::
        \sigma^{(h)}_{rs} = \tilde q_r^\top \tilde k_s / \sqrt{d_h} + \gamma^{(h)}_{rs}
                            + b_{untimed} 1[\cdot] + b_{clamped} 1[\cdot],
        \qquad \gamma^{(h)}_{rs} = v^{(h)\top} W_\rho N_{role}[\rho(r, s)] + c^{(h)}_{dir(r, s)}

    with temporal RoPE on ``q`` and ``k``. The role bias is computed from the role's name, so an FK
    relation never seen in training still gets a bias.
    """

    def __init__(self, d_model: int, d_sig: int, role_name_emb: Tensor, ladder: TimeLadder, *,
                 n_heads: int = 8):
        super().__init__()
        _frozen(self, "role_name_emb", role_name_emb)
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.ladder = ladder
        self.norm = RMSNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.n_rot = min(2 * ladder.n_freq, self.d_h - self.d_h % 2)
        self.w_rho = nn.Linear(role_name_emb.shape[-1], d_sig, bias=False)
        self.v_head = nn.Parameter(torch.randn(n_heads, d_sig) * d_sig ** -0.5)
        self.c_dir = nn.Parameter(torch.zeros(n_heads, N_DIR))

    def _gamma(self, cb) -> Tensor:
        """``[B, H, R, R]`` role bias, gathered from a per-(head, adj code) table of ``2K + 2``."""
        K1 = self.role_name_emb.shape[0]
        K = K1 - 1
        tab = self.v_head @ self.w_rho(self.role_name_emb).t()            # [H, K + 1]
        code = torch.arange(2 * K + 2, device=tab.device)
        role_ix = torch.where(code > K, code - K, code).clamp(0, K1 - 1)
        role_ix = torch.where(code == 2 * K + 1, torch.zeros_like(code), role_ix)
        dirs = torch.full_like(code, DIR_CHILD)
        dirs = torch.where(code > K, torch.full_like(code, DIR_PARENT), dirs)
        dirs = torch.where(code == 2 * K + 1, torch.full_like(code, DIR_SELF), dirs)
        combined = tab[:, role_ix] + self.c_dir[:, dirs]                   # [H, 2K + 2]
        adj = cb.adj_role.clamp(0, 2 * K + 1)
        g = _ScalarTableGather.apply(combined, adj.reshape(-1))
        return g.view(self.h, *adj.shape).permute(1, 0, 2, 3)

    def forward(self, u: Tensor, cb) -> Tensor:
        B, R, _ = u.shape
        x = self.norm(u)
        q = self.w_q(x).view(B, R, self.h, self.d_h).transpose(1, 2)
        k = self.w_k(x).view(B, R, self.h, self.d_h).transpose(1, 2)
        v = self.w_v(x).view(B, R, self.h, self.d_h).transpose(1, 2)

        seed_t = cb.seed_time.unsqueeze(1)
        theta = self.ladder.theta(self.ladder.tau_from_times(seed_t, cb.row_time), cb.row_is_timed)
        q = self.ladder.rotate(q, theta.unsqueeze(1), self.n_rot)
        k = self.ladder.rotate(k, theta.unsqueeze(1), self.n_rot)

        scores = (q @ k.transpose(-2, -1)) / self.d_h ** 0.5
        scores = scores + self._gamma(cb).to(scores.dtype)
        clamped = self.ladder.was_clamped(seed_t, cb.row_time) & cb.row_is_timed
        scores = scores + self.ladder.time_bias(cb.row_is_timed, cb.row_is_timed,
                                                clamped, clamped).unsqueeze(1).to(scores.dtype)

        valid = cb.row_valid
        mask = (cb.adj_role != 0) & valid.unsqueeze(2) & valid.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        dead = ~mask.any(-1)                                               # padding rows
        a = torch.softmax(scores, dim=-1)
        a = torch.where(dead.view(B, 1, R, 1), torch.zeros_like(a), a)
        out = (a @ v).transpose(1, 2).reshape(B, R, -1)
        return u + self.w_o(out)


class Broadcast(nn.Module):
    """``h_i += W_b u_{row(i)}``: every cell receives its (un-normalised) row token."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w_b = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: Tensor, u: Tensor, cb) -> Tensor:
        idx = cb.cell_row.clamp_min(0)
        return h + self.w_b(torch.gather(u, 1, idx.unsqueeze(-1).expand(-1, -1, u.shape[-1])))


class RowMoE(nn.Module):
    r"""Row-level mixture of experts, routed on the row signature with a cosine router.

    ``logits_e = cos(s_r, w_e) / T`` with a learned temperature ``T``; top-``k`` routed experts plus
    ``num_shared`` always-on experts. Returns the updated row tokens and the auxiliary loss
    ``lambda_ortho * ||W_hat W_hat^T - I||^2 + lambda_balance * E sum_e f_e p_e``.
    """

    def __init__(self, d_model: int, d_ff: int, d_sig: int, *, num_experts: int = 4, k: int = 2,
                 num_shared: int = 1, lambda_ortho: float = 0.5, lambda_balance: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.lambda_ortho = lambda_ortho
        self.lambda_balance = lambda_balance
        self.norm = RMSNorm(d_model)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))
        self.shared = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_shared)) or None
        self.w_g = nn.Parameter(torch.randn(num_experts, d_sig) * d_sig ** -0.5)
        self.log_T = nn.Parameter(torch.zeros(()))

    def gates(self, z: Tensor) -> Tensor:
        return self._gates_topi(z)[0]

    def _gates_topi(self, z: Tensor) -> tuple[Tensor, Tensor]:
        logits = (F.normalize(z, dim=-1) @ F.normalize(self.w_g, dim=-1).t()) / self.log_T.exp()
        return top_k_gates(logits, self.k)

    def balance_loss(self, g: Tensor, valid: Tensor) -> Tensor:
        """``E sum_e f_e p_e`` over valid rows: ``f_e`` top-1 share, ``p_e`` mean gate."""
        gv = g.reshape(-1, self.num_experts)[valid.reshape(-1)]
        if gv.numel() == 0:
            return g.sum() * 0.0
        p = gv.mean(0)
        f = torch.bincount(gv.argmax(-1), minlength=self.num_experts).to(gv.dtype) / gv.shape[0]
        return self.num_experts * (f * p).sum()

    def forward(self, u: Tensor, z: Tensor, cb) -> tuple[Tensor, Tensor]:
        g, topi = self._gates_topi(z)
        x = self.norm(u)
        valid = cb.row_valid
        y = sparse_combine(x, g, topi, self.experts, valid=valid) + apply_shared(
            x, self.shared, valid=valid)
        aux = (self.lambda_ortho * orthogonality_loss(self.w_g)
               + self.lambda_balance * self.balance_loss(g, valid))
        return u + y, aux
