"""Fixed log-time frequency ladder: the only way time enters the model.

For a row with timestamp ``t_r`` and seed time ``t*`` (seconds),

    Delta_r = max(0, t* - t_r),   tau_r = log(1 + Delta_r),   theta_r^(k) = omega_k * tau_r,

with ``n_freq = 8`` frequencies log-spaced over ``[0.3, 5.0]``. The frequencies are a fixed,
non-persistent buffer, never learned and never fitted to a database. Rotary attention on ``theta``
makes scores depend on ``tau_i - tau_j = log((1 + Delta_i) / (1 + Delta_j))``, a relative log-time.

Two states that ``theta = 0`` cannot express get explicit flags: untimed rows, and rows dated after
their seed (whose Delta was clamped to 0). Attention adds a learned scalar for each
(:meth:`TimeLadder.time_bias`); the signature features carry them as channels
(:meth:`TimeLadder.feats`).

Delta and tau are computed in float64: unix timestamps are ~1.7e9 s, where float32 spacing is ~128 s.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeLadder(nn.Module):
    def __init__(self, n_freq: int = 8, omega: tuple[float, float] = (0.3, 5.0)):
        super().__init__()
        lo, hi = float(omega[0]), float(omega[1])
        if not 0.0 < lo <= hi:
            raise ValueError(f"need 0 < omega_min <= omega_max, got {omega}")
        self.n_freq = int(n_freq)
        w = (torch.logspace(math.log10(lo), math.log10(hi), self.n_freq, dtype=torch.float64)
             if self.n_freq > 1 else torch.tensor([lo], dtype=torch.float64))
        self.register_buffer("omega", w, persistent=False)
        self.b_untimed = nn.Parameter(torch.zeros(()))
        self.b_clamped = nn.Parameter(torch.zeros(()))

    @property
    def feat_dim(self) -> int:
        """Width of :meth:`feats`: ``2 * n_freq`` sinusoids plus the clamped flag."""
        return 2 * self.n_freq + 1

    @staticmethod
    def tau_from_times(seed_time: Tensor, row_time: Tensor) -> Tensor:
        delta = (seed_time.to(torch.float64) - row_time.to(torch.float64)).clamp(min=0.0)
        return torch.log1p(delta)

    @staticmethod
    def was_clamped(seed_time: Tensor, row_time: Tensor) -> Tensor:
        """``t_row > t*``: rows whose Delta the clamp set to 0."""
        return row_time.to(torch.float64) > seed_time.to(torch.float64)

    def theta(self, tau: Tensor, is_timed: Tensor | None = None) -> Tensor:
        """``[..., n_freq]``, zero where ``is_timed`` is False."""
        th = tau.to(torch.float64).unsqueeze(-1) * self.omega.to(tau.device)
        if is_timed is not None:
            th = th * is_timed.unsqueeze(-1).to(th.dtype)
        return th

    def feats(self, tau: Tensor, is_timed: Tensor | None = None,
              is_clamped: Tensor | None = None) -> Tensor:
        """``[sin theta ; cos theta ; 1[clamped]]``; the sinusoid pair is all-zero for untimed rows,
        so "unknown time" stays separable from "Delta = 0"."""
        th = self.theta(tau)
        f = torch.cat((torch.sin(th), torch.cos(th)), dim=-1)
        if is_timed is not None:
            f = f * is_timed.unsqueeze(-1).to(f.dtype)
        flag = (torch.zeros_like(f[..., :1]) if is_clamped is None
                else is_clamped.unsqueeze(-1).to(f.dtype))
        return torch.cat((f, flag), dim=-1)

    def rotate(self, x: Tensor, theta: Tensor, n_rot_dims: int | None = None) -> Tensor:
        """Rotate interleaved pairs of the leading ``n_rot_dims`` of ``x`` by ``theta``; the rest of
        ``x`` passes through. ``theta[..., :-1]`` must broadcast against ``x[..., :-1]``."""
        n_rot = 2 * theta.shape[-1] if n_rot_dims is None else int(n_rot_dims)
        if n_rot % 2 or n_rot > x.shape[-1] or n_rot > 2 * theta.shape[-1]:
            raise ValueError(f"invalid n_rot_dims={n_rot} for x dim {x.shape[-1]} and "
                             f"{theta.shape[-1]} frequencies")
        m = n_rot // 2
        if m == 0:
            return x
        th = theta[..., :m]
        cos, sin = torch.cos(th).to(x.dtype), torch.sin(th).to(x.dtype)
        rot, rest = x[..., :n_rot], x[..., n_rot:]
        x0, x1 = rot[..., 0::2], rot[..., 1::2]
        out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
        if rest.shape[-1] == 0:
            return out
        return torch.cat((out, rest.expand(*out.shape[:-1], rest.shape[-1])), dim=-1)

    def time_bias(self, is_timed_q: Tensor, is_timed_k: Tensor,
                  is_clamped_q: Tensor, is_clamped_k: Tensor) -> Tensor:
        """``b_untimed * 1[q or k untimed] + b_clamped * 1[q or k clamped]`` -> ``[..., Sq, Sk]``."""
        untimed = ~(is_timed_q.unsqueeze(-1) & is_timed_k.unsqueeze(-2))
        clamped = is_clamped_q.unsqueeze(-1) | is_clamped_k.unsqueeze(-2)
        return (self.b_untimed * untimed.to(self.b_untimed.dtype)
                + self.b_clamped * clamped.to(self.b_clamped.dtype))
