"""FlexAttention backend for cell attention.

The padding mask becomes a block mask, so all-padding blocks are skipped, and the additive time flags
become a ``score_mod``, so neither the ``[B, S, S]`` mask nor the bias is materialised. Numerically
equivalent to the SDPA path in ``two_level.CellAttention``. Requires torch >= 2.5 and a head
dimension of at least 16; compiled on CUDA, eager elsewhere.
"""
from __future__ import annotations

import torch
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX = True
except Exception:  # pragma: no cover - depends on the installed torch
    create_block_mask = flex_attention = None  # type: ignore[assignment]
    HAS_FLEX = False

FLEX_BLOCK = 128
FLEX_MIN_HEAD_DIM = 16

_COMPILED = None
_COMPILED_BUILDER = None


def _flex_fn(device: torch.device):
    global _COMPILED
    if device.type != "cuda":
        return flex_attention
    if _COMPILED is None:
        _COMPILED = torch.compile(flex_attention, dynamic=False)
    return _COMPILED


def build_cell_block_mask(is_padding: Tensor, *, block_size: int = FLEX_BLOCK):
    """Block mask for ``real[q] & real[k]``; built once per forward and shared by all blocks."""
    global _COMPILED_BUILDER
    if not HAS_FLEX:
        raise RuntimeError("the flex cell-attention backend needs torch >= 2.5")
    real = ~is_padding
    B, S = real.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return real[b, q_idx] & real[b, kv_idx]

    build = create_block_mask
    if real.is_cuda:
        if _COMPILED_BUILDER is None:
            _COMPILED_BUILDER = torch.compile(create_block_mask)
        build = _COMPILED_BUILDER
    return build(mask_mod, B, None, S, S, device=real.device, BLOCK_SIZE=block_size)


def cell_score_mod(is_timed: Tensor, is_clamped: Tensor, b_untimed: Tensor, b_clamped: Tensor, *,
                   dtype: torch.dtype | None = None):
    """``score + b_untimed * 1[q or k untimed] + b_clamped * 1[q or k clamped]`` as a score_mod.

    Each learned scalar is expanded to ``[B, S]`` and indexed once inside the closure: inductor cannot
    lower the backward of a 0-dim captured tensor that requires grad, while the broadcast form passes
    gradients to both scalars.
    """
    B, S = is_timed.shape
    dt = dtype or b_untimed.dtype
    ub = b_untimed.to(dt).expand(B, S).contiguous()
    cb = b_clamped.to(dt).expand(B, S).contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        untimed = ~(is_timed[b, q_idx] & is_timed[b, kv_idx])
        clamped = is_clamped[b, q_idx] | is_clamped[b, kv_idx]
        return (score
                + ub[b, q_idx] * untimed.to(score.dtype)
                + cb[b, q_idx] * clamped.to(score.dtype))

    return score_mod


def flex_cell_attention(q: Tensor, k: Tensor, v: Tensor, *, block_mask, score_mod=None) -> Tensor:
    """``[B, H, S, d_h]`` in and out; all-padding rows come back as zeros."""
    return torch.nan_to_num(_flex_fn(q.device)(q, k, v, score_mod=score_mod, block_mask=block_mask))
