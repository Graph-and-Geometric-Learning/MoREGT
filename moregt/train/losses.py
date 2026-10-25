"""Task losses over the seeds that carry a label."""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def _masked(fn, pred: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    if not bool(has_target.any()):
        return pred.sum() * 0.0
    return fn(pred[has_target], target[has_target].to(pred.dtype))


def masked_bce(logits: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    return _masked(F.binary_cross_entropy_with_logits, logits, target, has_target)


def masked_mse(pred: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    return _masked(F.mse_loss, pred, target, has_target)


def masked_l1(pred: Tensor, target: Tensor, has_target: Tensor) -> Tensor:
    return _masked(F.l1_loss, pred, target, has_target)


REGRESSION_LOSSES = {"mse": masked_mse, "l1": masked_l1}


def task_loss(out: Tensor, target: Tensor, has_target: Tensor, task_type: str,
              regression_loss: str = "mse") -> Tensor:
    """BCE for binary tasks; ``regression_loss`` (on the scaled target) for regression."""
    if task_type == "binary":
        return masked_bce(out, target, has_target)
    if regression_loss not in REGRESSION_LOSSES:
        raise ValueError(f"unknown regression loss {regression_loss!r}; expected {sorted(REGRESSION_LOSSES)}")
    return REGRESSION_LOSSES[regression_loss](out, target, has_target)
