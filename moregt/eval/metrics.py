"""Validation metrics (TEST uses RelBench's own ``task.evaluate``)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """``{ap, auroc}`` from logits; NaN when only one class is present."""
    prob = torch.sigmoid(logits.detach().float().cpu()).numpy()
    y = target.detach().cpu().numpy().astype(int)
    if len(np.unique(y)) < 2:
        return {"ap": float("nan"), "auroc": float("nan")}
    return {"ap": float(average_precision_score(y, prob)), "auroc": float(roc_auc_score(y, prob))}


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """``{mae, rmse}`` in the target's original units."""
    p = pred.detach().float().cpu().numpy().ravel()
    y = target.detach().float().cpu().numpy().ravel()
    diff = p - y
    return {"mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2)))}
