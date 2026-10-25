"""RelBench evaluation: predict every seed of a split and score it with ``task.evaluate``."""
from __future__ import annotations

import numpy as np
import torch

from ..data.collate import to_cell_batch
from ..data.graph import coerce_binary_target, make_loader


@torch.no_grad()
def predict_split(module, bundle, task, split: str, *, seq_len: int, num_neighbors=None,
                  batch_size: int = 64, num_workers: int = 0, device=None) -> np.ndarray:
    """Predictions for every row of ``task``'s ``split`` table, in table order: probabilities for
    binary tasks, values in original units for regression (clipped if the module has a clamp)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module.to(device)
    was_training = module.training
    module.eval()
    loader = make_loader(bundle, task, split, num_neighbors=num_neighbors, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    pred = np.full(len(task.get_table(split).df), np.nan, dtype=np.float64)
    for raw in loader:
        cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=seq_len).to(device)
        out = module(cb).float().cpu()
        p = module.destandardize(out) if module.task_type == "regression" else torch.sigmoid(out)
        # the seeds are the loader's input rows; `input_id` maps them to table rows
        pred[raw[task.entity_table].input_id.cpu().numpy()] = p.numpy()
    if was_training:
        module.train()
    assert not np.isnan(pred).any(), f"{split}: {int(np.isnan(pred).sum())} rows got no prediction"
    if module.task_type == "regression" and module.clamp is not None:
        pred = np.clip(pred, *module.clamp)
    return pred


def score(task, split: str, pred: np.ndarray) -> dict[str, float]:
    """RelBench metrics of ``pred`` on ``split`` (the labels of TEST are read from the task)."""
    target = coerce_binary_target(task.get_table(split, mask_input_cols=False), task)
    return {k: float(v) for k, v in task.evaluate(pred, target).items()}


def evaluate_split(module, bundle, task, split: str = "test", **kw) -> tuple[dict, np.ndarray]:
    pred = predict_split(module, bundle, task, split, **kw)
    return score(task, split, pred), pred
