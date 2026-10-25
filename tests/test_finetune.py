"""Fine-tuning: losses, target scaling, model selection and fitting a batch."""
from __future__ import annotations

import pytest
import torch

from conftest import ENTITY, make_batch, schema_of, tiny_config
from moregt.data.collate import to_cell_batch
from moregt.train.finetune import EpochSelector, FinetuneModule, average_states
from moregt.train.losses import task_loss


def test_losses_ignore_unlabelled_seeds():
    out = torch.tensor([0.0, 5.0])
    y = torch.tensor([1.0, 100.0])
    has = torch.tensor([True, False])
    assert task_loss(out, y, has, "regression", "l1").item() == pytest.approx(1.0)
    assert task_loss(out, y, has, "regression", "mse").item() == pytest.approx(1.0)
    y_bin = torch.tensor([1.0, 0.0])
    assert task_loss(out, y_bin, has, "binary").item() == pytest.approx(0.6931, abs=1e-4)
    assert task_loss(out, y_bin, torch.zeros(2, dtype=torch.bool), "binary").item() == 0.0


@pytest.mark.parametrize("log1p", [False, True])
def test_target_scaling_round_trip(bundle, log1p):
    m = FinetuneModule(bundle, schema_of(bundle), ENTITY, task_type="regression", config=tiny_config(),
                       target_mean=1.5, target_std=2.0, target_log=log1p)
    y = torch.tensor([0.0, 3.0, 250.0])
    assert torch.allclose(m.destandardize(m.standardize(y)), y, rtol=1e-5, atol=1e-4)


class _Trainer:
    sanity_checking = False

    def __init__(self, v):
        self.callback_metrics = {"val/auroc": torch.tensor(v)}


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.step = torch.nn.Parameter(torch.zeros(()), requires_grad=False)


def test_selector_keeps_best_and_top_k():
    sel = EpochSelector("val/auroc", "max", k=2)
    mod = _Module()
    for epoch, score in enumerate([0.6, 0.8, 0.7, 0.5]):
        mod.w.data.fill_(float(epoch))
        sel.on_validation_end(_Trainer(score), mod)
    assert sel.best_score == pytest.approx(0.8) and sel.best_state["w"].item() == 1.0
    assert sel.averaged_state()["w"].item() == pytest.approx(1.5)    # epochs 1 and 2


def test_selector_minimises_for_regression():
    sel = EpochSelector("val/auroc", "min", k=1)
    mod = _Module()
    for epoch, score in enumerate([3.0, 1.0, 2.0]):
        mod.w.data.fill_(float(epoch))
        sel.on_validation_end(_Trainer(score), mod)
    assert sel.best_state["w"].item() == 1.0 and sel.averaged_state() is sel.best_state


def test_average_states_keeps_integer_tensors():
    a = {"w": torch.tensor([0.0, 2.0]), "n": torch.tensor(3)}
    b = {"w": torch.tensor([2.0, 4.0]), "n": torch.tensor(9)}
    avg = average_states([a, b])
    assert torch.equal(avg["w"], torch.tensor([1.0, 3.0])) and avg["n"].item() == 3


@pytest.mark.parametrize("kind", ["binary", "regression"])
def test_fits_one_batch(bundle, kind):
    torch.manual_seed(0)
    targets = (1.0, 0.0) if kind == "binary" else (2.0, -1.0)
    cb = to_cell_batch(make_batch(targets=targets), bundle, ENTITY, seq_len=24)
    m = FinetuneModule(bundle, schema_of(bundle), ENTITY, task_type=kind, config=tiny_config(),
                       regression_loss="l1")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        logits, aux = m.model(cb)
        loss = task_loss(logits.squeeze(-1), m.standardize(cb.target), cb.has_target, kind, "l1")
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0]
