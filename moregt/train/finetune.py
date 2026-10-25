"""Supervised fine-tuning on one RelBench entity task.

Binary tasks train with BCE. Regression targets are scaled with TRAIN statistics (``zscore``,
``log1p`` = z-score of ``log(1 + y)``, or ``raw``), trained with MSE or L1, and predicted back in
original units. The router auxiliary loss is added with weight ``lambda_ortho``.

Model selection reads validation only. Every epoch is scored on RelBench val, and two models are
kept: the single best epoch on val (the reported model), and the average of the weights of the
``swa_window`` best epochs on the selection split. The selection split is val, or with
``select_split="holdout"`` the last ``holdout_windows`` timestamps of the train table, which are then
removed from training. Early stopping follows the selection split. Weight averaging is valid because
every normalisation layer is RMSNorm or LayerNorm, so there are no running statistics.
"""
from __future__ import annotations

import time

import numpy as np
import pytorch_lightning as pl
import torch

from ..data.collate import to_cell_batch
from ..data.graph import make_loader
from ..eval.metrics import binary_metrics, regression_metrics
from ..model.moregt import MoREGT, MoREGTConfig
from .losses import task_loss
from .pretrain import load_trunk


def task_kind(task) -> str:
    from relbench.base import TaskType

    if task.task_type == TaskType.REGRESSION:
        return "regression"
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return "binary"
    raise ValueError(f"unsupported task type {task.task_type}")


def _train_targets(task) -> np.ndarray:
    y = task.get_table("train").df[task.target_col].to_numpy(dtype="float64")
    return y[~np.isnan(y)]


def target_stats(task, log1p: bool = False) -> tuple[float, float]:
    """TRAIN mean and std of the target (of ``log(1 + y)`` with ``log1p``)."""
    y = _train_targets(task)
    if log1p:
        y = np.log1p(np.clip(y, 0, None))
    return float(y.mean()), float(max(y.std(), 1e-6))


def target_clamp(task, pct: float) -> tuple[float, float]:
    """TRAIN target percentiles ``[pct, 100 - pct]`` used to clip regression predictions."""
    lo, hi = np.percentile(_train_targets(task), [pct, 100.0 - pct])
    return float(lo), float(hi)


def split_holdout(task, windows: int):
    """``(train_table, holdout_table)``: the last ``windows`` timestamps of TRAIN held out."""
    src = task.get_table("train")
    df = src.df
    ts = np.sort(df[task.time_col].unique())
    if windows >= len(ts):
        raise ValueError(f"holdout_windows={windows} would hold out all {len(ts)} train timestamps")
    cut = ts[-windows]

    def table(d):
        return type(src)(df=d.reset_index(drop=True), fkey_col_to_pkey_table=src.fkey_col_to_pkey_table,
                         pkey_col=src.pkey_col, time_col=src.time_col)

    return table(df[df[task.time_col] < cut]), table(df[df[task.time_col] >= cut])


class FinetuneData(pl.LightningDataModule):
    """Train / validation loaders. With a holdout, validation has two loaders: holdout, then val."""

    def __init__(self, bundle, task, *, num_neighbors=None, batch_size: int = 128,
                 num_workers: int = 0, holdout_windows: int = 0):
        super().__init__()
        self.bundle, self.task = bundle, task
        self.kw = dict(num_neighbors=num_neighbors or [12, 12], batch_size=batch_size,
                       num_workers=num_workers)
        self.train_table = self.holdout_table = None
        if holdout_windows > 0:
            self.train_table, self.holdout_table = split_holdout(task, holdout_windows)

    def train_dataloader(self):
        return make_loader(self.bundle, self.task, "train", shuffle=True, table=self.train_table,
                           **self.kw)

    def val_dataloader(self):
        val = make_loader(self.bundle, self.task, "val", shuffle=False, **self.kw)
        if self.holdout_table is None:
            return val
        return [make_loader(self.bundle, self.task, "train", shuffle=False, table=self.holdout_table,
                            **self.kw), val]


class FinetuneModule(pl.LightningModule):
    def __init__(self, bundle, schema, entity_table: str, *, task_type: str,
                 config: MoREGTConfig | None = None, init_from: str | None = None,
                 target_mean: float = 0.0, target_std: float = 1.0, target_log: bool = False,
                 clamp: tuple[float, float] | None = None, regression_loss: str = "mse",
                 optimizer: str = "adamw", lr: float = 3e-4, weight_decay: float = 0.01,
                 lambda_ortho: float = 0.5, seq_len: int = 1024):
        super().__init__()
        if optimizer not in ("adamw", "adam"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self.bundle = bundle
        self.entity_table = entity_table
        self.task_type = task_type
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if target_std > 0 else 1.0
        self.target_log = target_log
        self.clamp = clamp
        self.regression_loss = regression_loss
        self.optimizer_name = optimizer
        self.lr, self.weight_decay = lr, weight_decay
        self.lambda_ortho = lambda_ortho
        self.seq_len = seq_len
        self.model = MoREGT(bundle, schema, config)
        if init_from:
            load_trunk(self.model, init_from)
        #: metric prefix of each validation loader (the first one drives selection)
        self.val_prefixes = ["val"]
        self._val: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def forward(self, cb):
        return self.model(cb)[0].squeeze(-1)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        return to_cell_batch(batch, self.bundle, self.entity_table, seq_len=self.seq_len).to(device)

    def standardize(self, y):
        if self.task_type != "regression":
            return y
        t = torch.log1p(y.clamp_min(0)) if self.target_log else y
        return (t - self.target_mean) / self.target_std

    def destandardize(self, out):
        """Model output -> target in original units."""
        y = out * self.target_std + self.target_mean
        return torch.expm1(y).clamp_min(0) if self.target_log else y

    def training_step(self, cb, _idx):
        logits, aux = self.model(cb)
        loss = task_loss(logits.squeeze(-1), self.standardize(cb.target), cb.has_target,
                         self.task_type, self.regression_loss) + self.lambda_ortho * aux
        self.log("train/loss", loss, prog_bar=True, batch_size=int(cb.num_seeds))
        return loss

    def on_before_optimizer_step(self, optimizer):
        # an occasional non-finite gradient is zeroed rather than allowed to poison the weights
        for p in self.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def validation_step(self, cb, _idx, dataloader_idx: int = 0):
        out = self(cb)
        m = cb.has_target
        if m.any():
            self._val.setdefault(int(dataloader_idx), []).append(
                (out[m].detach().cpu(), cb.target[m].detach().cpu()))

    def on_validation_epoch_end(self):
        for idx, acc in sorted(self._val.items()):
            out = torch.cat([a for a, _ in acc])
            target = torch.cat([b for _, b in acc])
            if self.task_type == "regression":
                pred = self.destandardize(out)
                if self.clamp is not None:
                    pred = pred.clamp(*self.clamp)
                metrics = regression_metrics(pred, target)
            else:
                metrics = binary_metrics(out, target)
            for k, v in metrics.items():
                self.log(f"{self.val_prefixes[idx]}/{k}", v, add_dataloader_idx=False)
        self._val.clear()

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.optimizer_name == "adamw" else torch.optim.Adam
        return cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def _snapshot(module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def average_states(states: list[dict]) -> dict:
    """Element-wise mean of floating-point tensors; other tensors are taken from ``states[0]``."""
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            acc = v.double().clone()
            for other in states[1:]:
                acc += other[k].double()
            out[k] = (acc / len(states)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


class EpochSelector(pl.Callback):
    """Keeps the weights of the best epoch, and of the ``k`` best epochs, by a validation metric."""

    def __init__(self, monitor: str, mode: str, k: int = 1):
        self.monitor, self.mode, self.k = monitor, mode, k
        self.best_score: float | None = None
        self.best_state: dict | None = None
        self.best_metrics: dict = {}
        self.top: list[tuple[float, dict]] = []

    def _better(self, a: float, b: float | None) -> bool:
        return b is None or (a > b if self.mode == "max" else a < b)

    def on_validation_end(self, trainer, pl_module) -> None:
        v = trainer.callback_metrics.get(self.monitor)
        if trainer.sanity_checking or v is None or float(v) != float(v):
            return
        score = float(v)
        state = None
        if self.k > 1 and (len(self.top) < self.k or self._better(score, self.top[-1][0])):
            state = _snapshot(pl_module)
            self.top.append((score, state))
            self.top.sort(key=lambda t: -t[0] if self.mode == "max" else t[0])
            del self.top[self.k:]
        if self._better(score, self.best_score):
            self.best_score = score
            self.best_state = state if state is not None else _snapshot(pl_module)
            self.best_metrics = {k: float(m) for k, m in trainer.callback_metrics.items()
                                 if k.startswith(("val/", "sel/"))}

    def averaged_state(self) -> dict | None:
        """Mean weights of the ``k`` best epochs (the best epoch if fewer than two were kept)."""
        if len(self.top) > 1:
            return average_states([s for _, s in self.top])
        return self.best_state


class EpochLog(pl.Callback):
    """One line per epoch: the monitored metric and the elapsed time."""

    def __init__(self, monitor: str, max_epochs: int):
        self.monitor, self.max_epochs = monitor, max_epochs
        self._t0 = None

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._t0 is None:
            return
        v = trainer.callback_metrics.get(self.monitor)
        print(f"epoch {trainer.current_epoch + 1}/{self.max_epochs}  {self.monitor}="
              f"{float(v) if v is not None else float('nan'):.4f}  "
              f"elapsed {(time.time() - self._t0) / 60:.1f} min", flush=True)


def finetune(bundle, task, schema, *, config: MoREGTConfig | None = None,
             init_from: str | None = None, seed: int = 0, epochs: int = 12, patience: int = 4,
             batch_size: int = 128, limit_train_batches: int | None = 1000, num_workers: int = 0,
             num_neighbors: list[int] | None = None, seq_len: int = 1024, lr: float = 3e-4,
             weight_decay: float = 0.01, optimizer: str = "adamw", regression_loss: str = "mse",
             target_scaling: str = "log1p", clamp_pct: float | None = None,
             select_split: str = "val", holdout_windows: int = 2, swa_window: int = 5,
             lambda_ortho: float = 0.5, accelerator: str = "auto"):
    """Fine-tune on ``task`` and return ``(module, best_state, swa_state, info)``.

    ``best_state`` is the best epoch on RelBench val and ``swa_state`` the averaged ``swa_window``
    best epochs on the selection split; ``info`` holds the validation metrics of the best epoch.
    """
    from ..utils import limit_torch_threads, seed_everything

    seed_everything(seed)
    limit_torch_threads()
    kind = task_kind(task)
    if select_split not in ("val", "holdout"):
        raise ValueError(f"unknown select_split {select_split!r}")
    if target_scaling not in ("zscore", "log1p", "raw"):
        raise ValueError(f"unknown target_scaling {target_scaling!r}")
    log1p = kind == "regression" and target_scaling == "log1p"
    mean, std = (target_stats(task, log1p=log1p)
                 if kind == "regression" and target_scaling != "raw" else (0.0, 1.0))
    clamp = target_clamp(task, clamp_pct) if kind == "regression" and clamp_pct else None

    module = FinetuneModule(bundle, schema, task.entity_table, task_type=kind, config=config,
                            init_from=init_from, target_mean=mean, target_std=std, target_log=log1p,
                            clamp=clamp, regression_loss=regression_loss, optimizer=optimizer, lr=lr,
                            weight_decay=weight_decay, lambda_ortho=lambda_ortho, seq_len=seq_len)
    holdout = select_split == "holdout"
    data = FinetuneData(bundle, task, num_neighbors=num_neighbors, batch_size=batch_size,
                        num_workers=num_workers, holdout_windows=holdout_windows if holdout else 0)
    if holdout:
        module.val_prefixes = ["sel", "val"]

    metric, mode = ("auroc", "max") if kind == "binary" else ("mae", "min")
    sel_monitor = f"{'sel' if holdout else 'val'}/{metric}"
    best_val = EpochSelector(f"val/{metric}", mode, k=1 if holdout else swa_window)
    callbacks = [best_val, EpochLog(sel_monitor, epochs),
                 pl.callbacks.EarlyStopping(monitor=sel_monitor, mode=mode, patience=patience,
                                            strict=False)]
    best_sel = best_val
    if holdout:
        best_sel = EpochSelector(sel_monitor, mode, k=swa_window)
        callbacks.insert(0, best_sel)

    trainer = pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, logger=False,
        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
        num_sanity_val_steps=0, gradient_clip_val=1.0, check_val_every_n_epoch=1,
        limit_train_batches=limit_train_batches or 1.0, callbacks=callbacks,
    )
    trainer.fit(module, data)
    train_table = data.train_table if data.train_table is not None else task.get_table("train")
    info = {"val_best": best_val.best_score, "sel_best": best_sel.best_score,
            "epochs_trained": int(trainer.current_epoch), "train_rows": len(train_table.df),
            **best_val.best_metrics}
    return module, best_val.best_state, best_sel.averaged_state(), info
