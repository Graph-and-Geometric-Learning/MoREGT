"""Masked-cell pretraining on one or several databases.

Each step draws a batch of seeds from one table of one database, masks cells of the seed rows
(:mod:`moregt.data.masking`) and reconstructs them from the relational context
(:class:`~moregt.model.mlm_head.MaskedCellHead`). The loss is ``L_num + lambda_cat L_cat +
lambda_ortho aux``.

Model selection uses ``val/skill``: per modality, the validation loss summed over the epoch divided by
the summed loss of a no-skill reference (the in-batch column mean for numerical cells, a uniform guess
for categorical cells), averaged over the two modalities. ``1`` means no skill; lower is better.

Optimisation is AdamW with linear warmup and linear decay, over the step budget or, with
``max_time_s``, over a wall-clock budget. Non-finite gradient steps are skipped.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from ..data.collate import CellBatch, to_cell_batch
from ..data.masking import build_column_target_spec, gather_masked_targets, sample_cell_mask
from ..data.pretrain_loader import split_key
from ..model.mlm_head import MaskedCellHead
from ..model.moregt import MoREGT, MoREGTConfig
from ..model.schema_bank import SchemaBank, SchemaTables

#: No-skill references below this are degenerate (near-constant masked columns) and are skipped.
MIN_REF = 1e-3
#: Skill readings above this are treated as a broken metric, not as stagnation.
SKILL_SANITY = 10.0


def linear_warmup_decay(max_steps: int, warmup_frac: float = 0.05):
    """LR multiplier: linear 0 -> 1 over ``warmup_frac`` of the steps, then linear 1 -> 0."""
    warmup = max(int(round(warmup_frac * max_steps)), 1)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, 1.0 - (step - warmup) / max(max_steps - warmup, 1))

    return fn


class KeyedCollate:
    """``(key, raw batch) -> CellBatch`` for the pretraining stream; runs inside dataloader workers."""

    def __init__(self, bundles: dict, seq_len: int):
        self.bundles = bundles
        self.seq_len = seq_len

    def __call__(self, key: str, raw) -> CellBatch:
        dataset, table = split_key(key)
        return to_cell_batch(raw, self.bundles[dataset], table, seq_len=self.seq_len)


class PretrainModule(pl.LightningModule):
    """MoREGT plus a masked-cell head, trained on a stream of ``(key, CellBatch)``.

    ``bundles`` and ``schemas`` map dataset names to graphs and name embeddings; the model is built
    on the first dataset and, with several, a :class:`SchemaBank` binds each batch's tables.
    """

    def __init__(self, bundles: dict, schemas: dict, config: MoREGTConfig | None = None, *,
                 p_random: float = 0.0, seed_frac: float = 0.5, lambda_cat: float = 4.0,
                 lambda_ortho: float = 0.5, lr: float = 3e-4, weight_decay: float = 0.1,
                 betas: tuple[float, float] = (0.9, 0.95), max_steps: int = 300_000,
                 max_time_s: float | None = None, warmup_frac: float = 0.05, seq_len: int = 1024,
                 mask_seed: int = 0):
        super().__init__()
        self.datasets = list(bundles)
        anchor = self.datasets[0]
        self.bundles = dict(bundles)
        self.p_random, self.seed_frac = p_random, seed_frac
        self.lambda_ortho = lambda_ortho
        self.lr, self.weight_decay, self.betas = lr, weight_decay, betas
        self.max_steps_budget, self.max_time_s, self.warmup_frac = max_steps, max_time_s, warmup_frac
        self.seq_len = seq_len
        self.mask_seed = mask_seed

        self.model = MoREGT(bundles[anchor], schemas[anchor], config)
        self.specs = {ds: build_column_target_spec(b) for ds, b in bundles.items()}
        self.bank = (SchemaBank({ds: SchemaTables.build(bundles[ds], schemas[ds]) for ds in self.datasets})
                     if len(self.datasets) > 1 else None)
        self.head = MaskedCellHead(self.model.config.d_model, int(schemas[anchor].category.shape[-1]),
                                   lambda_cat=lambda_cat)
        self.collate = KeyedCollate(self.bundles, seq_len)
        self._t0 = time.time()
        self._bad_steps = self._bad_steps_total = 0
        # epoch sums [n_num, n_num*loss_num, n_num*ref_num, n_cat, n_cat*loss_cat, n_cat*ref_cat]
        self._val_sums = torch.zeros(6)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        key, raw = batch
        cb = raw if isinstance(raw, CellBatch) else self.collate(key, raw)
        return key, cb.to(device)

    def _masked_loss(self, key: str, cb, *, step_seed: int):
        dataset, _table = split_key(key)
        spec = self.specs[dataset]
        if self.bank is not None:
            self.bank.bind(self.model, dataset)
        g = torch.Generator(device=cb.cell_col.device).manual_seed(step_seed)
        mask, seed = sample_cell_mask(cb, spec, p_random=self.p_random, seed_frac=self.seed_frac,
                                      generator=g)
        targets = gather_masked_targets(cb, spec, mask, seed)
        _logits, aux, cells = self.model(cb, cell_mask=mask, return_cells=True)
        loss, stats = self.head(cells, targets, spec, self.model.category_table())
        return loss, aux, stats

    def training_step(self, batch, batch_idx):
        key, cb = batch
        # the mask is seeded by the global step: reproducible on resume and identical across ranks
        loss, aux, stats = self._masked_loss(key, cb,
                                             step_seed=self.mask_seed * 1_000_003 + self.global_step)
        total = loss + self.lambda_ortho * aux
        bs = int(cb.num_seeds)
        self.log_dict({"train/loss": total, "train/mlm": loss, "train/aux": aux}, prog_bar=True,
                      batch_size=bs)
        self.log_dict({f"train/{k}": v for k, v in stats.items()}, batch_size=bs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], batch_size=bs)
        return total

    def validation_step(self, batch, batch_idx):
        key, cb = batch
        # a fixed seed per batch, so every validation asks the same questions
        loss, _aux, stats = self._masked_loss(key, cb, step_seed=7_919 + batch_idx)
        bs = int(cb.num_seeds)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log_dict({f"val/{k}": v for k, v in stats.items()}, batch_size=bs, sync_dist=True)
        for off, (lk, rk, nk) in enumerate((("loss_num", "loss_num_mean", "n_num"),
                                            ("loss_cat", "loss_cat_uniform", "n_cat"))):
            n, lv, rv = stats.get(nk, 0.0), stats.get(lk), stats.get(rk)
            if lv is None or rv is None or n <= 0 or rv <= MIN_REF:
                continue
            self._val_sums[3 * off:3 * off + 3] += torch.tensor([n, n * lv, n * rv],
                                                                device=self._val_sums.device)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_sums = torch.zeros(6, device=self.device)

    def on_validation_epoch_end(self) -> None:
        acc = self._val_sums
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
        parts = []
        for off, name in enumerate(("num", "cat")):
            n, num, den = (float(acc[3 * off + j]) for j in range(3))
            if n > 0 and den > 0:
                parts.append(num / den)
                self.log(f"val/skill_{name}", num / den)
        if parts:
            self.log("val/skill", sum(parts) / len(parts), prog_bar=True)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for p in (*self.model.parameters(), *self.head.parameters()):
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": self.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=self.lr, betas=self.betas)
        if self.max_time_s:
            warm, total = self.warmup_frac, float(self.max_time_s)

            def fn(_step: int) -> float:
                prog = min(max((time.time() - self._t0) / total, 0.0), 1.0)
                return prog / warm if prog < warm else max(0.0, (1.0 - prog) / (1.0 - warm))
        else:
            fn = linear_warmup_decay(self.max_steps_budget, self.warmup_frac)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, fn)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    MAX_CONSECUTIVE_BAD_STEPS = 50

    def on_before_optimizer_step(self, optimizer):
        """Skip a step with a non-finite gradient (zero grads and lr; the scheduler restores lr)."""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        gnorm = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])))
        if math.isfinite(gnorm):
            self._bad_steps = 0
            self.log("train/grad_norm", gnorm, batch_size=1)
            return
        self._bad_steps += 1
        self._bad_steps_total += 1
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        rank_zero_warn(f"non-finite gradient at step {self.global_step}: skipped "
                       f"({self._bad_steps} consecutive, {self._bad_steps_total} total)")
        if self._bad_steps >= self.MAX_CONSECUTIVE_BAD_STEPS:
            raise RuntimeError(f"{self._bad_steps} consecutive non-finite gradients at step "
                               f"{self.global_step}; training diverged")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # the wall-clock LR schedule continues where it stopped on resume
        checkpoint["elapsed_s"] = time.time() - self._t0

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if checkpoint.get("elapsed_s") is not None:
            self._t0 = time.time() - float(checkpoint["elapsed_s"])


class DivergenceGuard(pl.Callback):
    """Abort a run that stopped learning.

    * liveness: for ``liveness_patience`` validations in a row, categorical loss within ``margin``
      of the uniform reference and categorical accuracy near chance (a constant predictor);
    * stagnation: ``monitor`` not improved for ``patience`` validations.

    Both checks start after ``warmup_evals`` validations.
    """

    def __init__(self, *, margin: float = 0.05, liveness_patience: int = 3, patience: int = 100,
                 warmup_evals: int = 3, monitor: str = "val/skill", acc_factor: float = 2.0):
        self.margin, self.liveness_patience, self.patience = margin, liveness_patience, patience
        self.warmup_evals, self.monitor, self.acc_factor = warmup_evals, monitor, acc_factor
        self.n_evals = self.dead_evals = self.since_improved = 0
        self.best: float | None = None

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        self.n_evals += 1
        cur = m.get(self.monitor)
        if cur is not None and math.isfinite(cur) and abs(cur) <= SKILL_SANITY:
            if self.best is None or cur < self.best:
                self.best, self.since_improved = cur, 0
            else:
                self.since_improved += 1
        loss_cat, uniform = m.get("val/loss_cat"), m.get("val/loss_cat_uniform")
        if loss_cat is not None and uniform is not None and uniform > 0:
            cat_acc = m.get("val/cat_acc")
            near_chance = cat_acc is None or cat_acc <= math.exp(-uniform) * self.acc_factor
            dead = uniform - loss_cat < self.margin and near_chance
            self.dead_evals = self.dead_evals + 1 if dead else 0
        if self.n_evals <= self.warmup_evals:
            return
        if self.dead_evals >= self.liveness_patience:
            raise RuntimeError(f"categorical predictions at chance for {self.dead_evals} validations "
                               f"(step {trainer.global_step}); the model collapsed to a constant")
        if self.since_improved >= self.patience:
            raise RuntimeError(f"{self.monitor} has not improved on {self.best:.4f} for "
                               f"{self.since_improved} validations (step {trainer.global_step})")


class PretrainCheckpoint(pl.Callback):
    """Writes, under ``root``::

        config.json   run arguments and the model config
        trunk.pt      latest weights (``model.*`` and ``head.*``), loadable on any schema
        best_trunk.pt weights at the lowest ``monitor`` (its value in best.json)
        resume.ckpt   full Lightning checkpoint for ``--resume``
    """

    RESUME_NAME = "resume.ckpt"

    def __init__(self, root, config: dict, *, every_n_steps: int = 1000, monitor: str = "val/skill",
                 resume: bool = False):
        self.root = Path(root)
        self.every_n_steps = every_n_steps
        self.monitor = monitor
        self.best: float | None = None
        best_file = self.root / "best.json"
        if resume and best_file.exists():
            self.best = json.loads(best_file.read_text())["best"]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(config, indent=2, default=str))

    @staticmethod
    def _atomic_save(obj, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def _write_trunk(self, trainer, pl_module, name: str) -> None:
        if trainer.is_global_zero:
            self._atomic_save(pl_module.state_dict(), self.root / name)

    def _write_resume(self, trainer) -> None:
        # collective: every rank must call save_checkpoint
        tmp = self.root / (self.RESUME_NAME + ".tmp")
        trainer.save_checkpoint(tmp)
        trainer.strategy.barrier("resume-ckpt")
        if trainer.is_global_zero and tmp.exists():
            tmp.replace(self.root / self.RESUME_NAME)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.every_n_steps and trainer.global_step and trainer.global_step % self.every_n_steps == 0:
            self._write_trunk(trainer, pl_module, "trunk.pt")
            self._write_resume(trainer)

    def on_validation_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get(self.monitor)
        if v is not None and math.isfinite(float(v)) and (self.best is None or float(v) < self.best):
            self.best = float(v)
            self._write_trunk(trainer, pl_module, "best_trunk.pt")
            if trainer.is_global_zero:
                (self.root / "best.json").write_text(json.dumps({"best": self.best,
                                                                 "step": int(trainer.global_step)}))

    def on_train_end(self, trainer, pl_module):
        self._write_trunk(trainer, pl_module, "trunk.pt")
        self._write_resume(trainer)


def load_trunk(model, path, head=None) -> None:
    """Load pretrained ``model.*`` weights (strictly) and, if given, ``head.*`` weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)

    def sub(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    model.load_state_dict(sub("model."), strict=True)
    if head is not None:
        missing = head.load_state_dict(sub("head."), strict=False).missing_keys
        if missing:
            raise RuntimeError(f"pretrained head is missing {missing}")
