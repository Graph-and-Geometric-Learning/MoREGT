"""Masked-cell pretraining of MoREGT on one or several databases.

    python scripts/pretrain.py --holdout rel-f1 --devices 3            # LODO fold: the other six DBs
    python scripts/pretrain.py --datasets rel-f1 --max-steps 2000      # a single database

The defaults are the reported recipe. Checkpoints go to ``$MOREGT_CKPT_DIR/<run>/``
(``best_trunk.pt`` is the one fine-tuning loads).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = ap.add_argument_group("data")
    data.add_argument("--datasets", nargs="+", default=None, help="databases to pretrain on, or 'all'")
    data.add_argument("--holdout", default=None, help="pretrain on every database except this one")
    data.add_argument("--text-encoder", default="minilm")
    data.add_argument("--name-encoder", default="qwen")
    data.add_argument("--seq-len", type=int, default=1024)
    data.add_argument("--num-neighbors", type=int, nargs="+", default=[12, 12])
    data.add_argument("--max-seeds-per-table", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=2)
    model = ap.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=256)
    model.add_argument("--n-blocks", type=int, default=10)
    model.add_argument("--n-heads", type=int, default=8)
    model.add_argument("--d-ff", type=int, default=1024)
    model.add_argument("--d-sig", type=int, default=128)
    model.add_argument("--num-experts", type=int, default=8)
    model.add_argument("--num-shared", type=int, default=2)
    model.add_argument("--top-k", type=int, default=2)
    model.add_argument("--row-num-experts", type=int, default=4)
    model.add_argument("--row-num-shared", type=int, default=1)
    model.add_argument("--cell-attn-backend", default="flex", choices=["flex", "sdpa"])
    model.add_argument("--grad-checkpoint", action="store_true")
    obj = ap.add_argument_group("objective")
    obj.add_argument("--seed-frac", type=float, default=0.5,
                     help="fraction of the seed row's other maskable cells also masked")
    obj.add_argument("--p-random", type=float, default=0.0,
                     help="fraction of the other maskable cells of the sequence masked")
    obj.add_argument("--lambda-cat", type=float, default=4.0)
    obj.add_argument("--lambda-ortho", type=float, default=0.5)
    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--max-steps", type=int, default=300_000)
    opt.add_argument("--max-time", default="23:00:00",
                     help="wall-clock budget (HH:MM:SS or D-HH:MM:SS) that drives the LR schedule; "
                          "'none' schedules over --max-steps")
    opt.add_argument("--batch-size", type=int, default=64, help="per device")
    opt.add_argument("--lr", type=float, default=3e-4)
    opt.add_argument("--weight-decay", type=float, default=0.1)
    opt.add_argument("--warmup-frac", type=float, default=0.05)
    opt.add_argument("--precision", default="bf16-mixed")
    run = ap.add_argument_group("run")
    run.add_argument("--run-name", default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--devices", type=int, default=1)
    run.add_argument("--val-every", type=int, default=1000)
    run.add_argument("--val-steps", type=int, default=50)
    run.add_argument("--ckpt-every", type=int, default=1000)
    run.add_argument("--guard-patience", type=int, default=100,
                     help="validations without a val/skill improvement before aborting")
    run.add_argument("--no-guard", action="store_true")
    run.add_argument("--resume", action="store_true", help="continue from resume.ckpt if present")
    run.add_argument("--init-from", default=None,
                     help="initialise from a pretrained trunk (fresh optimiser), e.g. to continue "
                          "pretraining on another database")
    run.add_argument("--wandb", action="store_true")
    run.add_argument("--wandb-project", default="moregt-pretrain")
    return ap


def parse_seconds(spec: str | None) -> float | None:
    """``HH:MM:SS``, ``D-HH:MM:SS`` or ``DD:HH:MM:SS`` -> seconds."""
    if not spec or spec.lower() == "none":
        return None
    days, _, rest = spec.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    if len(parts) == 4:
        days, parts = str(int(days or 0) + parts[0]), parts[1:]
    h, m, s = ([0] * (3 - len(parts)) + parts)[-3:]
    return float(((int(days or 0) * 24 + h) * 60 + m) * 60 + s)


def lightning_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    s = int(seconds)
    return f"{s // 86400:02d}:{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main() -> int:
    args = build_argparser().parse_args()
    warnings.filterwarnings("ignore")

    import pytorch_lightning as pl

    from moregt.benchmark import DATABASES
    from moregt.data.graph import build_graph
    from moregt.data.pretrain_loader import build_pretrain_stream, build_worker_loader
    from moregt.model.moregt import MoREGTConfig
    from moregt.text.encoders import name_encoder
    from moregt.text.schema import embed_schema
    from moregt.train.pretrain import (DivergenceGuard, PretrainCheckpoint, PretrainModule,
                                       load_trunk)
    from moregt.utils import ckpt_dir, graph_cache_dir, seed_everything

    if args.holdout:
        datasets = [d for d in DATABASES if d != args.holdout]
    elif args.datasets == ["all"]:
        datasets = list(DATABASES)
    elif args.datasets:
        datasets = list(args.datasets)
    else:
        raise SystemExit("pass --datasets or --holdout")
    run = args.run_name or (f"lodo-{args.holdout.removeprefix('rel-')}" if args.holdout
                            else "pretrain-" + "-".join(d.removeprefix("rel-") for d in datasets))
    seed_everything(args.seed)

    bundles, schemas = {}, {}
    for ds in datasets:
        cache = graph_cache_dir(ds, args.text_encoder)
        if not cache.exists():
            raise SystemExit(f"no graph cache at {cache}; run scripts/prepare_data.py --datasets {ds}")
        bundles[ds] = build_graph(ds, cache_dir=str(cache), text_encoder=args.text_encoder)
        schemas[ds] = embed_schema(bundles[ds], name_encoder(ds, args.name_encoder))

    config = MoREGTConfig(
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads, d_ff=args.d_ff,
        d_sig=args.d_sig, num_experts=args.num_experts, num_shared=args.num_shared, top_k=args.top_k,
        row_num_experts=args.row_num_experts, row_num_shared=args.row_num_shared,
        cell_attn_backend=args.cell_attn_backend, grad_checkpoint=args.grad_checkpoint)
    max_time_s = parse_seconds(args.max_time)
    module = PretrainModule(bundles, schemas, config, p_random=args.p_random,
                            seed_frac=args.seed_frac, lambda_cat=args.lambda_cat,
                            lambda_ortho=args.lambda_ortho, lr=args.lr,
                            weight_decay=args.weight_decay, max_steps=args.max_steps,
                            max_time_s=max_time_s, warmup_frac=args.warmup_frac,
                            seq_len=args.seq_len, mask_seed=args.seed)
    n_par = sum(p.numel() for p in module.parameters())
    print(f"[{run}] {n_par / 1e6:.1f}M parameters; databases: {datasets}", flush=True)
    for ds in datasets:
        print(f"  {ds}: {module.specs[ds].summary()}", flush=True)

    common = dict(batch_size=args.batch_size, num_neighbors=args.num_neighbors, seed=args.seed,
                  max_seeds_per_table=args.max_seeds_per_table)
    train_rr = build_pretrain_stream(bundles, module.specs, split="train", steps=args.max_steps, **common)
    val_rr = build_pretrain_stream(bundles, module.specs, split="val", steps=args.val_steps, **common)
    train_stream = build_worker_loader(train_rr, collate=module.collate, num_workers=args.num_workers,
                                       seed=args.seed)
    val_stream = build_worker_loader(val_rr, collate=module.collate, num_workers=0, seed=args.seed)

    root = ckpt_dir(run)
    resume = root / PretrainCheckpoint.RESUME_NAME
    resume_path = str(resume) if args.resume and resume.exists() else None
    print(f"[{run}] {'resuming from ' + resume_path if resume_path else 'starting from step 0'}",
          flush=True)
    if args.init_from and not resume_path:
        load_trunk(module.model, args.init_from, head=module.head)
        print(f"[{run}] initialised from {args.init_from}", flush=True)

    cfg = {"run": run, "datasets": datasets, "args": vars(args), "model": config.to_dict(),
           "n_params": n_par}
    ckpt = PretrainCheckpoint(root, cfg, every_n_steps=args.ckpt_every, resume=resume_path is not None)
    callbacks = [ckpt]
    if not args.no_guard:
        callbacks.append(DivergenceGuard(patience=args.guard_patience))

    logger = False
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=run, save_dir=str(root))
        logger.log_hyperparams(cfg["args"] | {"n_params": n_par})

    strategy = "auto"
    if args.devices > 1:
        from pytorch_lightning.plugins.environments import LightningEnvironment
        from pytorch_lightning.strategies import DDPStrategy

        # Sparse expert dispatch leaves experts without tokens on some steps, hence unused
        # parameters. LightningEnvironment lets Lightning spawn the ranks itself under a scheduler.
        strategy = DDPStrategy(find_unused_parameters=True, cluster_environment=LightningEnvironment())

    trainer = pl.Trainer(
        max_steps=args.max_steps, max_time=lightning_duration(max_time_s), accelerator="auto",
        devices=args.devices, strategy=strategy, precision=args.precision, logger=logger,
        enable_checkpointing=False, enable_model_summary=False, log_every_n_steps=10,
        callbacks=callbacks, gradient_clip_val=1.0, val_check_interval=args.val_every,
        limit_val_batches=args.val_steps, num_sanity_val_steps=0,
    )
    trainer.fit(module, train_stream, val_stream, ckpt_path=resume_path)
    print(json.dumps({"run": run, "steps": int(trainer.global_step), "best_val_skill": ckpt.best,
                      "checkpoints": str(root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
