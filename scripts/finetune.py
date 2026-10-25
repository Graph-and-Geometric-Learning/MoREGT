"""Fine-tune MoREGT on one RelBench entity task, with the entity's past labels in context, and score
it on TEST.

    python scripts/finetune.py --dataset rel-f1 --task driver-dnf \
        --init-from "$MOREGT_CKPT_DIR/lodo-f1/best_trunk.pt" --seed 0

``--init-from`` loads a pretrained trunk and takes the architecture from its ``config.json``; use the
leave-one-database-out trunk that never saw ``--dataset``. Without it the model trains from scratch.
``--agent-columns agent_columns`` adds the agent-derived columns of
``agent_columns/<dataset>.<task>.json`` (MoREGT with agent; see ``scripts/derive_columns.py``).

Writes ``<out-dir>/<dataset>__<task>__s<seed>.json`` with the validation and TEST metrics of two
models (the best epoch on RelBench val, and the weight average of the best epochs), plus their TEST
predictions as ``.npy``.
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--init-from", default=None, help="pretrained trunk (best_trunk.pt)")
    ap.add_argument("--no-task-context", dest="task_context", action="store_false",
                    help="train without the past-label (task_history) table")
    ap.add_argument("--agent-columns", default=None, metavar="DIR",
                    help="add the agent-derived columns of DIR/<dataset>.<task>.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/finetune")
    ap.add_argument("--save-model", action="store_true", help="also save the selected weights")
    data = ap.add_argument_group("data")
    data.add_argument("--text-encoder", default="minilm")
    data.add_argument("--name-encoder", default="qwen")
    data.add_argument("--seq-len", type=int, default=1024)
    data.add_argument("--num-neighbors", type=int, nargs="+", default=[12, 12])
    data.add_argument("--num-workers", type=int, default=8)
    opt = ap.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=12)
    opt.add_argument("--patience", type=int, default=4)
    opt.add_argument("--batch-size", type=int, default=128)
    opt.add_argument("--limit-train-batches", type=int, default=1000, help="train batches per epoch")
    opt.add_argument("--lr", type=float, default=3e-4)
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    opt.add_argument("--lambda-ortho", type=float, default=0.5)
    reg = ap.add_argument_group("regression tasks")
    reg.add_argument("--reg-loss", default="mse", choices=["mse", "l1"])
    reg.add_argument("--target-scaling", default="log1p", choices=["log1p", "zscore", "raw"])
    reg.add_argument("--clamp-pct", type=float, default=None,
                     help="clip predictions to the train target's [p, 100 - p] percentiles")
    sel = ap.add_argument_group("model selection")
    sel.add_argument("--select-split", default="val", choices=["val", "holdout"],
                     help="split for early stopping and weight averaging; 'holdout' holds out the "
                          "last --holdout-windows train timestamps")
    sel.add_argument("--holdout-windows", type=int, default=2)
    sel.add_argument("--swa-window", type=int, default=5, help="best epochs averaged")
    ap.add_argument("--cell-attn-backend", default="flex", choices=["flex", "sdpa"])
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    warnings.filterwarnings("ignore")

    import numpy as np
    import torch
    from relbench.tasks import get_task

    from moregt.data.graph import build_graph
    from moregt.data.task_context import task_context_cache_dir
    from moregt.eval.evaluate import evaluate_split
    from moregt.model.moregt import MoREGTConfig
    from moregt.text.encoders import name_encoder
    from moregt.text.schema import embed_schema
    from moregt.train.finetune import finetune, target_stats, task_kind
    from moregt.utils import graph_cache_dir, limit_torch_threads

    limit_torch_threads()
    ds, tk = args.dataset, args.task
    task = get_task(ds, tk, download=False)
    kind = task_kind(task)

    cache = graph_cache_dir(ds, args.text_encoder)
    if not cache.exists():
        raise SystemExit(f"no graph cache at {cache}; run scripts/prepare_data.py --datasets {ds}")
    if args.task_context:
        bundle = build_graph(ds, cache_dir=str(task_context_cache_dir(cache, tk)),
                             text_encoder=args.text_encoder, task=task, task_name=tk)
    else:
        bundle = build_graph(ds, cache_dir=str(cache), text_encoder=args.text_encoder)
    n_agent_columns = 0
    if args.agent_columns:
        from moregt.agent.columns import apply_agent_columns

        bundle = apply_agent_columns(bundle, args.agent_columns, tk)
        n_agent_columns = sum(len(v) for v in bundle.derived_columns.values())
        print(f"agent columns: +{n_agent_columns} ({bundle.derived_columns})", flush=True)
    schema = embed_schema(bundle, name_encoder(ds, args.name_encoder))

    config = MoREGTConfig()
    if args.init_from:
        config = MoREGTConfig.from_checkpoint(args.init_from)
        trunk_args = json.loads((Path(args.init_from).parent / "config.json").read_text()).get("args", {})
        for key in ("text_encoder", "name_encoder"):
            if trunk_args.get(key, getattr(args, key)) != getattr(args, key):
                raise SystemExit(f"the trunk was pretrained with --{key.replace('_', '-')} "
                                 f"{trunk_args[key]}, not {getattr(args, key)}")
    config.cell_attn_backend = args.cell_attn_backend

    module, best_state, swa_state, info = finetune(
        bundle, task, schema, config=config, init_from=args.init_from, seed=args.seed,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        limit_train_batches=args.limit_train_batches, num_workers=args.num_workers,
        num_neighbors=args.num_neighbors, seq_len=args.seq_len, lr=args.lr,
        weight_decay=args.weight_decay, optimizer=args.optimizer, regression_loss=args.reg_loss,
        target_scaling=args.target_scaling, clamp_pct=args.clamp_pct,
        select_split=args.select_split, holdout_windows=args.holdout_windows,
        swa_window=args.swa_window, lambda_ortho=args.lambda_ortho)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ds}__{tk}__s{args.seed}"
    record = {"dataset": ds, "task": tk, "task_type": kind, "seed": args.seed,
              "n_agent_columns": n_agent_columns, "args": vars(args), "model": config.to_dict(),
              **info}
    train_std = target_stats(task)[1] if kind == "regression" else None
    eval_kw = dict(seq_len=args.seq_len, num_neighbors=args.num_neighbors,
                   batch_size=args.batch_size, num_workers=args.num_workers)
    for name, state in (("test", best_state), ("test_swa", swa_state)):
        module.load_state_dict(state)
        metrics, pred = evaluate_split(module, bundle, task, "test", **eval_kw)
        if train_std is not None:
            metrics["nmae"] = metrics["mae"] / train_std
        record[name] = metrics
        np.save(out_dir / f"{stem}_{name}_pred.npy", pred)
    if args.save_model:
        torch.save(best_state, out_dir / f"{stem}.pt")
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"dataset": ds, "task": tk, "seed": args.seed, "val_best": info["val_best"],
                      "test": record["test"], "test_swa": record["test_swa"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
