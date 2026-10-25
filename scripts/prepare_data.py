"""Download RelBench databases, materialise their graphs and cache the schema-name embeddings.

    python scripts/prepare_data.py --datasets rel-f1 rel-trial --download
    python scripts/prepare_data.py --datasets all --task-context      # also the past-label graphs

For each database: the task tables of its benchmark tasks are materialised, the graph is built with
MiniLM embeddings of the free-text cell values (``$MOREGT_GRAPH_CACHE/<text-encoder>/<db>``), and every
column, table, FK-role and category name is embedded with Qwen3-Embedding-4B
(``$MOREGT_SCHEMA_CACHE/<db>``). With ``--task-context``, each task's past-label graph is built under
``<graph cache>/taskctx/<task>`` and its new names are embedded too, so training never loads an
encoder. ``--agent-columns DIR`` also embeds the names of the agent-derived columns in ``DIR``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from moregt.benchmark import DATABASES, tasks_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["all"], help="database names, or 'all'")
    ap.add_argument("--download", action="store_true", help="download databases and task tables")
    ap.add_argument("--text-encoder", default="minilm", help="encoder of free-text cell values")
    ap.add_argument("--name-encoder", default="qwen", help="encoder of schema names")
    ap.add_argument("--text-batch-size", type=int, default=8192)
    ap.add_argument("--task-context", action="store_true",
                    help="also build each benchmark task's past-label (task_history) graph")
    ap.add_argument("--agent-columns", default=None, metavar="DIR",
                    help="also embed the names of the agent-derived columns in DIR")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from moregt.data.graph import build_graph
    from moregt.data.task_context import task_context_cache_dir
    from moregt.text.encoders import ENCODER_MODELS, EmbeddingCache, SentenceEncoder
    from moregt.text.schema import embed_schema
    from moregt.utils import graph_cache_dir, schema_cache_path

    datasets = list(DATABASES) if args.datasets == ["all"] else args.datasets
    names = SentenceEncoder(ENCODER_MODELS.get(args.name_encoder, args.name_encoder))
    for ds in datasets:
        print(f"[{ds}] loading database", flush=True)
        db = get_dataset(ds, download=args.download).get_db(upto_test_timestamp=False)
        tasks = {}
        for name in tasks_of(ds):
            tasks[name] = get_task(ds, name, download=args.download)
            for split in ("train", "val", "test"):
                tasks[name].get_table(split)
        cache = graph_cache_dir(ds, args.text_encoder)
        bundle = build_graph(ds, cache_dir=str(cache), text_encoder=args.text_encoder,
                             text_batch_size=args.text_batch_size)
        encode = EmbeddingCache(names, schema_cache_path(ds, args.name_encoder))
        schema = embed_schema(bundle, encode)
        print(f"[{ds}] graph -> {cache}; names: {tuple(schema.column.shape)} columns, "
              f"{tuple(schema.table.shape)} tables, {tuple(schema.role.shape)} roles", flush=True)
        if args.task_context:
            for name, task in tasks.items():
                tc = build_graph(ds, cache_dir=str(task_context_cache_dir(cache, name)),
                                 text_encoder=args.text_encoder, task=task, task_name=name)
                embed_schema(tc, encode)
                print(f"[{ds}/{name}] past-label graph: {tc.data['task_history'].num_nodes} rows",
                      flush=True)
        if args.agent_columns:
            from moregt.agent.columns import agent_columns_path, apply_agent_columns

            for name in tasks:
                if agent_columns_path(args.agent_columns, ds, name).exists():
                    embed_schema(apply_agent_columns(bundle, args.agent_columns, name, db=db), encode)
                    print(f"[{ds}/{name}] agent-column names embedded", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
