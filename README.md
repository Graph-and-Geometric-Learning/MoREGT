# MoREGT

MoREGT is a relational graph transformer for predictive tasks on relational databases. It
tokenises every cell of an entity's temporal relational neighbourhood, processes the tokens with a
two-level (cell / row) transformer whose feed-forward layers are mixtures of experts routed on
value-free relational signatures, is pretrained by masked-cell reconstruction on several databases,
and is fine-tuned on a task of an unseen database with the entity's past labels in context.
Optionally (MoREGT with agent), an LLM agent first adds row-local derived columns to the database.

## Method

**Cell tokens.** Every cell `(row v, column c)` of every sampled row becomes the token
`x = W_v Enc_stype(value) + W_name n_c`, where `n_c` is a frozen Qwen3-Embedding-4B embedding of
`"table T, column C"`. Values have one shared encoder per datatype (numerical, categorical via
frozen label embeddings, free text via frozen MiniLM vectors, timestamp), so no parameter is sized
by any database's schema and one set of weights loads on every database.

**Relational signatures.** The cell signature `z = RMSNorm(W_s n_c + E[stype(c)] + W_tau phi(t* - t_v))`
and the row signature (table name, FK role by which the row was reached, hop distance, age) depend
only on what a cell or row *is*, never on its value or on which neighbours were sampled. Time enters
through a fixed log-time frequency ladder.

**Two-level blocks** (x10), each with six sublayers: cell attention over all cells of the seed with
temporal rotary encoding; a cell MoE FFN (8 routed + 2 shared SwiGLU experts, top-2, router on `z`);
RowPool (cells to row tokens, keyed by column-name embeddings); row attention over the sampled FK
graph with a bias computed from the FK role's name; a row MoE FFN (4 routed + 1 shared experts,
cosine router on the row signature); and Broadcast (row tokens back to their cells). Routers are
regularised by an orthogonality penalty. The task head reads the seed row's token. The default model
has 133M parameters.

**Pretraining.** Seeds are rows of any table with a maskable column. One cell of the seed row (and
half of its other maskable cells) is masked and reconstructed from the relational context: Huber
loss for numerical cells, cross-entropy against the column's frozen label embeddings for categorical
cells. Seeds never pass RelBench's validation timestamp. Leave-one-database-out (LODO): for each of
the 7 RelBench databases, a trunk is pretrained on the other six.

**Fine-tuning with past labels.** The task's train and val rows are appended to the database as an
ordinary table, `task_history`, with each label timestamped at the moment its prediction window
closes. The temporal sampler only admits rows dated before the seed time, so a seed sees exactly the
labels known at its own timestamp and never its own label. The model is initialised from the LODO
trunk that never saw the target database and trained end to end.

**Agent-derived columns (MoREGT with agent).** Before fine-tuning, an LLM agent (Claude Code, at
most 200 turns per round) proposes new columns for the task's database. It is given only the dataset
name, the task name and its entity table, and it may run one command, `scripts/agent_tool.py`, whose
four read-only tools report tables, column profiles and key candidates inferred from the data. It
never sees labels, the task table, or the declared primary keys, foreign keys and time columns. A
proposal is an expression over the columns of one row of one table (arithmetic, comparisons, `abs`,
`log`, `log1p`, `sqrt`, `minimum`, `maximum`), with units and an expected range stated in advance.
Nothing that could read another row parses, so a derived value is visible to the temporal sampler
exactly when its row is. Proposals are screened without a model or labels: they are rejected if they
are constant, create nulls their inputs do not have, put fewer than 90% of their values in the
stated range, or duplicate (|corr| > 0.999) an existing or earlier accepted column. Rejections and
their reasons go back to the agent for at most 3 rounds. Accepted columns are appended to their
tables as ordinary numerical cells; the model is unchanged.

## Layout

```
moregt/
  data/    graph.py (RelBench -> graph, temporal loader), task_context.py (past labels),
           collate.py + row_graph.py (cell-token batch), masking.py, pretrain_loader.py
  text/    encoders.py (frozen encoders, embedding cache), schema.py (name embeddings)
  model/   moregt.py (MoREGT, MoREGTConfig), encoder.py, signature.py, time_encoding.py,
           two_level.py, row_level.py, moe.py, flex_cell.py, heads.py, mlm_head.py, schema_bank.py
  train/   pretrain.py, finetune.py, losses.py
  eval/    evaluate.py, metrics.py
  agent/   blind.py + tools.py + keyinfer.py (the agent's tools), grammar.py (expressions, screening),
           runner.py (Claude Code subprocess, audit), derive_loop.py, columns.py (derived cells)
  benchmark.py   the 21 benchmark tasks
agent_columns/   the accepted agent-derived columns of the 21 tasks
scripts/   prepare_data.py, pretrain.py, finetune.py, derive_columns.py, agent_tool.py,
           pretrain_lodo.sh, finetune_benchmark.sh, env.sh
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric          # matching your CUDA version
pip install -r requirements.txt
pip install -e .
source scripts/env.sh                      # cache locations, default ./data
```

## Data

```bash
python scripts/prepare_data.py --datasets all --download --task-context
```

This downloads the RelBench databases and task tables, materialises each graph with MiniLM
embeddings of the free-text cells, and caches the Qwen embeddings of all schema names, including
the past-label graphs of the 21 benchmark tasks. It needs a GPU for the two text encoders; the
largest database (rel-amazon) needs about 500 GB of memory.

## Pretraining

```bash
python scripts/pretrain.py --holdout rel-f1 --devices 3    # one LODO fold
bash scripts/pretrain_lodo.sh --devices 3                  # all seven folds
```

The defaults are the reported recipe: batch 64 per GPU, AdamW (lr 3e-4, betas 0.9/0.95, weight
decay 0.1), 5% warmup and linear decay over a 23 h wall-clock budget, bf16. Checkpoints go to
`$MOREGT_CKPT_DIR/lodo-<db>/`; `best_trunk.pt` (lowest validation skill) is the fine-tuning
initialisation. `--resume` continues an interrupted run.

## Fine-tuning

```bash
python scripts/finetune.py --dataset rel-f1 --task driver-dnf \
    --init-from "$MOREGT_CKPT_DIR/lodo-f1/best_trunk.pt" --seed 0
bash scripts/finetune_benchmark.sh                          # 21 tasks x 3 seeds
```

Defaults: AdamW (lr 3e-4, weight decay 0.01), batch 128 (64 on rel-event), 12 epochs of 1000
batches, early stopping after 4 epochs without improvement; regression targets are trained with MSE
on z-scored `log(1 + y)`. Each run reports TEST metrics for the epoch with the best RelBench
validation metric (`test`), and for the weight average of the 5 best epochs (`test_swa`). Binary
tasks are scored by AUROC, regression tasks by MAE normalised by the train target standard deviation
(`nmae`). `--no-task-context` trains without past labels; `--help` lists the remaining options
(learning rate, weight decay, regression loss and scaling, holdout selection).

## Agent-derived columns

`agent_columns/<db>.<task>.json` holds the columns the agent produced for each benchmark task (8 to
14 per task, 210 in total), together with the rejected proposals and their reasons and a per-round
audit of the agent's tool calls. To train MoREGT with agent:

```bash
python scripts/prepare_data.py --datasets all --task-context --agent-columns agent_columns
python scripts/finetune.py --dataset rel-f1 --task driver-dnf \
    --init-from "$MOREGT_CKPT_DIR/lodo-f1/best_trunk.pt" --agent-columns agent_columns --seed 0
bash scripts/finetune_benchmark.sh -- --agent-columns agent_columns
```

To generate new columns, install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and
run

```bash
python scripts/derive_columns.py --tasks rel-f1/driver-dnf --out-dir my_columns   # or --tasks all
```

which writes `my_columns/<db>.<task>.json` and the raw transcripts. The agent runs from the
repository root and calls `.venv/bin/python scripts/agent_tool.py` (set `$MOREGT_AGENT_PYTHON` for
another interpreter). A run is discarded if the agent reports an error or any of its tool calls
reaches the raw data files. LLM output is not deterministic, so new columns differ from the released
ones. On rel-trial, the duplicate check now ignores correlations over fewer than 30 rows, which the
released runs did not, so it accepts more of the same proposals there.

## Tests

```bash
pytest tests
```

The tests use synthetic databases. Integration tests on rel-f1 run when it is in the RelBench cache,
and the flex-attention test runs when a GPU is available.
