#!/bin/bash
# Fine-tune every benchmark task from the LODO trunk that held its database out, over 3 seeds.
#   bash scripts/finetune_benchmark.sh [rel-f1 ...] [-- extra scripts/finetune.py flags]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

DBS=()
while [[ $# -gt 0 && $1 != "--" ]]; do DBS+=("$1"); shift; done
[[ $# -gt 0 ]] && shift
[[ ${#DBS[@]} -eq 0 ]] && DBS=(rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial)

for db in "${DBS[@]}"; do
  trunk="$MOREGT_CKPT_DIR/lodo-${db#rel-}/best_trunk.pt"
  batch=128
  [[ $db == rel-event ]] && batch=64            # memory
  for task in $(python -c "from moregt.benchmark import tasks_of; print(' '.join(tasks_of('$db')))"); do
    for seed in 0 1 2; do
      python scripts/finetune.py --dataset "$db" --task "$task" --init-from "$trunk" \
        --seed "$seed" --batch-size "$batch" "$@"
    done
  done
done
