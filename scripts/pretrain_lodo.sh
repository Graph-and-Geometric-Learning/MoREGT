#!/bin/bash
# Leave-one-database-out pretraining: one trunk per database, pretrained on the other six.
#   bash scripts/pretrain_lodo.sh [--devices 3] [other scripts/pretrain.py flags]
# The reported trunks used 3 GPUs and the default 23 h budget per fold; on a cluster, submit one
# fold per job instead of running the loop.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

for db in rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial; do
  python scripts/pretrain.py --holdout "$db" --resume "$@"
done
