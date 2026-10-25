# Cache and checkpoint locations. Source before running the scripts:
#   source scripts/env.sh
# Every path defaults to ./data and can be overridden by setting it beforehand.
export MOREGT_DATA="${MOREGT_DATA:-$PWD/data}"
export RELBENCH_CACHE_DIR="${RELBENCH_CACHE_DIR:-$MOREGT_DATA/relbench}"
export HF_HOME="${HF_HOME:-$MOREGT_DATA/hf}"
export MOREGT_GRAPH_CACHE="${MOREGT_GRAPH_CACHE:-$MOREGT_DATA/graph_cache}"
export MOREGT_SCHEMA_CACHE="${MOREGT_SCHEMA_CACHE:-$MOREGT_DATA/schema_cache}"
export MOREGT_CKPT_DIR="${MOREGT_CKPT_DIR:-$MOREGT_DATA/checkpoints}"
export MOREGT_AGENT_CACHE="${MOREGT_AGENT_CACHE:-$MOREGT_DATA/agent_cache}"
