"""Run the agent's column-derivation loop for benchmark tasks.

    python scripts/derive_columns.py --tasks rel-f1/driver-dnf rel-trial/study-outcome
    python scripts/derive_columns.py --tasks all

Needs Claude Code (``claude``) on PATH and the databases in the RelBench cache. For each task, writes
``<out-dir>/<dataset>.<task>.json`` (the accepted columns, the rejected ones with their reasons, and a
per-round audit) and the raw transcripts under ``<out-dir>/transcripts/``. ``scripts/finetune.py
--agent-columns <out-dir>`` reads these files. A task whose file exists is skipped unless ``--force``.

A run is not recorded if the agent reported an error or any of its tool calls reached the raw data
files; re-run it instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, help="DATASET/TASK pairs, or 'all'")
    ap.add_argument("--out-dir", default="agent_columns")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per round")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="overwrite an existing record")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from moregt.agent import derive_loop
    from moregt.agent.columns import agent_columns_path
    from moregt.agent.runner import ClaudeCodeTransport
    from moregt.benchmark import TASKS

    pairs = list(TASKS) if args.tasks == ["all"] else [tuple(t.split("/", 1)) for t in args.tasks]
    out_dir = Path(args.out_dir)
    transport = ClaudeCodeTransport(max_turns=args.max_turns, timeout=args.timeout, model=args.model)
    total = 0.0
    for dataset, task in pairs:
        path = agent_columns_path(out_dir, dataset, task)
        if path.exists() and not args.force:
            print(f"skip (exists): {path}", flush=True)
            continue
        t0 = time.time()
        record = derive_loop.run(dataset, task, transport=transport, max_rounds=args.max_rounds,
                                 transcript_dir=out_dir / "transcripts")
        if any(r["is_error"] for r in record["rounds"]):
            print(f"{dataset}/{task}: the agent reported an error; not recorded", flush=True)
            continue
        dirty = [r["round"] for r in record["rounds"] if not r["audit"]["usable"]]
        if dirty:
            print(f"{dataset}/{task}: round(s) {dirty} reached the raw data; not recorded", flush=True)
            continue
        record["model"] = record["model"] or args.model
        record["seconds"] = round(time.time() - t0, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        total += record["cost_usd"]
        print(f"{dataset}/{task}: {record['n_accepted']} accepted in {record['n_rounds']} round(s), "
              f"${record['cost_usd']:.2f} -> {path}", flush=True)
        for c in record["accepted"]:
            print(f"    {c['table']}.{c['name']} = {c['expression']}", flush=True)
    print(f"total cost ${total:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
