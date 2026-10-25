"""Propose, screen, revise: the agent's column-derivation loop for one task.

The agent gets the dataset name, the task name and its entity table, and the tools of
:mod:`moregt.agent.tools`; it never sees labels, the task table or the declared keys. Each round it
returns a JSON array of proposals, :func:`moregt.agent.grammar.screen` checks them, and the rejected
ones come back with their reasons. The feedback is screening reasons only, never a model score. The
loop stops when the agent says ``DONE``, nothing is rejected, nothing new is proposed, or after
:data:`MAX_ROUNDS` rounds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .grammar import ALLOWED_FUNCS, Candidate, forbidden_columns, screen
from .runner import TOOL_COMMAND, AgentResult, audit, write_transcript
from .tools import load_db, render_catalog

MAX_ROUNDS = 3

SYSTEM = f"""\
You are inventing new quantities from a database's existing columns. You may run exactly one shell
command: `{TOOL_COMMAND} <tool> --args '<json>'`. Every other command is
denied. Each call prints one JSON object.
"""

PROMPT = """\
Invent derived quantities for `{dataset}`, to help predict: {task_description}

Available tools:
{catalog}

A derived quantity is a NEW COLUMN computed from other columns **of the same row**. For example, if a
table has a speed and a duration, their product is a distance -- a quantity the database never stored
but which is what someone actually cares about.

## The rules

Your expression may contain ONLY: column names of that one table, numbers, `+ - * / // % **`,
comparisons, and {funcs}. Nothing else parses. In particular you cannot group, sort, shift, or read
another table -- that restriction exists so a quantity can only ever see the row it is computed on,
and therefore cannot see the future.

## What is worth proposing

A brute-force generator already tries **every pairwise difference and ratio** of numeric columns in
this database. You are not competing on coverage; you will lose. You are competing on **meaning**:

- quantities where the combination denotes something a person would name — a rate, a margin, a
  duration per unit, an efficiency, a gap;
- combinations where knowing what the columns ARE matters. An integer column can be a measurement or
  a label; subtracting two labels is nonsense that the brute-force generator happily produces.

Do not propose arithmetic on identifiers. Do not propose a rescaling of one column — it will be
rejected as a duplicate.

## How to work

1. `get_table_info` for the database, then `profile_table` on the tables that look relevant. The
   profiles give you dtype, cardinality, range and the most frequent values.
2. For each quantity, decide what it MEANS first and only then write the expression.
3. State the units and a range you expect the values to fall in. **You will be scored against your
   own claims**: the harness computes your column and rejects it if it does not behave as you said.
   An honest wide range is better than a confident wrong one, but a range so wide it excludes nothing
   tells us you did not know what you were computing.

Return ONLY a JSON array, no prose around it:

```json
[{{"name": "positions_gained", "table": "results", "expression": "grid - position",
   "description": "Places gained between the starting grid slot and the finish.",
   "units": "grid positions", "expected_range": [-25, 25],
   "why_it_matters": "overtaking ability, distinct from raw finishing position"}}]
```

Propose between 5 and 25 quantities. Fewer good ones beats many weak ones -- every candidate is
tested, and a rejected candidate costs you nothing but tells us you did not know.
"""

REVISION = """\
{n_rejected} of your {n_total} proposals were rejected. The reasons are below. Some are arithmetic
(you divided by something that is zero); some mean the quantity does not compute what you said it
does, which is worth more attention than a syntax error.

{rejections}

{n_kept} were accepted: {kept}

Propose replacements for the rejected ones, or say `DONE` if you have nothing better. Same JSON
format, same rules.
"""


def task_description(task, task_name: str) -> str:
    return f"the RelBench task `{task_name}` — one row per `{task.entity_table}` at a point in time"


def build_prompt(dataset: str, description: str) -> tuple[str, str]:
    return SYSTEM, PROMPT.format(dataset=dataset, task_description=description,
                                 catalog=render_catalog(),
                                 funcs=", ".join(f"`{f}`" for f in sorted(ALLOWED_FUNCS)))


def parse_proposals(text: str) -> list[Candidate]:
    """The first well-formed JSON array of proposals in a reply (fenced or not)."""
    if not text:
        return []
    blocks = (re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
              or re.findall(r"(\[.*\])", text, re.S))
    for block in blocks:
        try:
            raw = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, list):
            continue
        out = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("expression") or not item.get("table"):
                continue
            rng = item.get("expected_range")
            out.append(Candidate(
                name=str(item.get("name") or "").strip() or "unnamed",
                table=str(item["table"]).strip(),
                expression=str(item["expression"]).strip(),
                description=str(item.get("description") or ""),
                units=str(item.get("units") or ""),
                expected_range=(float(rng[0]), float(rng[1]))
                if isinstance(rng, (list, tuple)) and len(rng) == 2 else None,
                why_it_matters=str(item.get("why_it_matters") or "")))
        if out:
            return out
    return []


def _rejection_text(rejected: list[Candidate], limit: int = 25) -> str:
    lines = [f"- `{c.name}` = `{c.expression}` on {c.table}: {'; '.join(c.problems) or 'rejected'}"
             for c in rejected[:limit]]
    if len(rejected) > limit:
        lines.append(f"- ... and {len(rejected) - limit} more")
    return "\n".join(lines)


def run(dataset: str, task_name: str, *, transport, db=None, task=None,
        description: str | None = None, forbidden: dict[str, set[str]] | None = None,
        transcript_dir: Path | None = None, max_rounds: int = MAX_ROUNDS) -> dict:
    """Run the loop and return its record: the accepted columns and a per-round audit.

    ``db``, ``description`` and ``forbidden`` default to those of the RelBench task.
    """
    database = db if db is not None else load_db(dataset)
    if description is None or forbidden is None:
        if task is None:
            from relbench.tasks import get_task

            task = get_task(dataset, task_name, download=False)
        description = description or task_description(task, task_name)
        forbidden = forbidden if forbidden is not None else forbidden_columns(task, database)

    system, prompt = build_prompt(dataset, description)
    accepted: list[Candidate] = []
    rounds: list[dict] = []
    seen: set[tuple[str, str]] = set()
    model = None
    for rnd in range(max_rounds):
        result: AgentResult = transport(prompt, system=system)
        model = model or result.model
        if transcript_dir:
            write_transcript(result, Path(transcript_dir) / f"{dataset}.{task_name}.r{rnd}.jsonl")

        proposals = [c for c in parse_proposals(result.text) if (c.table, c.expression) not in seen]
        seen.update((c.table, c.expression) for c in proposals)
        kept, rejected = screen(database, proposals, forbidden=forbidden)
        accepted.extend(kept)
        a = audit(result)
        rounds.append({"round": rnd, "n_proposed": len(proposals), "n_accepted": len(kept),
                       "n_rejected": len(rejected), "is_error": result.is_error,
                       "cost_usd": result.cost_usd,
                       "audit": {k: a[k] for k in ("n_tool_calls", "n_off_allowlist", "n_data_access",
                                                   "usable")},
                       "rejected": [c.as_dict() for c in rejected]})

        done = "DONE" in (result.text or "")[-200:].upper()
        if done or not proposals or not rejected or rnd == max_rounds - 1:
            break
        prompt = REVISION.format(n_rejected=len(rejected), n_total=len(proposals), n_kept=len(kept),
                                 rejections=_rejection_text(rejected),
                                 kept=", ".join(f"`{c.name}`" for c in kept) or "none")

    errored = any(r["is_error"] for r in rounds)
    return {"dataset": dataset, "task": task_name, "model": model,
            "complete": not errored and bool(accepted),
            "n_accepted": len(accepted), "n_rounds": len(rounds),
            "cost_usd": sum(r["cost_usd"] or 0.0 for r in rounds),
            "accepted": [c.as_dict() for c in accepted],
            "rounds": rounds}
