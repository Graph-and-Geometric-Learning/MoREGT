"""Past labels as context: the task table becomes an ordinary relational table.

:func:`append_task_history` adds one table, ``task_history``, to the database before the graph is
built. It has one row per train/val task row: an FK to the entity table, the label as a numerical
column named after the task, and the time ``label_known_at = timestamp + task.timedelta``, the moment
the label's prediction window closes.

No model change is needed. The temporal sampler only admits rows with ``time <= seed_time``, so a seed
sees exactly the labels that were known at its own timestamp and never its own label, whose window
closes ``timedelta`` after the seed time. Test rows are never added.

A task-history graph reuses the task-free graph cache: :func:`task_context_cache_dir` symlinks every
materialised table and only ``task_history`` is materialised anew.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

TASK_TABLE = "task_history"
KNOWN_AT = "label_known_at"


def task_history_table(task, task_name: str, *, splits: tuple[str, ...] = ("train", "val")):
    """The ``task_history`` relbench ``Table`` built from ``splits`` of ``task``."""
    from relbench.base import Table

    from .graph import coerce_binary_target

    frames = []
    for split in splits:
        df = coerce_binary_target(task.get_table(split), task).df
        if task.target_col not in df.columns:
            raise ValueError(f"split {split!r} of {task_name} has no {task.target_col!r} column")
        frames.append(df[[task.time_col, task.entity_col, task.target_col]])
    df = pd.concat(frames, ignore_index=True)
    df = df[df[task.target_col].notna()].reset_index(drop=True)

    # relbench converts times assuming nanoseconds; pin the unit so this table shares the DB's clock.
    ts = pd.to_datetime(df[task.time_col]).astype("datetime64[ns]")
    delta = pd.Timedelta(task.timedelta)

    # Consecutive task rows of one entity must be at least one window apart, otherwise a label would
    # still be undetermined when the next seed reads it.
    order = np.lexsort((ts.to_numpy(), df[task.entity_col].to_numpy()))
    ent = df[task.entity_col].to_numpy()[order]
    tt = ts.to_numpy()[order]
    gaps = (tt[1:] - tt[:-1])[ent[1:] == ent[:-1]]
    if len(gaps) and gaps.min() < np.timedelta64(delta.to_timedelta64()):
        raise ValueError(f"{task_name}: an entity has two task rows closer than one window ({delta})")

    out = pd.DataFrame({
        task.entity_col: df[task.entity_col].to_numpy().astype(np.int64),
        task_name: df[task.target_col].to_numpy().astype(np.float64),
        KNOWN_AT: ts + delta,
    })
    return Table(df=out, fkey_col_to_pkey_table={task.entity_col: task.entity_table},
                 pkey_col=None, time_col=KNOWN_AT)


def append_task_history(db, task, task_name: str):
    """A new ``Database``: ``db`` plus the ``task_history`` table of ``task``."""
    from relbench.base import Database

    if TASK_TABLE in db.table_dict:
        raise ValueError(f"database already has a {TASK_TABLE!r} table")
    return Database(table_dict={**db.table_dict, TASK_TABLE: task_history_table(task, task_name)})


def task_history_stypes(task_name: str) -> dict:
    import torch_frame

    return {task_name: torch_frame.numerical, KNOWN_AT: torch_frame.timestamp}


def task_context_cache_dir(base: str | os.PathLike, task_name: str) -> Path:
    """``<base>/taskctx/<task_name>``, pre-populated with symlinks to ``base``'s table caches."""
    base = Path(base)
    target = base / "taskctx" / task_name
    target.mkdir(parents=True, exist_ok=True)
    for src in base.glob("*.pt"):
        dst = target / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return target
