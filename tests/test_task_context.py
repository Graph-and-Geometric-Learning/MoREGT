"""Past labels as context: the task_history table and its temporal safety."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import rel_f1_available
from moregt.data.task_context import KNOWN_AT, task_history_table


def _task(rows, delta="7D"):
    from relbench.base import Table, TaskType

    df = pd.DataFrame(rows, columns=["timestamp", "user_id", "label"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    table = Table(df=df, fkey_col_to_pkey_table={"user_id": "users"}, pkey_col=None,
                  time_col="timestamp")
    return SimpleNamespace(time_col="timestamp", entity_col="user_id", entity_table="users",
                           target_col="label", timedelta=pd.Timedelta(delta),
                           task_type=TaskType.REGRESSION,
                           get_table=lambda split: table if split == "train" else Table(
                               df=df.iloc[:0], fkey_col_to_pkey_table={"user_id": "users"},
                               pkey_col=None, time_col="timestamp"))


def test_labels_become_known_when_their_window_closes():
    task = _task([("2020-01-01", 0, 1.0), ("2020-01-08", 0, 2.0), ("2020-01-01", 1, np.nan)])
    tbl = task_history_table(task, "my-task")
    assert list(tbl.df.columns) == ["user_id", "my-task", KNOWN_AT]
    assert len(tbl.df) == 2                                         # unlabelled rows dropped
    assert (tbl.df[KNOWN_AT] == pd.to_datetime(["2020-01-08", "2020-01-15"])).all()
    assert tbl.time_col == KNOWN_AT and tbl.fkey_col_to_pkey_table == {"user_id": "users"}


def test_overlapping_windows_are_rejected():
    task = _task([("2020-01-01", 0, 1.0), ("2020-01-03", 0, 2.0)])
    with pytest.raises(ValueError, match="closer than one window"):
        task_history_table(task, "my-task")


@rel_f1_available
def test_seeds_never_see_their_own_label(tmp_path):
    """On rel-f1: every sampled task_history row was known before its seed's time."""
    from relbench.tasks import get_task

    from moregt.data.collate import to_cell_batch
    from moregt.data.graph import build_graph, make_loader

    task = get_task("rel-f1", "driver-dnf", download=False)
    bundle = build_graph("rel-f1", cache_dir=str(tmp_path), task=task, task_name="driver-dnf")
    assert "task_history" in bundle.node_types
    tid = bundle.node_type_id["task_history"]
    seen = 0
    for split in ("train", "val", "test"):
        loader = make_loader(bundle, task, split, batch_size=64, shuffle=False)
        raw = next(iter(loader))
        st = raw["task_history"]
        if st.num_nodes:
            seed_t = raw[task.entity_table].seed_time[st.batch]
            assert (st.time <= seed_t).all()
            seen += st.num_nodes
        cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=1024)
        rows = cb.row_table == tid
        assert (cb.row_time[rows] <= cb.seed_time.view(-1, 1).expand_as(cb.row_time)[rows]).all()
    assert seen > 0
