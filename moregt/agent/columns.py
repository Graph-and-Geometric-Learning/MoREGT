"""Accepted derived columns -> extra numerical cells of a :class:`~moregt.data.graph.GraphBundle`.

The files written by ``scripts/derive_columns.py`` (``<dir>/<dataset>.<task>.json``) list the accepted
columns. :func:`add_derived_columns` evaluates each over its full table and appends it to that
table's ``TensorFrame`` as a numerical column with its own mean/std, NaN and +-inf mapped to 0. The
global column vocabulary is derived from the frames, so the new columns get column ids and name
embeddings like any native column, and the model is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .grammar import Candidate, evaluate, parse_expression


def agent_columns_path(directory: str | Path, dataset: str, task: str) -> Path:
    return Path(directory) / f"{dataset}.{task}.json"


def load_agent_columns(path: str | Path) -> list[Candidate]:
    """The accepted columns of one record, re-checked against the grammar."""
    out = []
    for a in json.loads(Path(path).read_text())["accepted"]:
        cand = Candidate(name=a["name"], table=a["table"], expression=a["expression"])
        cand.columns = parse_expression(cand.expression)
        out.append(cand)
    return out


def add_derived_columns(bundle, db, candidates: list[Candidate]):
    """A copy of ``bundle`` with ``candidates`` appended as numerical cells.

    ``db`` is the relbench ``Database`` the bundle was built from (its rows are the graph's nodes, in
    order). A name already present in a table is skipped, as is a repeated name (first one wins).
    """
    import copy

    import torch
    import torch_frame
    from torch_frame import TensorFrame
    from torch_frame.data.stats import StatType

    by_table: dict[str, list[Candidate]] = {}
    for cand in candidates:
        by_table.setdefault(cand.table, []).append(cand)

    data = copy.copy(bundle.data)  # copies the node stores, not the tensors
    stats = dict(bundle.col_stats_dict)
    added: dict[str, list[str]] = {}
    for table, cands in by_table.items():
        if table not in data.node_types:
            continue
        tf = data[table].tf
        raw = db.table_dict[table].df
        if len(raw) != tf.num_rows:
            raise ValueError(f"{table}: {len(raw)} database rows but {tf.num_rows} graph rows")
        feats = dict(tf.feat_dict)
        names = {st: list(cols) for st, cols in tf.col_names_dict.items()}
        existing = {c for cols in names.values() for c in cols}
        table_stats = dict(stats.get(table, {}))

        new_cols, new_names = [], []
        for cand in cands:
            if cand.name in existing or cand.name in new_names:
                continue
            values = evaluate(raw, cand).to_numpy()
            column = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
            new_cols.append(torch.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0))
            new_names.append(cand.name)
            finite = values[~np.isnan(values)]
            table_stats[cand.name] = {
                StatType.MEAN: float(finite.mean()) if finite.size else 0.0,
                StatType.STD: float(finite.std()) if finite.size > 1 else 1.0,
                StatType.QUANTILES: ([float(q) for q in np.quantile(finite, [0, .25, .5, .75, 1])]
                                     if finite.size else [0.0] * 5),
            }
        if not new_cols:
            continue
        block = torch.cat(new_cols, dim=1)
        if torch_frame.numerical in feats:
            feats[torch_frame.numerical] = torch.cat([feats[torch_frame.numerical], block], dim=1)
            names[torch_frame.numerical] = names[torch_frame.numerical] + new_names
        else:
            feats[torch_frame.numerical] = block
            names[torch_frame.numerical] = new_names
        data[table].tf = TensorFrame(feat_dict=feats, col_names_dict=names, y=tf.y)
        stats[table] = table_stats
        added[table] = new_names

    out = replace(bundle, data=data, col_stats_dict=stats)
    out.derived_columns = added
    return out


def apply_agent_columns(bundle, directory: str | Path, task: str, *, db=None):
    """:func:`add_derived_columns` with the record ``<directory>/<dataset>.<task>.json``.

    Raises if the record does not exist, so a run never silently trains without its columns.
    """
    path = agent_columns_path(directory, bundle.dataset_name, task)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run scripts/derive_columns.py "
                                f"--tasks {bundle.dataset_name}/{task}")
    if db is None:
        from relbench.datasets import get_dataset

        db = get_dataset(bundle.dataset_name, download=False).get_db(upto_test_timestamp=False)
    return add_derived_columns(bundle, db, load_agent_columns(path))
