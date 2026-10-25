"""A database with RelBench's declared schema removed.

The agent sees table names, column names, dtypes, row counts and the rows themselves. It never sees
what RelBench *declares*: primary keys, foreign keys and time columns. :class:`BlindDB` has nowhere to
store them, and :func:`assert_keys_hidden` checks every view handed to the agent for a leaked key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

#: Field names that must never appear in a view handed to the agent.
FORBIDDEN = ("pkey_col", "fkey_col_to_pkey_table", "fkey_col_to_pkey", "time_col")


class ToolError(Exception):
    """A tool refused the call. The message is read by the agent, so it says what to do instead."""


class BlindnessViolation(AssertionError):
    """A view handed to the agent carries a declared key."""


@dataclass(frozen=True)
class BlindTable:
    name: str
    n_rows: int
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    df: pd.DataFrame

    def describe(self) -> dict:
        return {"table": self.name, "n_rows": self.n_rows,
                "columns": [{"name": c, "dtype": self.dtypes[c]} for c in self.columns]}


class BlindDB:
    """Read-only tables of ``dataset``, without keys."""

    def __init__(self, dataset: str, tables: dict[str, BlindTable]):
        self.dataset = dataset
        self.tables = tables

    def table_names(self) -> list[str]:
        return sorted(self.tables)

    def table(self, name: str) -> BlindTable:
        if name not in self.tables:
            raise ToolError(f"no table {name!r} in {self.dataset}; "
                            f"tables are {', '.join(self.table_names())}")
        return self.tables[name]

    def column(self, table: str, col: str) -> pd.Series:
        t = self.table(table)
        if col not in t.columns:
            raise ToolError(f"no column {col!r} in {self.dataset}.{table}; "
                            f"columns are {', '.join(t.columns)}")
        return t.df[col]

    def describe(self) -> dict:
        return {"dataset": self.dataset,
                "tables": [self.tables[n].describe() for n in self.table_names()]}


def blind_db(db, dataset: str = "") -> BlindDB:
    """Strip a relbench ``Database`` down to its data and names. Frames are shared, not copied."""
    tables = {}
    for name, t in db.table_dict.items():
        df = t.df
        tables[name] = BlindTable(name=name, n_rows=int(len(df)),
                                  columns=tuple(str(c) for c in df.columns),
                                  dtypes={str(c): str(df[c].dtype) for c in df.columns}, df=df)
    return BlindDB(dataset, tables)


def _walk(obj: Any, path: str = "$"):
    """``(path, key, value)`` for every mapping entry of a JSON-like structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def assert_keys_hidden(view: Any, db) -> None:
    """Raise :class:`BlindnessViolation` if ``view`` names a declared-schema field, or maps an FK
    column to its true parent table anywhere in its structure."""
    if hasattr(view, "table_dict") or any(hasattr(view, a) for a in FORBIDDEN):
        raise BlindnessViolation(f"{type(view).__name__} exposes the declared schema")
    rendered = json.loads(json.dumps(view, default=str))
    truth = {(c, parent) for t in db.table_dict.values()
             for c, parent in t.fkey_col_to_pkey_table.items()}
    for where, key, value in _walk(rendered):
        if key in FORBIDDEN:
            raise BlindnessViolation(f"{where}.{key} is a declared-schema field")
        if isinstance(value, str) and (key, value) in truth:
            raise BlindnessViolation(f"{where} maps {key!r} -> {value!r}, a declared FK edge")
