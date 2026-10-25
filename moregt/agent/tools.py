"""The agent's tools: a registry, and four read-only tools over the blind database.

* ``get_table_info``  tables, columns, dtypes, semantic types and row counts
* ``profile_table``   :func:`profile` of every column of one table
* ``profile_column``  null rate, cardinality, and quantiles or the most frequent values of one column
* ``infer_keys``      key candidates inferred from the data (:mod:`.keyinfer`)

None of them can return a declared key, a task table or a label. Each call is one process of
``scripts/agent_tool.py``; :func:`dispatch` turns every failure into ``{"error": ...}`` so a failed call
reaches the agent as a message instead of a traceback.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import keyinfer
from .blind import ToolError, assert_keys_hidden, blind_db

#: Most frequent values reported for a non-numeric column, and rows sampled for a profile.
TOP_K = 15
PROFILE_SAMPLE = 2_000_000
#: Seed of relbench's stype proposal, which samples rows.
STYPE_SEED = 0

TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list,
         "object": dict}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    fn: Callable[..., dict]
    params: dict = field(default_factory=dict)
    required: tuple[str, ...] = ()

    def schema(self) -> dict:
        return {"name": self.name, "description": self.summary,
                "input_schema": {"type": "object", "properties": self.params,
                                 "required": list(self.required)}}


TOOLS: dict[str, ToolSpec] = {}


def tool(name: str, *, summary: str, params: dict, required: tuple[str, ...] = ()):
    """Register a function as a tool; ``summary`` is the description the agent reads."""

    def deco(fn):
        if name in TOOLS:
            raise RuntimeError(f"tool {name!r} registered twice")
        TOOLS[name] = ToolSpec(name=name, summary=summary, fn=fn, params=params, required=tuple(required))
        return fn

    return deco


def schemas() -> list[dict]:
    return [TOOLS[n].schema() for n in sorted(TOOLS)]


def render_catalog() -> str:
    """The tool list as it appears in the agent's prompt."""
    lines = []
    for n in sorted(TOOLS):
        spec = TOOLS[n]
        args = ", ".join(p + ("" if p in spec.required else "?") for p in spec.params)
        lines.append(f"- `{n}({args})` — {spec.summary}")
    return "\n".join(lines)


def _check(spec: ToolSpec, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ToolError(f"{spec.name}: arguments must be a JSON object, got {type(args).__name__}")
    unknown = sorted(set(args) - set(spec.params))
    if unknown:
        raise ToolError(f"{spec.name}: unknown argument(s) {unknown}; accepted are {sorted(spec.params)}")
    missing = sorted(set(spec.required) - set(args))
    if missing:
        raise ToolError(f"{spec.name}: missing required argument(s) {missing}")
    out = {}
    for k, v in args.items():
        want = spec.params[k].get("type")
        if want in TYPES and v is not None and not isinstance(v, TYPES[want]):
            try:
                v = {"integer": int, "number": float, "string": str, "boolean": bool}[want](v)
            except (KeyError, TypeError, ValueError):
                raise ToolError(f"{spec.name}: argument {k!r} should be a {want}, "
                                f"got {type(v).__name__}") from None
        out[k] = v
    return out


def dispatch(name: str, args: dict | None = None) -> dict:
    """Call a tool by name. Always returns a dict."""
    t0 = time.time()
    if name not in TOOLS:
        return {"error": {"type": "UnknownTool", "tool": name,
                          "message": f"no tool named {name!r}; available: {', '.join(sorted(TOOLS))}"}}
    spec = TOOLS[name]
    try:
        out = spec.fn(**_check(spec, args or {}))
    except ToolError as exc:
        return {"error": {"type": type(exc).__name__, "tool": name, "message": str(exc)}}
    except Exception as exc:  # noqa: BLE001
        return {"error": {"type": type(exc).__name__, "tool": name, "message": str(exc),
                          "unexpected": True}}
    out.setdefault("tool", name)
    out.setdefault("seconds", round(time.time() - t0, 3))
    return out


# ---- databases -------------------------------------------------------------------------------------

_DBS: dict[str, object] = {}
_BLIND: dict[str, object] = {}
_STYPES: dict[str, dict] = {}


def register_db(dataset: str, db) -> None:
    """Serve ``db`` under the name ``dataset`` instead of loading it from RelBench."""
    _DBS[dataset] = db
    _BLIND.pop(dataset, None)
    _STYPES.pop(dataset, None)


def load_db(dataset: str):
    if dataset not in _DBS:
        from relbench.datasets import get_dataset

        try:
            _DBS[dataset] = get_dataset(dataset, download=False).get_db(upto_test_timestamp=False)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not load dataset {dataset!r}: {exc}") from None
    return _DBS[dataset]


def load_blind(dataset: str):
    if dataset not in _BLIND:
        _BLIND[dataset] = blind_db(load_db(dataset), dataset)
    return _BLIND[dataset]


def stype_proposal(db) -> dict:
    """relbench's stype proposal with a fixed seed, as ``{table: {column: stype}}``."""
    from relbench.modeling.utils import get_stype_proposal

    state = np.random.get_state()
    try:
        np.random.seed(STYPE_SEED)
        proposal = get_stype_proposal(db)
    except Exception:  # noqa: BLE001
        return {}
    finally:
        np.random.set_state(state)
    return {t: {str(c): str(st) for c, st in cols.items()} for t, cols in proposal.items()}


def _stypes(dataset: str) -> dict:
    if dataset not in _STYPES:
        _STYPES[dataset] = stype_proposal(load_db(dataset))
    return _STYPES[dataset]


# ---- tools -----------------------------------------------------------------------------------------

@tool("get_table_info",
      summary="Tables, columns, dtypes and row counts for a dataset. Pass `table` for one table's detail.",
      params={"dataset": {"type": "string", "description": "e.g. rel-f1"},
              "table": {"type": "string", "description": "one table; omit for the whole database"}},
      required=("dataset",))
def get_table_info(dataset: str, table: str | None = None) -> dict:
    bdb = load_blind(dataset)
    out = bdb.table(table).describe() if table else bdb.describe()
    stypes = _stypes(dataset)
    for entry in ([out] if table else out["tables"]):
        per_col = stypes.get(entry["table"], {})
        for col in entry["columns"]:
            col["stype"] = per_col.get(col["name"])
    assert_keys_hidden(out, load_db(dataset))
    out["blind"] = True
    return out


def profile(bdb, table: str, column: str, stypes: dict | None = None) -> dict:
    """Null rate, cardinality, and quantiles (numeric), range (datetime) or top values (other)."""
    series = bdb.column(table, column)
    n = int(len(series))
    n_valid = int(series.notna().sum())
    s = series.dropna()
    sampled = len(s) > PROFILE_SAMPLE
    if sampled:
        s = s.sample(PROFILE_SAMPLE, random_state=keyinfer.SAMPLE_SEED)
    out = {"dataset": bdb.dataset, "table": table, "column": column,
           "dtype": str(series.dtype), "stype": (stypes or {}).get(column),
           "n_rows": n, "n_null": n - n_valid,
           "null_rate": round(1 - n_valid / n, 6) if n else None,
           "sampled": sampled, "sample_cap": PROFILE_SAMPLE if sampled else None}
    try:
        n_distinct = int(s.nunique())
    except TypeError:
        out.update(n_distinct=None, array_valued=True,
                   note="cells are list-like, so cardinality is undefined")
        return out
    out["n_distinct"] = n_distinct
    out["distinct_ratio"] = round(n_distinct / len(s), 6) if len(s) else None
    out["is_unique"] = bool(n_distinct == n and n > 0)

    is_bool = pd.api.types.is_bool_dtype(series)
    if pd.api.types.is_numeric_dtype(series) and not is_bool and len(s):
        q = s.quantile([0, 0.25, 0.5, 0.75, 1.0])
        out["numeric"] = {"min": float(q.iloc[0]), "p25": float(q.iloc[1]), "median": float(q.iloc[2]),
                          "p75": float(q.iloc[3]), "max": float(q.iloc[4]),
                          "mean": float(s.mean()), "std": float(s.std()) if len(s) > 1 else 0.0,
                          "n_negative": int((s < 0).sum()),
                          "integral": bool(np.all(np.equal(np.mod(s.to_numpy(dtype="float64"), 1), 0)))}
    elif pd.api.types.is_datetime64_any_dtype(series) and len(s):
        out["datetime"] = {"min": str(s.min()), "max": str(s.max())}
    elif len(s):
        counts = s.value_counts().head(TOP_K)
        out["boolean"] = bool(is_bool)
        out["top_values"] = [{"value": str(v), "count": int(c)} for v, c in counts.items()]
        lens = s.astype(str).str.len()
        out["text_length"] = {"min": int(lens.min()), "median": float(lens.median()),
                              "max": int(lens.max())}
    return out


@tool("profile_column",
      summary=("Null rate, cardinality, uniqueness and a value summary for one column. The evidence "
               "for deciding what a column is, without reading the whole table."),
      params={"dataset": {"type": "string"}, "table": {"type": "string"},
              "column": {"type": "string"}},
      required=("dataset", "table", "column"))
def profile_column(dataset: str, table: str, column: str) -> dict:
    return profile(load_blind(dataset), table, column, _stypes(dataset).get(table, {}))


@tool("profile_table",
      summary=("Profile EVERY column of one table in a single call. Prefer this over calling "
               "profile_column repeatedly -- each tool call is a fresh process that reloads the "
               "database, so one call per table is far faster than one per column."),
      params={"dataset": {"type": "string"}, "table": {"type": "string"}},
      required=("dataset", "table"))
def profile_table(dataset: str, table: str) -> dict:
    bdb = load_blind(dataset)
    stypes = _stypes(dataset).get(table, {})
    t = bdb.table(table)
    return {"dataset": dataset, "table": table, "n_rows": t.n_rows,
            "columns": [profile(bdb, table, col, stypes) for col in t.columns]}


@tool("infer_keys",
      summary=("Primary- and foreign-key candidates from data alone, with the evidence for each "
               "(containment, cardinality, which name rule fired). Does not read the declared schema."),
      params={"dataset": {"type": "string"},
              "min_confidence": {"type": "number", "description": "drop candidates below this (default 0)"}},
      required=("dataset",))
def infer_keys(dataset: str, min_confidence: float = 0.0) -> dict:
    cache = os.environ.get("MOREGT_AGENT_CACHE")
    out = keyinfer.infer(load_blind(dataset), cache_dir=Path(cache) / "infer_keys" if cache else None)
    if min_confidence:
        out = dict(out, foreign_keys=[e for e in out["foreign_keys"] if e["confidence"] >= min_confidence])
    # The FK candidates are predictions and may be right; everything else must not carry a key.
    assert_keys_hidden({k: v for k, v in out.items() if k != "foreign_keys"}, load_db(dataset))
    return out
