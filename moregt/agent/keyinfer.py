"""Primary- and foreign-key candidates inferred from data alone.

A primary key is a unique, non-null column with an id-like name. A foreign key needs either name
evidence (the column is named after the parent table or its key) confirmed by >= 95% value
containment in the parent key, or, without name evidence, near-total containment plus size guards.
Value containment alone is not enough: a small-integer count is often fully contained in an id range.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

#: Values sampled per column before set operations.
SAMPLE_CAP = 2_000_000
SAMPLE_SEED = 0

NAME_CONTAINMENT = 0.95
BLIND_CONTAINMENT = 0.999
BLIND_MIN_DISTINCT = 100
BLIND_MIN_RATIO = 0.05
BLIND_MAX_RATIO = 1.0


def singularise(word: str) -> str:
    """``events -> event``, ``studies -> study``, ``addresses -> address``."""
    for suffix in ("ies", "ses", "s"):
        if word.endswith(suffix):
            return word[:-3] + "y" if suffix == "ies" else (word[:-3] if suffix == "ses" else word[:-1])
    return word


def _values(series: pd.Series) -> set | None:
    """Distinct non-null values (sampled above the cap); ``None`` for unhashable cells."""
    s = series.dropna()
    if len(s) > SAMPLE_CAP:
        s = s.sample(SAMPLE_CAP, random_state=SAMPLE_SEED)
    try:
        return set(s.unique().tolist())
    except TypeError:
        return None


def infer_pkeys(bdb) -> dict[str, dict]:
    """Per table, the best primary-key candidate, or ``column=None`` when none qualifies."""
    out: dict[str, dict] = {}
    for name in bdb.table_names():
        t = bdb.table(name)
        n = t.n_rows
        best = None
        for col in t.columns:
            s = t.df[col]
            if s.isna().any():
                continue
            try:
                if s.nunique() != n:
                    continue
            except TypeError:
                continue
            score = 2.0 if col.lower().endswith("id") else 0.0
            score += 1.0 if singularise(name).lower() in col.lower() else 0.0
            if best is None or score > best[1]:
                best = (col, score)
        if best and best[1] > 0:
            out[name] = {"column": best[0], "confidence": round(min(1.0, best[1] / 3.0), 3),
                         "evidence": {"unique": True, "non_null": True,
                                      "name_is_id_like": best[0].lower().endswith("id"),
                                      "name_matches_table": singularise(name).lower() in best[0].lower()}}
        else:
            out[name] = {"column": None, "confidence": 0.0,
                         "evidence": {"reason": "no unique non-null column with an id-like name"}}
    return out


def _name_rule(col: str, parent: str, parent_pk: str | None) -> str | None:
    c, p = col.lower(), singularise(parent).lower()
    if parent_pk and c == parent_pk.lower():
        return "column == parent primary key"
    if c == p:
        return "column == singular(parent table)"
    if c in (p + "id", p + "_id"):
        return "column == singular(parent table) + id"
    if c.endswith(p + "id"):
        return "column ends with singular(parent table) + id"
    return None


def infer_fkeys(bdb, pkeys: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """Candidate FK edges keyed by ``(child table, child column)``; self-edges are allowed."""
    parent_values = {}
    for tbl, pk in pkeys.items():
        if not pk["column"]:
            continue
        vals = _values(bdb.table(tbl).df[pk["column"]])
        if vals:
            parent_values[tbl] = vals

    out: dict[tuple[str, str], dict] = {}
    for child in bdb.table_names():
        t = bdb.table(child)
        for col in t.columns:
            if pkeys.get(child, {}).get("column") == col:
                continue
            cv = _values(t.df[col])
            if not cv:
                continue
            cands = []
            for parent, pv in parent_values.items():
                if type(next(iter(cv))) is not type(next(iter(pv))):
                    continue
                containment = len(cv & pv) / len(cv)
                ratio = len(cv) / len(pv)
                rule = _name_rule(col, parent, pkeys[parent]["column"])
                ev = {"containment": round(containment, 4), "n_distinct_child": len(cv),
                      "n_distinct_parent": len(pv), "cardinality_ratio": round(ratio, 4),
                      "name_rule": rule, "parent_key": pkeys[parent]["column"]}
                if rule and containment >= NAME_CONTAINMENT:
                    cands.append((parent, 2.0 + containment, ev, "name confirmed by containment"))
                elif (containment >= BLIND_CONTAINMENT and len(cv) >= BLIND_MIN_DISTINCT
                      and BLIND_MIN_RATIO <= ratio <= BLIND_MAX_RATIO):
                    cands.append((parent, containment, ev, "containment only, no name evidence"))
            if not cands:
                continue
            parent, score, ev, why = max(cands, key=lambda c: c[1])
            out[(child, col)] = {
                "child_table": child, "child_column": col, "parent_table": parent,
                "confidence": round(min(1.0, score / 3.0), 3), "basis": why, "evidence": ev,
                "alternatives": [{"parent_table": p, "containment": e["containment"],
                                  "name_rule": e["name_rule"]}
                                 for p, s, e, _ in sorted(cands, key=lambda c: -c[1])[1:4]],
            }
    return out


def _fingerprint(bdb) -> str:
    return hashlib.sha1(json.dumps(bdb.describe(), sort_keys=True).encode()).hexdigest()[:16]


def infer(bdb, *, cache_dir: Path | None = None) -> dict:
    """Keys of a whole :class:`~moregt.agent.blind.BlindDB`, cached by schema fingerprint."""
    key = _fingerprint(bdb)
    path = (cache_dir / f"{bdb.dataset or 'db'}-{key}.json") if cache_dir else None
    if path and path.exists():
        return json.loads(path.read_text()) | {"cached": True}

    pkeys = infer_pkeys(bdb)
    fkeys = infer_fkeys(bdb, pkeys)
    out = {
        "dataset": bdb.dataset,
        "primary_keys": pkeys,
        "foreign_keys": [fkeys[k] for k in sorted(fkeys)],
        "sampled": any(bdb.table(t).n_rows > SAMPLE_CAP for t in bdb.table_names()),
        "sample_cap": SAMPLE_CAP,
        "cached": False,
    }
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
    return out
