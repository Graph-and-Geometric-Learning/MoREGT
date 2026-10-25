"""Row-local derived columns: the expression grammar, and the checks a proposal must pass.

A derived column is an expression over the columns of **one row of one table**: column names,
numeric constants, ``+ - * / // % **``, unary signs, comparisons (which yield 0/1) and the element-wise
functions in :data:`ALLOWED_FUNCS`. Anything else (attributes, subscripts, ``and``/``or``, other
functions) does not parse, so an expression cannot aggregate, shift or read another row, and a derived
value is visible to the temporal sampler exactly when its row is.

Each proposal states its units and an expected range before it is computed. :func:`screen` rejects it
if it fails the grammar, references a missing or forbidden column, fails to evaluate, or if the
computed column (:func:`falsify`) is constant, creates nulls its inputs do not have, mostly falls
outside the stated range, or duplicates (|corr| > 0.999) an existing or earlier accepted column.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Call,  # only of ALLOWED_FUNCS, checked separately
)

ALLOWED_FUNCS = {"abs": np.abs, "log": np.log, "log1p": np.log1p, "sqrt": np.sqrt,
                 "minimum": np.minimum, "maximum": np.maximum}

#: Overlapping non-null rows required before a correlation can reject a candidate.
MIN_CORR_OVERLAP = 30


class GrammarError(ValueError):
    """The expression uses something the grammar does not allow."""


@dataclass
class Candidate:
    """One proposed derived column and the claims it is checked against."""

    name: str
    table: str
    expression: str
    description: str = ""
    units: str = ""
    expected_range: tuple[float, float] | None = None
    why_it_matters: str = ""
    columns: tuple[str, ...] = ()  # filled by parse_expression
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"name": self.name, "table": self.table, "expression": self.expression,
                "description": self.description, "units": self.units,
                "expected_range": list(self.expected_range) if self.expected_range else None,
                "why_it_matters": self.why_it_matters, "columns": list(self.columns),
                "problems": list(self.problems)}


def parse_expression(expression: str) -> tuple[str, ...]:
    """Check ``expression`` against the grammar and return the columns it references, sorted.

    The :class:`GrammarError` message is written for the agent to act on.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise GrammarError(f"not a valid expression: {exc.msg}") from None

    columns: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id not in ALLOWED_FUNCS:
                raise GrammarError(
                    f"only these functions are allowed: {', '.join(sorted(ALLOWED_FUNCS))}. "
                    "Anything that aggregates across rows (groupby, shift, rolling, cumsum) is "
                    "refused: it could read rows from the future.")
            continue
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_FUNCS:
                columns.add(node.id)
            continue
        if not isinstance(node, ALLOWED_NODES):
            raise GrammarError(
                f"{type(node).__name__} is not allowed. An expression may contain only column "
                "names, numbers, arithmetic, comparisons and "
                f"{', '.join(sorted(ALLOWED_FUNCS))} -- so that it can only ever read the row it is "
                "computed on.")
    if not columns:
        raise GrammarError("references no columns, so it is a constant")
    return tuple(sorted(columns))


def static_check(candidate: Candidate, *, table_columns, forbidden: set[str] | None = None) -> list[str]:
    """Problems found without touching data. Sets ``candidate.columns``."""
    try:
        candidate.columns = parse_expression(candidate.expression)
    except GrammarError as exc:
        return [str(exc)]
    problems: list[str] = []
    missing = sorted(set(candidate.columns) - set(table_columns))
    if missing:
        problems.append(f"no such column(s) in {candidate.table}: {', '.join(missing)}")
    banned = sorted(set(candidate.columns) & set(forbidden or ()))
    if banned:
        problems.append(f"references label column(s) {', '.join(banned)}; that is the answer, "
                        "not a feature")
    if not candidate.name or not candidate.name.replace("_", "").isalnum():
        problems.append("name must be alphanumeric with underscores")
    return problems


def evaluate(df: pd.DataFrame, candidate: Candidate) -> pd.Series:
    """The derived column over ``df``. Re-checks the grammar, so a stored expression is never run
    unchecked; only the referenced columns and :data:`ALLOWED_FUNCS` are in scope."""
    columns = parse_expression(candidate.expression)
    env = {c: pd.to_numeric(df[c], errors="coerce") for c in columns if c in df.columns}
    env.update(ALLOWED_FUNCS)
    out = eval(compile(ast.parse(candidate.expression, mode="eval"), "<derived>", "eval"),  # noqa: S307
               {"__builtins__": {}}, env)
    return pd.Series(out, index=df.index, dtype="float64")


def falsify(values: pd.Series, candidate: Candidate, df: pd.DataFrame, *, dup_corr: float = 0.999,
            min_in_range: float = 0.90, min_overlap: int = MIN_CORR_OVERLAP) -> list[str]:
    """Check a computed column against the proposer's own claims and against ``df``'s columns."""
    problems: list[str] = []
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return ["every value is null or infinite"]
    if finite.nunique() <= 1:
        problems.append("constant, so it carries nothing")

    input_null = float(df[list(candidate.columns)].isna().any(axis=1).mean()) if candidate.columns else 0.0
    out_null = float(values.replace([np.inf, -np.inf], np.nan).isna().mean())
    if out_null > input_null + 0.05:
        problems.append(f"null rate {out_null:.1%} against {input_null:.1%} in its inputs -- the "
                        "expression is producing undefined values (division by zero, log of a "
                        "negative)")

    if candidate.expected_range:
        lo, hi = candidate.expected_range
        inside = float(((finite >= lo) & (finite <= hi)).mean())
        if inside < min_in_range:
            problems.append(
                f"only {inside:.1%} of values fall in the stated range [{lo:g}, {hi:g}] "
                f"(actual {finite.min():g} to {finite.max():g}); the derivation does not compute "
                "what it claims to")

    # Every numeric column counts, including the expression's own inputs (``grid * 2`` is ``grid``).
    for col in df.columns:
        other = pd.to_numeric(df[col], errors="coerce")
        if other.nunique(dropna=True) <= 1:
            continue
        aligned = other.reindex(finite.index)
        overlap = int((finite.notna() & aligned.notna()).sum())
        if overlap < min_overlap or aligned.nunique(dropna=True) <= 1:
            continue
        corr = finite.corr(aligned)
        if corr is not None and np.isfinite(corr) and abs(corr) > dup_corr:
            problems.append(f"|corr| {abs(corr):.4f} with existing column `{col}` -- a rescaling of "
                            "something the model already has")
            break
    return problems


def screen(db, candidates: list[Candidate], *,
           forbidden: dict[str, set[str]] | None = None) -> tuple[list[Candidate], list[Candidate]]:
    """Split ``candidates`` into ``(accepted, rejected)``; rejected ones carry ``problems``.

    A candidate is also checked against the candidates accepted before it in the same batch.
    """
    accepted, rejected = [], []
    accepted_so_far: dict[str, pd.DataFrame] = {}
    for cand in candidates:
        if cand.table not in db.table_dict:
            cand.problems = [f"no such table {cand.table!r}"]
            rejected.append(cand)
            continue
        df = db.table_dict[cand.table].df
        problems = static_check(cand, table_columns=df.columns,
                                forbidden=(forbidden or {}).get(cand.table, set()))
        if problems:
            cand.problems = problems
            rejected.append(cand)
            continue
        try:
            values = evaluate(df, cand)
        except Exception as exc:  # noqa: BLE001
            cand.problems = [f"failed to evaluate: {type(exc).__name__}: {exc}"]
            rejected.append(cand)
            continue
        prior = accepted_so_far.get(cand.table)
        problems = falsify(values, cand, df if prior is None else pd.concat([df, prior], axis=1))
        if problems:
            cand.problems = problems
            rejected.append(cand)
        else:
            accepted.append(cand)
            accepted_so_far[cand.table] = (values.rename(cand.name).to_frame() if prior is None
                                           else prior.assign(**{cand.name: values}))
    return accepted, rejected


def forbidden_columns(task, db) -> dict[str, set[str]]:
    """Columns no expression may read: the target, when it is a column of the entity table."""
    entity = task.entity_table
    if task.target_col in db.table_dict[entity].df.columns:
        return {entity: {task.target_col}}
    return {}
