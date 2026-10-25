"""The agent layer: grammar, screening, blind tools, the derive loop, and derived cells."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import ENTITY, make_batch, make_bundle, schema_of, tiny_config

REPO = Path(__file__).resolve().parents[1]


def synthetic_db():
    from relbench.base import Database, Table

    rng = np.random.default_rng(0)
    n_users, n_events = 200, 1000
    users = pd.DataFrame({"user_id": np.arange(n_users),
                          "age": rng.integers(18, 80, n_users).astype(float),
                          "tier": rng.choice(["bronze", "silver", "gold"], n_users)})
    events = pd.DataFrame({"event_id": np.arange(n_events),
                           "user_id": rng.integers(0, n_users, n_events),
                           "amount": rng.gamma(2.0, 10.0, n_events),
                           "qty": rng.integers(1, 6, n_events).astype(float),
                           "placed_at": pd.Timestamp("2020-01-01")
                           + pd.to_timedelta(rng.integers(0, 365, n_events), unit="D")})
    return Database(table_dict={
        "users": Table(df=users, fkey_col_to_pkey_table={}, pkey_col="user_id", time_col=None),
        "events": Table(df=events, fkey_col_to_pkey_table={"user_id": "users"}, pkey_col="event_id",
                        time_col="placed_at")})


# ---- grammar and screening -------------------------------------------------------------------------

def test_grammar_accepts_row_local_expressions():
    from moregt.agent.grammar import parse_expression

    assert parse_expression("grid - position") == ("grid", "position")
    assert parse_expression("log1p(maximum(amount, 0)) / (qty >= 2)") == ("amount", "qty")


@pytest.mark.parametrize("expr", [
    "df.groupby('a')", "a[0]", "a and b", "sum(a)", "(lambda: a)()", "__import__('os')", "3 + 4",
    "a if b else c", "a.real", "maximum(a, b, out=c)", "a +",
])
def test_grammar_refuses_everything_else(expr):
    from moregt.agent.grammar import GrammarError, parse_expression

    with pytest.raises(GrammarError):
        parse_expression(expr)


def test_evaluate_is_elementwise():
    from moregt.agent.grammar import Candidate, evaluate

    df = pd.DataFrame({"a": [1.0, 4.0, None], "b": [2.0, 2.0, 2.0]})
    out = evaluate(df, Candidate("x", "t", "(a > b) + sqrt(a)"))
    assert out.iloc[:2].tolist() == [1.0, 3.0] and np.isnan(out.iloc[2])


def test_falsify_checks_the_claims():
    from moregt.agent.grammar import Candidate, evaluate, falsify, static_check

    df = synthetic_db().table_dict["events"].df

    def problems(expr, rng=None, frame=df):
        cand = Candidate("x", "events", expr, expected_range=rng)
        static_check(cand, table_columns=frame.columns)
        return falsify(evaluate(frame, cand), cand, frame)

    assert problems("amount / qty", (0, 1000)) == []
    assert any("constant" in p for p in problems("qty * 0 + 1"))
    assert any("stated range" in p for p in problems("amount", (0, 1)))
    assert any("|corr|" in p for p in problems("amount * 2"))
    assert any("null rate" in p for p in problems("log(qty - 3)"))
    # a correlation over fewer than MIN_CORR_OVERLAP rows rejects nothing
    sparse = df.assign(note=[1.0, 2.0] + [None] * (len(df) - 2))
    assert not any("note" in p for p in problems("amount / qty", frame=sparse))


def test_screen():
    from moregt.agent.grammar import Candidate, screen

    db = synthetic_db()
    cands = [Candidate("per_unit", "events", "amount / qty"),
             Candidate("per_unit_x2", "events", "2 * amount / qty"),       # duplicate of the first
             Candidate("nope", "orders", "amount"),
             Candidate("missing", "events", "price * qty"),
             Candidate("leak", "users", "age + 1"),
             Candidate("bad name", "events", "qty * amount")]
    kept, rejected = screen(db, cands, forbidden={"users": {"age"}})
    assert [c.name for c in kept] == ["per_unit"]
    reasons = {c.name: " ".join(c.problems) for c in rejected}
    assert "per_unit" in reasons["per_unit_x2"] and "no such table" in reasons["nope"]
    assert "price" in reasons["missing"] and "label" in reasons["leak"]
    assert "alphanumeric" in reasons["bad name"]


def test_forbidden_columns():
    from types import SimpleNamespace

    from moregt.agent.grammar import forbidden_columns

    db = synthetic_db()
    assert forbidden_columns(SimpleNamespace(entity_table="users", target_col="churn"), db) == {}
    task = SimpleNamespace(entity_table="users", target_col="age")
    assert forbidden_columns(task, db) == {"users": {"age"}}


# ---- blind tools -----------------------------------------------------------------------------------

@pytest.fixture
def served_db():
    from moregt.agent import tools

    db = synthetic_db()
    tools.register_db("synthetic-db", db)
    yield db
    tools._DBS.pop("synthetic-db", None)


def test_tools_hide_the_declared_schema(served_db):
    from moregt.agent.tools import dispatch

    info = dispatch("get_table_info", {"dataset": "synthetic-db"})
    assert info["blind"] and [t["table"] for t in info["tables"]] == ["events", "users"]
    text = json.dumps(info)
    assert not any(k in text for k in ("pkey_col", "fkey_col_to_pkey_table", "time_col", "primary_key"))
    events = dispatch("profile_table", {"dataset": "synthetic-db", "table": "events"})
    by_col = {c["column"]: c for c in events["columns"]}
    assert by_col["amount"]["numeric"]["min"] > 0 and "datetime" in by_col["placed_at"]
    tier = dispatch("profile_column", {"dataset": "synthetic-db", "table": "users", "column": "tier"})
    assert {v["value"] for v in tier["top_values"]} == {"bronze", "silver", "gold"}


def test_infer_keys_from_data(served_db):
    from moregt.agent.tools import dispatch

    out = dispatch("infer_keys", {"dataset": "synthetic-db"})
    assert out["primary_keys"]["users"]["column"] == "user_id"
    assert [(e["child_table"], e["child_column"], e["parent_table"]) for e in out["foreign_keys"]] == \
        [("events", "user_id", "users")]


def test_tool_errors_are_returned(served_db):
    from moregt.agent.tools import dispatch

    assert dispatch("read_file", {})["error"]["type"] == "UnknownTool"
    err = dispatch("get_table_info", {"dataset": "synthetic-db", "x": 1})["error"]
    assert "unknown argument" in err["message"]
    err = dispatch("profile_table", {"dataset": "synthetic-db", "table": "orders"})["error"]
    assert "no table" in err["message"]


def test_assert_keys_hidden():
    from moregt.agent.blind import BlindnessViolation, assert_keys_hidden

    db = synthetic_db()
    assert_keys_hidden({"columns": ["user_id"]}, db)
    with pytest.raises(BlindnessViolation):
        assert_keys_hidden({"a": [{"user_id": "users"}]}, db)
    with pytest.raises(BlindnessViolation):
        assert_keys_hidden({"pkey_col": "x"}, db)
    with pytest.raises(BlindnessViolation):
        assert_keys_hidden(db, db)


def test_agent_tool_cli():
    run = lambda *a: subprocess.run([sys.executable, "scripts/agent_tool.py", *a], cwd=REPO,
                                    capture_output=True, text=True, check=True).stdout
    listing = run("--list")
    assert [line.split("(")[0] for line in listing.splitlines()] == \
        ["- `get_table_info", "- `infer_keys", "- `profile_column", "- `profile_table"]
    assert json.loads(run("get_table_info", "--args", "{oops"))["error"]["type"] == "BadArguments"


# ---- runner, audit and the loop --------------------------------------------------------------------

def _events(text, calls=(), *, is_error=False, cost=0.5):
    content = [{"type": "tool_use", "name": n, "input": i} for n, i in calls]
    return [{"type": "system", "subtype": "init", "model": "test-model"},
            {"type": "assistant", "message": {"content": content}},
            {"type": "assistant", "message": {"content": "plain text turn"}},
            {"type": "result", "result": text, "total_cost_usd": cost, "is_error": is_error}]


def test_audit():
    from moregt.agent.runner import TOOL_COMMAND, ReplayTransport, audit

    ok = ("Bash", {"command": f"{TOOL_COMMAND} profile_table --args '{{}}'"})
    spill = ("Read", {"file_path": "/home/x/.claude/projects/p/tool-results/1.txt"})
    data = ("Read", {"file_path": "/data/relbench/rel-f1/db/results.parquet"})
    other = ("Bash", {"command": "cat notes.txt"})
    a = audit(ReplayTransport(_events("[]", [ok, spill, other]))(""))
    assert (a["n_tool_calls"], a["n_off_allowlist"], a["n_data_access"], a["usable"]) == (3, 2, 0, True)
    a = audit(ReplayTransport(_events("[]", [ok, data]))(""))
    assert (a["n_data_access"], a["usable"]) == (1, False)


def test_parse_stream_flags_errors():
    from moregt.agent.runner import ReplayTransport, parse_stream

    r = ReplayTransport(_events("x", is_error=True))("")
    assert r.is_error and r.model == "test-model" and r.cost_usd == 0.5
    assert parse_stream("not json\n").is_error  # no result event


def test_parse_proposals():
    from moregt.agent.derive_loop import parse_proposals

    reply = ('Here you go:\n```json\n[{"name": "per_unit", "table": "events", "expression": '
             '"amount / qty", "expected_range": [0, 500]}, {"name": "no_expr", "table": "events"}]\n```')
    (c,) = parse_proposals(reply)
    assert (c.name, c.expected_range) == ("per_unit", (0.0, 500.0))
    assert parse_proposals('[{"table": "t", "expression": "a"}]')[0].name == "unnamed"
    assert parse_proposals("no array here") == [] and parse_proposals("[1, 2]") == []


class ScriptedTransport:
    """Returns one scripted reply per round and records the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt, *, system=None, env=None):
        from moregt.agent.runner import ReplayTransport

        self.prompts.append((system, prompt))
        return ReplayTransport(self.replies.pop(0))("")


def test_derive_loop_revises_and_records(served_db, tmp_path):
    from moregt.agent import derive_loop
    from moregt.agent.runner import TOOL_COMMAND

    r0 = json.dumps([{"name": "per_unit", "table": "events", "expression": "amount / qty",
                      "units": "currency per item", "expected_range": [0, 1000]},
                     {"name": "doubled", "table": "events", "expression": "amount * 2"}])
    r1 = json.dumps([{"name": "bulk", "table": "events", "expression": "qty >= 4",
                      "expected_range": [0, 1]}]) + "\nDONE"
    call = ("Bash", {"command": f"{TOOL_COMMAND} get_table_info --args '{{}}'"})
    transport = ScriptedTransport([_events(r0, [call]), _events(r1)])
    rec = derive_loop.run("synthetic-db", "event-task", transport=transport, description="a test task",
                          forbidden={}, transcript_dir=tmp_path)

    assert [c["name"] for c in rec["accepted"]] == ["per_unit", "bulk"]
    assert rec["n_rounds"] == 2 and rec["complete"] and rec["model"] == "test-model"
    assert rec["cost_usd"] == 1.0 and rec["rounds"][0]["audit"]["n_tool_calls"] == 1
    assert rec["rounds"][0]["rejected"][0]["name"] == "doubled"
    system, first = transport.prompts[0]
    assert TOOL_COMMAND in system and "a test task" in first and "`profile_table(" in first
    revision = transport.prompts[1][1]
    assert "1 of your 2 proposals were rejected" in revision and "|corr|" in revision
    assert len(list(tmp_path.glob("synthetic-db.event-task.r*.jsonl"))) == 2


# ---- derived cells ---------------------------------------------------------------------------------

def _db_for(bundle):
    """A relbench Database holding the synthetic bundle's ``event`` values."""
    from relbench.base import Database, Table

    num = bundle.data["event"].tf.feat_dict
    import torch_frame

    vals = num[torch_frame.numerical].numpy()
    df = pd.DataFrame({"amount": vals[:, 0], "qty": vals[:, 1]})
    return Database(table_dict={"event": Table(df=df, fkey_col_to_pkey_table={}, pkey_col=None,
                                                time_col=None)})


def test_add_derived_columns(bundle):
    import torch_frame

    from moregt.agent.columns import add_derived_columns
    from moregt.agent.grammar import Candidate
    from moregt.data.collate import column_vocab, to_cell_batch
    from moregt.model.moregt import MoREGT

    cands = [Candidate("ratio", "event", "amount / qty"), Candidate("ratio", "event", "qty"),
             Candidate("big", "event", "qty > 0"), Candidate("qty", "event", "qty + 1")]
    out = add_derived_columns(bundle, _db_for(bundle), cands)

    assert out.derived_columns == {"event": ["ratio", "big"]}
    assert bundle.data["event"].tf.col_names_dict[torch_frame.numerical] == ["amount", "qty"]
    tf = out.data["event"].tf
    assert tf.col_names_dict[torch_frame.numerical] == ["amount", "qty", "ratio", "big"]
    x = tf.feat_dict[torch_frame.numerical]
    expected = torch.nan_to_num(x[:, 0] / x[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.allclose(x[:, 2], expected) and set(x[:, 3].tolist()) <= {0.0, 1.0}
    assert len(column_vocab(out)) == len(column_vocab(bundle)) + 2
    from torch_frame.data.stats import StatType

    assert out.col_stats_dict["event"]["big"][StatType.STD] >= 0

    raw = make_batch()
    raw["user"].tf = out.data["user"].tf[torch.tensor([0, 1])]
    raw["event"].tf = tf[torch.arange(raw["event"].num_nodes)]
    torch.manual_seed(0)
    model = MoREGT(out, schema_of(out), tiny_config())
    logits, _ = model(to_cell_batch(raw, out, ENTITY, seq_len=32))
    assert logits.shape == (2, 1) and torch.isfinite(logits).all()


def test_stored_expressions_are_rechecked(tmp_path):
    from moregt.agent.columns import add_derived_columns, load_agent_columns
    from moregt.agent.grammar import Candidate, GrammarError

    path = tmp_path / "rec.json"
    path.write_text(json.dumps({"accepted": [{"name": "x", "table": "event",
                                              "expression": "__import__('os').getcwd()"}]}))
    with pytest.raises(GrammarError):
        load_agent_columns(path)
    bundle = make_bundle()
    with pytest.raises(GrammarError):
        add_derived_columns(bundle, _db_for(bundle), [Candidate("x", "event", "().__class__")])


def test_released_agent_columns():
    from moregt.agent.columns import agent_columns_path, load_agent_columns
    from moregt.benchmark import TASKS

    files = sorted((REPO / "agent_columns").glob("*.json"))
    assert len(files) == len(TASKS)
    for ds, task in TASKS:
        path = agent_columns_path(REPO / "agent_columns", ds, task)
        rec = json.loads(path.read_text())
        assert (rec["dataset"], rec["task"]) == (ds, task) and rec["complete"]
        assert all(r["audit"]["usable"] and r["audit"]["n_data_access"] == 0 for r in rec["rounds"])
        cands = load_agent_columns(path)
        assert len(cands) == rec["n_accepted"] > 0
        for cand, stored in zip(cands, rec["accepted"], strict=True):
            assert list(cand.columns) == stored["columns"]
