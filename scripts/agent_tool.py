"""One agent tool call: one process, one JSON object on stdout. The only command the agent may run.

    .venv/bin/python scripts/agent_tool.py --list
    .venv/bin/python scripts/agent_tool.py get_table_info --args '{"dataset": "rel-f1"}'
    .venv/bin/python scripts/agent_tool.py profile_table --args '{"dataset": "rel-f1", "table": "results"}'

A failed call still exits 0, with the error in the JSON. Everything a tool prints goes to stderr, so
stdout carries only the result.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", nargs="?", help="tool name; omit with --list")
    ap.add_argument("--args", default="{}", help="arguments as a JSON object")
    ap.add_argument("--list", action="store_true", help="print the tool catalogue and exit")
    ap.add_argument("--schemas", action="store_true", help="print the tools' JSON schemas and exit")
    a = ap.parse_args(argv)

    from moregt.agent.tools import dispatch, render_catalog, schemas

    if a.list:
        print(render_catalog())
        return 0
    if a.schemas:
        print(json.dumps(schemas(), indent=2))
        return 0
    if not a.tool:
        ap.error("give a tool name, or --list")
    try:
        args = json.loads(a.args)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": {"type": "BadArguments", "tool": a.tool,
                                    "message": f"--args is not valid JSON: {exc}"}}))
        return 0
    with contextlib.redirect_stdout(sys.stderr):
        result = dispatch(a.tool, args)
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
