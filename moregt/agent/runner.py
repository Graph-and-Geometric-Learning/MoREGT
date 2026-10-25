"""Run the agent as a Claude Code subprocess that may call one command, and audit what it did.

The subprocess may run only ``scripts/agent_tool.py``; file, search, web and sub-agent tools are
denied. An allowlist is not proof, so :func:`audit` re-reads every tool call of the transcript: a call
outside the allowlist is recorded, and one that reaches the raw data files makes the run unusable.

Transports are injected: :class:`ClaudeCodeTransport` runs ``claude -p``, :class:`ReplayTransport`
replays a recorded transcript (tests).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: The one command the agent may run. ``$MOREGT_AGENT_PYTHON`` is the interpreter, relative to the
#: repository root or absolute.
PYTHON = os.environ.get("MOREGT_AGENT_PYTHON", ".venv/bin/python")
TOOL_COMMAND = f"{PYTHON} scripts/agent_tool.py"
ALLOWED_TOOLS = (f"Bash({TOOL_COMMAND}:*)",)
DISALLOWED_TOOLS = ("Read", "Write", "Edit", "Grep", "Glob", "NotebookEdit", "WebFetch", "WebSearch",
                    "Task", "Agent")
#: Claude Code writes an over-long tool result to a file and reads it back; such a read is benign.
SPILL_MARKERS = ("/tool-results/", "/.claude/projects/")


@dataclass
class AgentResult:
    text: str
    events: list[dict] = field(default_factory=list)
    cost_usd: float | None = None
    num_turns: int | None = None
    model: str | None = None
    returncode: int = 0
    stderr: str = ""
    #: Set on the stream's result event for API errors and turn limits, which still report success.
    is_error: bool = False


class ClaudeCodeTransport:
    """``claude -p`` with a one-command allowlist and a stream-json transcript."""

    def __init__(self, *, binary: str = "claude", max_turns: int = 200, timeout: int = 3600,
                 model: str | None = None):
        self.binary = binary
        self.max_turns = max_turns
        self.timeout = timeout
        self.model = model

    def __call__(self, prompt: str, *, system: str | None = None, env: dict | None = None) -> AgentResult:
        if not shutil.which(self.binary):
            raise RuntimeError(f"{self.binary!r} is not on PATH; the agent needs Claude Code installed")
        cmd = [self.binary, "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--allowed-tools", *ALLOWED_TOOLS, "--disallowed-tools", *DISALLOWED_TOOLS,
               "--max-turns", str(self.max_turns)]
        if self.model:
            cmd += ["--model", self.model]
        if system:
            cmd += ["--append-system-prompt", system]
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                              timeout=self.timeout, env={**os.environ, **(env or {})})
        return parse_stream(proc.stdout, returncode=proc.returncode, stderr=proc.stderr)


class ReplayTransport:
    """Replays recorded stream events (a ``.jsonl`` transcript or a list of dicts)."""

    def __init__(self, transcript: Path | str | list[dict]):
        if isinstance(transcript, (str, Path)):
            lines = Path(transcript).read_text().splitlines()
            self.events = [json.loads(line) for line in lines if line.strip()]
        else:
            self.events = list(transcript)

    def __call__(self, prompt: str, *, system: str | None = None, env: dict | None = None) -> AgentResult:
        return parse_stream("\n".join(json.dumps(e) for e in self.events))


def parse_stream(stdout: str, *, returncode: int = 0, stderr: str = "") -> AgentResult:
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    final = next((e for e in reversed(events) if e.get("type") == "result"), {})
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    return AgentResult(text=final.get("result", "") or "", events=events,
                       cost_usd=final.get("total_cost_usd"), num_turns=final.get("num_turns"),
                       model=init.get("model"), returncode=returncode, stderr=stderr,
                       is_error=bool(final.get("is_error")) or not final)


def tool_calls(result: AgentResult) -> list[dict]:
    """Every tool call of the transcript, in order, as ``{"name", "input"}``."""
    out = []
    for event in result.events:
        message = event.get("message") if isinstance(event, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):  # a plain string on text-only turns
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append({"name": block.get("name"), "input": block.get("input") or {}})
    return out


def _reaches_data(call: dict, data_roots: tuple[str, ...]) -> bool:
    blob = json.dumps(call.get("input") or {}, default=str)
    if any(marker in blob for marker in SPILL_MARKERS):
        return False
    return any(root and root in blob for root in data_roots)


def audit(result: AgentResult, *, data_roots: tuple[str, ...] | None = None) -> dict:
    """Counts of tool calls, of calls outside the allowlist, and of those that reached the data.

    ``usable`` is False when any call reached the data; such a run must not be recorded.
    """
    roots = data_roots if data_roots is not None else (
        os.environ.get("RELBENCH_CACHE_DIR") or "", "/relbench/", ".parquet", "/db/")
    calls = tool_calls(result)
    off = []
    for call in calls:
        inp = call["input"]
        command = str(inp.get("command", "")) if isinstance(inp, dict) else str(inp)
        if call["name"] != "Bash" or TOOL_COMMAND not in command:
            off.append(call)
    reached = [c for c in off if _reaches_data(c, roots)]
    return {"n_tool_calls": len(calls), "n_off_allowlist": len(off), "off_allowlist": off[:20],
            "n_data_access": len(reached), "usable": not reached}


def write_transcript(result: AgentResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for event in result.events:
            fh.write(json.dumps(event, default=str) + "\n")
    return path
