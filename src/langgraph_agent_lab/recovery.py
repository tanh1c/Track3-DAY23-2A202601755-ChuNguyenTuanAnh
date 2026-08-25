"""Executable proof that SQLite checkpoints survive Python process restart."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .bonus_evidence import RecoveryEvidence

_PROVIDER_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in _PROVIDER_VARS:
        env.pop(name, None)
    env["LLM_MODEL"] = ""
    env["LLM_JUDGE"] = "false"
    env["LANGGRAPH_INTERRUPT"] = "false"
    return env


def _run_worker(mode: str, database: Path, thread_id: str) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "langgraph_agent_lab.recovery_worker",
            mode,
            "--database",
            str(database),
            "--thread-id",
            thread_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=_child_env(),
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{mode} recovery worker produced no JSON output")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise ValueError(f"{mode} recovery worker returned a non-object payload")
    return payload


def _coerce_pid(value: object) -> int:
    """Convert the JSON PID field only when its runtime type is safe to parse."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def verify_subprocess_recovery(database: str | Path) -> RecoveryEvidence:
    """Write in one Python process, exit, then recover in a distinct process."""
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    thread_id = f"subprocess-recovery-{uuid.uuid4().hex}"

    try:
        writer = _run_worker("write", path, thread_id)
        reader = _run_worker("read", path, thread_id)
        writer_pid = _coerce_pid(writer.get("pid", 0))
        reader_pid = _coerce_pid(reader.get("pid", 0))
        same_thread_id = (
            str(writer.get("thread_id", ""))
            == str(reader.get("thread_id", ""))
            == thread_id
        )
        persisted_finalized = bool(writer.get("finalized")) and bool(
            reader.get("finalized")
        )
        distinct_processes = writer_pid > 0 and reader_pid > 0 and writer_pid != reader_pid
        verified = distinct_processes and same_thread_id and persisted_finalized
        return RecoveryEvidence(
            implemented=True,
            verified=verified,
            writer_pid=writer_pid,
            reader_pid=reader_pid,
            distinct_processes=distinct_processes,
            same_thread_id=same_thread_id,
            persisted_finalized=persisted_finalized,
            thread_id=thread_id,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return RecoveryEvidence(implemented=True, thread_id=thread_id)
