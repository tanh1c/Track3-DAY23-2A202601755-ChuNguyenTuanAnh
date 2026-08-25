"""Small subprocess worker used to prove durable SQLite checkpoint recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .graph import build_graph
from .persistence import build_checkpointer
from .state import Route, Scenario, initial_state


def _finalized(values: dict[str, Any]) -> bool:
    events = list(values.get("events", []) or [])
    return any(event.get("node") == "finalize" for event in events)


def _write(database: Path, thread_id: str) -> dict[str, object]:
    graph = build_graph(checkpointer=build_checkpointer("sqlite", str(database)))
    scenario = Scenario(
        id="subprocess-recovery",
        query="How do I reset my password?",
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)
    state["thread_id"] = thread_id
    config = {"configurable": {"thread_id": thread_id}}
    final_state = dict(graph.invoke(state, config=config))
    return {
        "pid": os.getpid(),
        "thread_id": thread_id,
        "finalized": _finalized(final_state),
    }


def _read(database: Path, thread_id: str) -> dict[str, object]:
    graph = build_graph(checkpointer=build_checkpointer("sqlite", str(database)))
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    values = dict(snapshot.values or {})
    return {
        "pid": os.getpid(),
        "thread_id": thread_id,
        "finalized": _finalized(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    if args.mode == "write":
        payload = _write(args.database, args.thread_id)
    else:
        payload = _read(args.database, args.thread_id)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
