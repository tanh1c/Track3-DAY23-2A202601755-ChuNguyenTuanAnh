from pathlib import Path

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer


def _initial_state(thread_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "scenario_id": "synthetic-recovery",
        "query": "How do I reset my password?",
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": 3,
        "final_answer": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def test_sqlite_checkpointer_can_be_created(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    saver = build_checkpointer("sqlite", str(db))
    assert saver is not None
    assert db.exists()


def test_sqlite_state_survives_fresh_graph_instance(tmp_path: Path) -> None:
    db = tmp_path / "recovery.sqlite"
    thread_id = "thread-recovery-test"
    config = {"configurable": {"thread_id": thread_id}}

    first = build_graph(build_checkpointer("sqlite", str(db)))
    result = first.invoke(_initial_state(thread_id), config=config)
    assert result["events"][-1]["node"] == "finalize"

    second = build_graph(build_checkpointer("sqlite", str(db)))
    snapshot = second.get_state(config)
    assert snapshot.values.get("events")
    assert snapshot.values.get("route") == "simple"

    history = list(second.get_state_history(config))
    assert len(history) >= 2
    assert any(item.values.get("events") for item in history)
