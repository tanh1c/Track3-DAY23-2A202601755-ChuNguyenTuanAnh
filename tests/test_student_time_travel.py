from pathlib import Path

import pytest

from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.time_travel import (
    build_time_travel_demo_graph,
    find_checkpoint,
    fork_checkpoint,
    list_checkpoints,
    replay_checkpoint,
    verify_time_travel,
)


def _seed_graph(database: Path, thread_id: str):
    graph = build_time_travel_demo_graph(build_checkpointer("sqlite", str(database)))
    graph.invoke(
        {"topic": "baseline", "events": [], "rendered": ""},
        config={"configurable": {"thread_id": thread_id}},
    )
    return graph


def test_time_travel_lists_exact_checkpoint_and_rejects_unknown(tmp_path: Path) -> None:
    graph = _seed_graph(tmp_path / "history.sqlite", "thread-history")
    history = list_checkpoints(graph, "thread-history")
    assert history
    checkpoint_info = next(item for item in history if item.next_nodes)
    selected = find_checkpoint(graph, "thread-history", checkpoint_info.checkpoint_id)
    assert selected.config["configurable"]["checkpoint_id"] == checkpoint_info.checkpoint_id

    with pytest.raises(ValueError, match="Unknown checkpoint"):
        find_checkpoint(graph, "thread-history", "missing-checkpoint")


def test_replay_and_fork_preserve_original_history(tmp_path: Path) -> None:
    graph = _seed_graph(tmp_path / "travel.sqlite", "thread-travel")
    history = list_checkpoints(graph, "thread-travel")
    checkpoint_info = next(item for item in history if item.next_nodes == ("render",))
    selected = find_checkpoint(graph, "thread-travel", checkpoint_info.checkpoint_id)

    replayed = replay_checkpoint(graph, selected)
    assert replayed["rendered"] == "render:baseline"
    assert any(event["node"] == "finalize" for event in replayed["events"])

    before = {item.checkpoint_id for item in list_checkpoints(graph, "thread-travel")}
    fork_config, forked = fork_checkpoint(graph, selected, {"topic": "forked"})
    after = {item.checkpoint_id for item in list_checkpoints(graph, "thread-travel")}

    assert before <= after
    assert fork_config["configurable"]["checkpoint_id"] not in before
    assert forked["rendered"] == "render:forked"
    assert any(event["node"] == "finalize" for event in forked["events"])


def test_verify_time_travel_returns_strict_evidence(tmp_path: Path) -> None:
    evidence = verify_time_travel(str(tmp_path / "verify.sqlite"))
    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.replay_verified is True
    assert evidence.fork_verified is True
    assert evidence.original_history_preserved is True
    assert evidence.checkpoint_id
