"""Checkpoint inspection, replay, fork, and executable time-travel evidence."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel

from .bonus_evidence import TimeTravelEvidence
from .persistence import build_checkpointer


class CheckpointInfo(BaseModel):
    """Compact facts needed to select an exact checkpoint."""

    checkpoint_id: str
    thread_id: str
    next_nodes: tuple[str, ...]
    route: str = ""
    attempt: int = 0


class TimeTravelDemoState(TypedDict, total=False):
    """Small deterministic graph state used only to prove time-travel semantics."""

    topic: str
    rendered: str
    events: Annotated[list[dict[str, str]], add]


def _prepare(state: TimeTravelDemoState) -> dict[str, object]:
    topic = str(state.get("topic", "baseline"))
    return {"topic": topic, "events": [{"node": "prepare", "topic": topic}]}


def _render(state: TimeTravelDemoState) -> dict[str, object]:
    topic = str(state.get("topic", "baseline"))
    return {
        "rendered": f"render:{topic}",
        "events": [{"node": "render", "topic": topic}],
    }


def _finalize_demo(state: TimeTravelDemoState) -> dict[str, object]:
    return {"events": [{"node": "finalize", "topic": str(state.get("topic", ""))}]}


def build_time_travel_demo_graph(checkpointer: Any) -> Any:
    """Build a deterministic three-node graph with checkpoint boundaries."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(TimeTravelDemoState)
    builder.add_node("prepare", _prepare)
    builder.add_node("render", _render)
    builder.add_node("finalize", _finalize_demo)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "render")
    builder.add_edge("render", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def list_checkpoints(graph: Any, thread_id: str) -> list[CheckpointInfo]:
    """Return compact checkpoint facts for one exact thread."""
    config = {"configurable": {"thread_id": thread_id}}
    items: list[CheckpointInfo] = []
    for snapshot in graph.get_state_history(config):
        snapshot_config = dict(snapshot.config or {})
        configurable = dict(snapshot_config.get("configurable", {}) or {})
        checkpoint_id = str(configurable.get("checkpoint_id", ""))
        if not checkpoint_id:
            continue
        values = dict(snapshot.values or {})
        items.append(
            CheckpointInfo(
                checkpoint_id=checkpoint_id,
                thread_id=str(configurable.get("thread_id", thread_id)),
                next_nodes=tuple(snapshot.next or ()),
                route=str(values.get("route", "")),
                attempt=int(values.get("attempt", 0) or 0),
            )
        )
    return items


def find_checkpoint(graph: Any, thread_id: str, checkpoint_id: str) -> Any:
    """Return the exact requested snapshot or fail instead of silently using latest."""
    config = {"configurable": {"thread_id": thread_id}}
    for snapshot in graph.get_state_history(config):
        snapshot_config = dict(snapshot.config or {})
        configurable = dict(snapshot_config.get("configurable", {}) or {})
        if configurable.get("checkpoint_id") == checkpoint_id:
            return snapshot
    raise ValueError(f"Unknown checkpoint: {checkpoint_id}")


def replay_checkpoint(graph: Any, checkpoint: Any) -> dict[str, Any]:
    """Replay execution from the selected persisted checkpoint."""
    result = graph.invoke(None, checkpoint.config)
    return dict(result)


def fork_checkpoint(
    graph: Any,
    checkpoint: Any,
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a new checkpoint branch with explicit state updates, then continue it."""
    if not values:
        raise ValueError("Fork requires at least one explicit state override")
    fork_config = graph.update_state(checkpoint.config, values=values)
    result = graph.invoke(None, fork_config)
    return dict(fork_config), dict(result)


def _finalized(state: dict[str, Any]) -> bool:
    return any(event.get("node") == "finalize" for event in state.get("events", []) or [])


def verify_time_travel(database_url: str) -> TimeTravelEvidence:
    """Exercise replay and fork semantics against a fresh SQLite-backed demo thread."""
    graph = build_time_travel_demo_graph(build_checkpointer("sqlite", database_url))
    thread_id = f"bonus-time-travel-{uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"topic": "baseline", "rendered": "", "events": []}, config=config)

    history = list_checkpoints(graph, thread_id)
    selectable = next((item for item in history if item.next_nodes == ("render",)), None)
    if selectable is None:
        raise RuntimeError("No replayable checkpoint before render")
    selected = find_checkpoint(graph, thread_id, selectable.checkpoint_id)

    replayed = replay_checkpoint(graph, selected)
    replay_verified = replayed.get("rendered") == "render:baseline" and _finalized(replayed)

    before_fork = {item.checkpoint_id for item in list_checkpoints(graph, thread_id)}
    fork_config, forked = fork_checkpoint(graph, selected, {"topic": "forked"})
    after_fork = {item.checkpoint_id for item in list_checkpoints(graph, thread_id)}
    fork_checkpoint_id = str(dict(fork_config.get("configurable", {}) or {}).get("checkpoint_id", ""))
    history_preserved = before_fork <= after_fork and selectable.checkpoint_id in after_fork
    fork_verified = (
        bool(fork_checkpoint_id)
        and fork_checkpoint_id not in before_fork
        and forked.get("rendered") == "render:forked"
        and _finalized(forked)
    )

    verified = replay_verified and fork_verified and history_preserved
    return TimeTravelEvidence(
        implemented=True,
        verified=verified,
        replay_verified=replay_verified,
        fork_verified=fork_verified,
        original_history_preserved=history_preserved,
        checkpoint_id=selectable.checkpoint_id,
    )
