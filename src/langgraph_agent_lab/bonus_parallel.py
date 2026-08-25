"""Isolated LangGraph map-reduce bonus graph using the real Send API."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .bonus_evidence import ParallelSendEvidence


class ParallelState(TypedDict, total=False):
    """State for the isolated fan-out demonstration."""

    tasks: list[str]
    results: Annotated[list[str], add]
    aggregate: str


class TaskState(TypedDict):
    """Per-worker payload dispatched with Send."""

    task: str


def dispatch_node(state: ParallelState) -> dict[str, object]:
    """Anchor the conditional fan-out without mutating state."""
    return {}


def plan_tasks(state: ParallelState) -> list[Send]:
    """Map each independent task to the worker using LangGraph Send."""
    return [Send("run_task", {"task": task}) for task in state.get("tasks", [])]


def run_task(state: TaskState) -> dict[str, list[str]]:
    """Return one deterministic reducer contribution for a dispatched task."""
    return {"results": [state["task"]]}


def aggregate_results(state: ParallelState) -> dict[str, str]:
    """Create an order-independent aggregate for grading evidence."""
    return {"aggregate": "|".join(sorted(state.get("results", [])))}


def build_parallel_bonus_graph() -> Any:
    """Compile the separate three-node Send fan-out graph."""
    builder = StateGraph(ParallelState)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("run_task", run_task)
    builder.add_node("aggregate", aggregate_results)
    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", plan_tasks, ["run_task"])
    builder.add_edge("run_task", "aggregate")
    builder.add_edge("aggregate", END)
    return builder.compile()


def verify_parallel_send(tasks: list[str] | None = None) -> ParallelSendEvidence:
    """Run fan-out and derive evidence from actual Send objects and reducer output."""
    requested = list(tasks or ["account", "order", "policy"])
    sends = plan_tasks({"tasks": requested, "results": []})
    used_send = len(sends) == len(requested) and all(isinstance(item, Send) for item in sends)
    final = build_parallel_bonus_graph().invoke({"tasks": requested, "results": []})
    results = list(final.get("results", []) or [])
    expected = sorted(requested)
    deterministic = sorted(results) == expected and final.get("aggregate") == "|".join(expected)
    verified = used_send and len(requested) > 1 and len(results) == len(requested) and deterministic

    return ParallelSendEvidence(
        implemented=True,
        verified=verified,
        used_send=used_send,
        task_count=len(requested),
        result_count=len(results),
        aggregate_deterministic=deterministic,
    )
