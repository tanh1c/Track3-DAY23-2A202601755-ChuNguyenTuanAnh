"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next registered graph node."""
    return {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }.get(str(state.get("route", "")), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Route failed tool evaluations back to retry, otherwise answer."""
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    """Bound retries using the post-increment attempt value."""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 0))
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Proceed only when the serialized approval decision is affirmative."""
    approval = state.get("approval") or {}
    return "tool" if bool(approval.get("approved")) else "clarify"
