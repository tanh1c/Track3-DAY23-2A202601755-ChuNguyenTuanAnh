"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update. Reducer-managed
lists contain only newly appended entries; input state is never mutated.
"""

from __future__ import annotations

import os
from typing import Any

from .state import AgentState, ApprovalDecision, make_event


def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query using the starter partial-update contract."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify intent using a real structured-output LLM (implemented in next TDD task)."""
    raise NotImplementedError("TODO(student): implement LLM-based classification")


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute the deterministic mock tool while enforcing approval safety."""
    route = str(state.get("route", ""))
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "").strip()
    approval = state.get("approval") or {}

    if route == "risky" and not bool(approval.get("approved")):
        result = "ERROR: risky action blocked because approval is missing or rejected"
        event_type = "blocked"
    elif route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt}"
        event_type = "failed"
    elif route == "risky":
        action = state.get("proposed_action", query or "requested action")
        result = f"SUCCESS: approved mock action completed: {action}"
        event_type = "completed"
    else:
        result = f"SUCCESS: mock tool result for: {query or 'support request'}"
        event_type = "completed"

    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Base evaluator: deterministic latest-result heuristic."""
    results = state.get("tool_results") or []
    latest = str(results[-1]) if results else "ERROR: no tool result available"
    verdict = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": verdict,
        "events": [make_event("evaluate", "completed", f"verdict={verdict}", mode="heuristic")],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded final answer using a real LLM (implemented in next TDD task)."""
    raise NotImplementedError("TODO(student): implement LLM-grounded answer generation")


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for the missing detail rather than inventing one."""
    query = state.get("query", "").strip()
    approval = state.get("approval") or {}
    if approval and not bool(approval.get("approved")):
        question = (
            "The proposed action was not approved. What safer alternative or additional "
            "instruction would you like me to use?"
        )
    else:
        excerpt = query or "your request"
        question = f"Could you provide the missing details needed to handle '{excerpt}' safely?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare, but do not execute, a side-effecting action."""
    query = state.get("query", "").strip()
    proposed = f"Perform the requested side effect after explicit approval: {query}"
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "prepared", "risky action awaiting approval")],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Use mock approval by default; optionally interrupt for real HITL review."""
    proposed_action = state.get("proposed_action", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        resumed = interrupt(
            {
                "type": "approval_required",
                "proposed_action": proposed_action,
                "instruction": "Resume with approved/reviewer/comment fields.",
            }
        )
        if isinstance(resumed, bool):
            decision = ApprovalDecision(approved=resumed, reviewer="human-reviewer")
        else:
            decision = ApprovalDecision.model_validate(resumed)
        event_type = "resumed"
        interrupt_used = True
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="automatic mock approval for non-interactive lab execution",
        )
        event_type = "completed"
        interrupt_used = False

    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                event_type,
                "approval decision recorded",
                approved=decision.approved,
                real_interrupt=interrupt_used,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Own the bounded retry counter and append one failure record."""
    old_attempt = int(state.get("attempt", 0))
    new_attempt = old_attempt + 1
    max_attempts = int(state.get("max_attempts", 0))
    error = f"retry recorded: attempt {new_attempt}/{max_attempts}"
    return {
        "attempt": new_attempt,
        "errors": [error],
        "events": [
            make_event(
                "retry",
                "recorded",
                error,
                attempt=new_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Terminate exhausted failures with an explicit escalation response."""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 0))
    answer = (
        f"The request could not be completed after {attempt} retry attempt(s) "
        f"with a configured limit of {max_attempts}. Escalation is required."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "exhausted", answer)],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit the single completion audit event required on every terminal path."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
