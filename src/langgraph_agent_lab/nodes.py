"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update. Reducer-managed
lists contain only newly appended entries; input state is never mutated.
"""

from __future__ import annotations

import os
from typing import Any

from .llm import configured_provider, get_llm
from .schemas import ClassificationDecision, EvaluationDecision
from .state import AgentState, ApprovalDecision, make_event


CLASSIFICATION_PROMPT = """You route support tickets for a LangGraph workflow.
Return one route using the structured schema. Use semantic intent, never scenario IDs.
Apply this priority when multiple intents appear: risky > tool > missing_info > error > simple.

Definitions:
- risky: requests that cause side effects or mutations, e.g. refund, delete, send, update,
  cancel, charge, or change an account/order; these require approval.
- tool: lookup/retrieval/status requests that need an external tool but do not mutate data.
- missing_info: too vague/incomplete to act safely or answer specifically.
- error: the ticket primarily reports a timeout, failure, exception, outage, or processing error.
- simple: informational/how-to support that can be answered directly.

Ticket:
{query}
"""


def _message_text(response: Any) -> str:
    """Normalize common LangChain response shapes to non-empty text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


def _fallback_route(query: str) -> str:
    """Conservative auditable fallback used only when structured LLM routing fails."""
    text = query.lower()
    risky = (
        "refund",
        "delete",
        "remove account",
        "send email",
        "send confirmation",
        "cancel",
        "charge",
        "change plan",
        "update account",
        "reset another",
    )
    tool = ("lookup", "look up", "order status", "track order", "find order", "check status")
    missing = ("fix it", "can you fix", "help with it", "not working", "do this")
    error = ("timeout", "failure", "failed", "exception", "error", "outage", "cannot recover")
    if any(token in text for token in risky):
        return "risky"
    if any(token in text for token in tool):
        return "tool"
    if any(token in text for token in missing) or len(text.split()) <= 3:
        return "missing_info"
    if any(token in text for token in error):
        return "error"
    return "simple"


def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query using the starter partial-update contract."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify intent through real structured-output LLM routing."""
    query = state.get("query", "").strip()
    try:
        decision = (
            get_llm(temperature=0.0)
            .with_structured_output(ClassificationDecision)
            .invoke(CLASSIFICATION_PROMPT.format(query=query))
        )
        if not isinstance(decision, ClassificationDecision):
            decision = ClassificationDecision.model_validate(decision)
        route = decision.route
        reason = decision.reason
        event_type = "completed"
        errors: list[str] = []
    except Exception as exc:  # provider/schema failures must terminate predictably
        route = _fallback_route(query)
        reason = f"structured classifier fallback: {type(exc).__name__}"
        event_type = "fallback"
        errors = [reason]

    update: dict[str, Any] = {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "events": [make_event("classify", event_type, f"route={route}", reason=reason)],
    }
    if errors:
        update["errors"] = errors
    return update


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
    """Evaluate the latest tool result; live runs use one structured judge call."""
    results = state.get("tool_results") or []
    latest = str(results[-1]) if results else "ERROR: no tool result available"
    heuristic = "needs_retry" if "ERROR" in latest.upper() else "success"

    if configured_provider() is None or os.getenv("LLM_JUDGE", "true").lower() == "false":
        return {
            "evaluation_result": heuristic,
            "events": [
                make_event("evaluate", "completed", f"verdict={heuristic}", mode="heuristic")
            ],
        }

    prompt = (
        "Judge exactly one tool result. Return needs_retry only when the result indicates "
        "failure, incomplete execution, or unusable evidence; otherwise return success.\n\n"
        f"Tool result:\n{latest}"
    )
    try:
        decision = (
            get_llm(temperature=0.0)
            .with_structured_output(EvaluationDecision)
            .invoke(prompt)
        )
        if not isinstance(decision, EvaluationDecision):
            decision = EvaluationDecision.model_validate(decision)
        return {
            "evaluation_result": decision.verdict,
            "events": [
                make_event(
                    "evaluate",
                    "completed",
                    f"verdict={decision.verdict}",
                    mode="llm-as-judge",
                    reason=decision.reason,
                )
            ],
        }
    except Exception as exc:
        error = f"judge fallback: {type(exc).__name__}"
        return {
            "evaluation_result": heuristic,
            "errors": [error],
            "events": [
                make_event(
                    "evaluate",
                    "fallback",
                    f"verdict={heuristic}",
                    mode="heuristic",
                    error=error,
                )
            ],
        }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded final answer using the configured real LLM."""
    query = state.get("query", "").strip()
    tool_results = state.get("tool_results") or []
    proposed_action = state.get("proposed_action", "")
    approval = state.get("approval") or {}
    context = (
        f"User query: {query}\n"
        f"Tool results: {tool_results}\n"
        f"Proposed action: {proposed_action or 'none'}\n"
        f"Approval: {approval or 'none'}\n"
    )
    prompt = (
        "You are a concise support assistant. Answer only from the supplied workflow context. "
        "Do not claim an action happened unless a successful tool result proves it. If evidence is "
        "limited, state that limitation.\n\n" + context
    )
    try:
        text = _message_text(get_llm(temperature=0.0).invoke(prompt))
        if not text:
            raise ValueError("empty LLM response")
        return {
            "final_answer": text,
            "events": [make_event("answer", "completed", "grounded LLM answer generated")],
        }
    except Exception as exc:
        error = f"answer fallback: {type(exc).__name__}"
        if tool_results:
            fallback = f"I could not generate the model response. Latest verified result: {tool_results[-1]}"
        else:
            fallback = "I could not generate the model response from the available context."
        return {
            "final_answer": fallback,
            "errors": [error],
            "events": [make_event("answer", "fallback", error)],
        }


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

    approval = decision.model_dump()
    return {
        "approval": approval,
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
