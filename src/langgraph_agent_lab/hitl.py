"""Executable real human-in-the-loop interrupt/resume verification."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from .bonus_evidence import HitlEvidence
from .graph import build_graph
from .persistence import build_checkpointer
from .state import Route, Scenario, initial_state

_PROVIDER_VARS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_MODEL",
)


def _event_nodes(state: dict[str, Any]) -> list[str]:
    return [str(event.get("node", "")) for event in state.get("events", []) or []]


def _run_review_case(
    database_url: str,
    thread_id: str,
    *,
    approved: bool,
    reviewer: str,
    comment: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Interrupt once, resume with a reviewer decision, and return observed state facts."""
    checkpointer = build_checkpointer("sqlite", database_url)
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id=thread_id,
        query="Please refund order 42 and send confirmation",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    state["thread_id"] = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    initial = graph.invoke(state, config=config)
    interrupts = list(initial.get("__interrupt__", []) or [])
    if not interrupts:
        raise RuntimeError("Expected approval interrupt")

    payload = getattr(interrupts[0], "value", None)
    if not isinstance(payload, dict) or payload.get("type") != "approval_required":
        raise RuntimeError("Unexpected approval interrupt payload")

    resumed = graph.invoke(
        Command(
            resume={
                "approved": approved,
                "reviewer": reviewer,
                "comment": comment,
            }
        ),
        config=config,
    )
    final_state = dict(resumed)
    snapshot = graph.get_state(config)
    snapshot_config = dict(snapshot.config or {})
    configurable = dict(snapshot_config.get("configurable", {}) or {})
    same_thread = configurable.get("thread_id") == thread_id
    return final_state, True, same_thread


def verify_hitl_round_trip(
    database_url: str,
    *,
    thread_id: str = "bonus-hitl-approved",
) -> HitlEvidence:
    """Prove real approval and rejection resumes without making any provider API call."""
    previous_interrupt = os.environ.get("LANGGRAPH_INTERRUPT")
    previous_provider = {name: os.environ.get(name) for name in _PROVIDER_VARS}
    suffix = uuid4().hex[:10]
    approved_thread = f"{thread_id}-{suffix}-approved"
    rejected_thread = f"{thread_id}-{suffix}-rejected"

    try:
        os.environ["LANGGRAPH_INTERRUPT"] = "true"
        for name in _PROVIDER_VARS:
            os.environ[name] = ""

        approved_state, approved_interrupt, approved_same_thread = _run_review_case(
            database_url,
            approved_thread,
            approved=True,
            reviewer="ci-reviewer",
            comment="approved by automated reviewer fixture",
        )
        rejected_state, rejected_interrupt, rejected_same_thread = _run_review_case(
            database_url,
            rejected_thread,
            approved=False,
            reviewer="ci-reviewer",
            comment="rejected by automated reviewer fixture",
        )
    finally:
        if previous_interrupt is None:
            os.environ.pop("LANGGRAPH_INTERRUPT", None)
        else:
            os.environ["LANGGRAPH_INTERRUPT"] = previous_interrupt
        for name, value in previous_provider.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    approved_nodes = _event_nodes(approved_state)
    rejected_nodes = _event_nodes(rejected_state)
    approval_events = [
        event
        for event in approved_state.get("events", []) or []
        if event.get("node") == "approval"
    ]
    approval = dict(approved_state.get("approval") or {})

    real_interrupt_recorded = bool(approval_events) and any(
        bool(dict(event.get("metadata", {}) or {}).get("real_interrupt"))
        for event in approval_events
    )
    approved_order = all(
        node in approved_nodes for node in ("risky_action", "approval", "tool")
    )
    if approved_order:
        approved_order = (
            approved_nodes.index("risky_action")
            < approved_nodes.index("approval")
            < approved_nodes.index("tool")
        )
    approved_finalized = "finalize" in approved_nodes

    rejected_approval = dict(rejected_state.get("approval") or {})
    rejection_verified = (
        rejected_interrupt
        and rejected_same_thread
        and rejected_approval.get("approved") is False
        and "approval" in rejected_nodes
        and "clarify" in rejected_nodes
        and "tool" not in rejected_nodes
        and "finalize" in rejected_nodes
    )

    resume_success = (
        approved_interrupt
        and approved_same_thread
        and real_interrupt_recorded
        and approved_order
        and approved_finalized
        and approval.get("approved") is True
        and approval.get("reviewer") == "ci-reviewer"
    )
    verified = resume_success and rejection_verified

    return HitlEvidence(
        implemented=True,
        verified=verified,
        interrupt_observed=approved_interrupt and rejected_interrupt,
        same_thread_id=approved_same_thread and rejected_same_thread,
        resume_success=resume_success,
        rejection_verified=rejection_verified,
        reviewer=str(approval.get("reviewer", "")),
    )
