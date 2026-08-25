"""Optional Streamlit presentation layer over the existing agent state contract."""

from __future__ import annotations

import json
import os

from langgraph_agent_lab.bonus_evidence import UiEvidence
from langgraph_agent_lab.state import AgentState

_PRESENTATION_KEYS = (
    "thread_id",
    "checkpoint_id",
    "query",
    "route",
    "risk_level",
    "proposed_action",
    "approval",
    "final_answer",
    "events",
)
_PROVIDER_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")
_SECRET_SENTINELS = {
    "OPENAI_API_KEY": "openai-ui-secret-sentinel",
    "GEMINI_API_KEY": "gemini-ui-secret-sentinel",
    "ANTHROPIC_API_KEY": "anthropic-ui-secret-sentinel",
}


def build_view_model(
    state: AgentState,
    *,
    checkpoint_id: str = "",
) -> dict[str, object]:
    """Project agent state into a deliberately small, secret-agnostic UI model."""
    return {
        "thread_id": str(state.get("thread_id", "")),
        "checkpoint_id": checkpoint_id,
        "query": str(state.get("query", "")),
        "route": str(state.get("route", "")),
        "risk_level": str(state.get("risk_level", "")),
        "proposed_action": str(state.get("proposed_action", "")),
        "approval": dict(state.get("approval") or {}),
        "final_answer": str(state.get("final_answer") or ""),
        "events": list(state.get("events") or []),
    }


def verify_ui_view_model() -> UiEvidence:
    """Verify required display fields and non-vacuous environment secret isolation."""
    state: AgentState = {
        "thread_id": "bonus-ui-thread",
        "query": "refund order",
        "route": "risky",
        "risk_level": "high",
        "proposed_action": "refund order after approval",
        "approval": {
            "approved": True,
            "reviewer": "ui-reviewer",
            "comment": "ok",
        },
        "final_answer": "completed",
        "events": [
            {
                "node": "finalize",
                "event_type": "completed",
                "message": "done",
                "metadata": {},
            }
        ],
    }
    previous = {name: os.environ.get(name) for name in _PROVIDER_VARS}
    try:
        for name, sentinel in _SECRET_SENTINELS.items():
            os.environ[name] = sentinel
        view = build_view_model(state, checkpoint_id="bonus-ui-checkpoint")
        serialized = json.dumps(view, sort_keys=True, default=str)
        secret_safe = not any(
            sentinel in serialized for sentinel in _SECRET_SENTINELS.values()
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    view_model_verified = tuple(view) == _PRESENTATION_KEYS
    verified = view_model_verified and secret_safe
    return UiEvidence(
        implemented=True,
        verified=verified,
        view_model_verified=view_model_verified,
        secret_safe=secret_safe,
    )


def main() -> None:
    """Run the optional grading-oriented Streamlit UI."""
    import streamlit as st

    st.set_page_config(page_title="Day 23 LangGraph Evidence", layout="wide")
    st.title("Day 23 LangGraph Evidence UI")
    st.caption(
        "Presentation-only view of the existing state contract; "
        "no provider secrets shown."
    )

    query = st.text_input("Ticket", value="Please refund order 42")
    route = str(
        st.selectbox(
            "Route",
            ["simple", "tool", "missing_info", "risky", "error"],
        )
    )
    risk_level = "high" if route == "risky" else "low"
    proposed_action = st.text_input(
        "Proposed action",
        value="Refund order after explicit approval" if route == "risky" else "",
    )
    approved = st.checkbox("Approved", value=route == "risky")
    reviewer = st.text_input("Reviewer", value="demo-reviewer")
    comment = st.text_input("Approval comment", value="reviewed in UI")
    final_answer = st.text_area("Final answer", value="Evidence preview")

    state: AgentState = {
        "thread_id": "ui-preview-thread",
        "query": query,
        "route": route,
        "risk_level": risk_level,
        "proposed_action": proposed_action,
        "approval": {
            "approved": approved,
            "reviewer": reviewer,
            "comment": comment,
        },
        "final_answer": final_answer,
        "events": [
            {
                "node": "finalize",
                "event_type": "completed",
                "message": "preview event",
                "metadata": {},
            }
        ],
    }
    st.subheader("State evidence")
    st.json(build_view_model(state, checkpoint_id="ui-preview-checkpoint"))


if __name__ == "__main__":
    main()
