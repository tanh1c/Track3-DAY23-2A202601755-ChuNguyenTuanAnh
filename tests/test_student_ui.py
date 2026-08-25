import json

from langgraph_agent_lab.ui import build_view_model, verify_ui_view_model


REQUIRED_KEYS = {
    "thread_id",
    "checkpoint_id",
    "query",
    "route",
    "risk_level",
    "proposed_action",
    "approval",
    "final_answer",
    "events",
}


def test_view_model_contains_only_presentation_fields(monkeypatch) -> None:
    openai_sentinel = "openai-secret-sentinel-123456789"
    gemini_sentinel = "gemini-secret-sentinel-123456789"
    anthropic_sentinel = "anthropic-secret-sentinel-123456789"
    monkeypatch.setenv("OPENAI_API_KEY", openai_sentinel)
    monkeypatch.setenv("GEMINI_API_KEY", gemini_sentinel)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_sentinel)
    state = {
        "thread_id": "thread-ui",
        "query": "refund order",
        "route": "risky",
        "risk_level": "high",
        "proposed_action": "refund order",
        "approval": {"approved": True, "reviewer": "alice", "comment": "ok"},
        "final_answer": "completed",
        "events": [
            {"node": "finalize", "event_type": "completed", "message": "done"}
        ],
    }

    view = build_view_model(state, checkpoint_id="cp-ui")
    assert set(view) == REQUIRED_KEYS
    rendered = json.dumps(view, sort_keys=True)
    assert openai_sentinel not in rendered
    assert gemini_sentinel not in rendered
    assert anthropic_sentinel not in rendered


def test_ui_verifier_reports_view_model_and_secret_safety(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-sentinel-another-value")
    evidence = verify_ui_view_model()
    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.view_model_verified is True
    assert evidence.secret_safe is True
