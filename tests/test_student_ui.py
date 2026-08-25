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
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value-not-for-display")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value-not-for-display")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-value-not-for-display")
    state = {
        "thread_id": "thread-ui",
        "query": "refund order",
        "route": "risky",
        "risk_level": "high",
        "proposed_action": "refund order",
        "approval": {"approved": True, "reviewer": "alice", "comment": "ok"},
        "final_answer": "completed",
        "events": [{"node": "finalize", "event_type": "completed", "message": "done"}],
    }

    view = build_view_model(state, checkpoint_id="cp-ui")
    assert set(view) == REQUIRED_KEYS
    rendered = json.dumps(view, sort_keys=True)
    assert "sk-test-secret-value-not-for-display" not in rendered
    assert "AIza-test-secret-value-not-for-display" not in rendered
    assert "sk-ant-test-secret-value-not-for-display" not in rendered


def test_ui_verifier_reports_view_model_and_secret_safety(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-another-secret-value-not-for-display")
    evidence = verify_ui_view_model()
    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.view_model_verified is True
    assert evidence.secret_safe is True
