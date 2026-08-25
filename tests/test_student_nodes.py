from copy import deepcopy

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.nodes import (
    approval_node,
    ask_clarification_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)


def test_tool_appends_only_new_result_without_mutating_input() -> None:
    state = {
        "route": "error",
        "attempt": 1,
        "query": "service failed",
        "tool_results": ["old"],
    }
    before = deepcopy(state)
    update = tool_node(state)
    assert state == before
    assert len(update["tool_results"]) == 1
    assert "ERROR" in update["tool_results"][0]


def test_error_tool_succeeds_after_transient_threshold() -> None:
    update = tool_node({"route": "error", "attempt": 2, "query": "service failed"})
    assert "ERROR" not in update["tool_results"][0]


def test_risky_tool_fails_closed_without_approval() -> None:
    update = tool_node({"route": "risky", "attempt": 0, "query": "refund customer"})
    assert "ERROR" in update["tool_results"][0]
    assert "approval" in update["tool_results"][0].lower()


def test_retry_is_only_counter_owner_and_does_not_mutate_input() -> None:
    state = {"attempt": 0, "max_attempts": 1, "errors": []}
    before = deepcopy(state)
    update = retry_or_fallback_node(state)
    assert state == before
    assert update["attempt"] == 1
    assert len(update["errors"]) == 1


def test_mock_approval_is_non_interactive_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_INTERRUPT", raising=False)
    update = approval_node({"proposed_action": "refund order"})
    assert update["approval"]["approved"] is True
    assert update["approval"]["reviewer"] == "mock-reviewer"


def test_real_hitl_mode_uses_patchable_interrupt(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    observed: dict[str, object] = {}

    def fake_interrupt(payload: dict[str, object]) -> dict[str, object]:
        observed.update(payload)
        return {
            "approved": False,
            "reviewer": "human",
            "comment": "choose another action",
        }

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt, raising=False)
    update = approval_node({"proposed_action": "delete account"})
    assert observed["proposed_action"] == "delete account"
    assert update["approval"]["approved"] is False
    assert update["events"][0]["event_type"] == "resumed"
    assert update["events"][0]["metadata"]["real_interrupt"] is True


def test_evaluator_heuristic_detects_error_without_api_key(monkeypatch) -> None:
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    update = evaluate_node({"tool_results": ["ERROR: transient failure"]})
    assert update["evaluation_result"] == "needs_retry"


def test_evaluator_heuristic_accepts_success_without_api_key(monkeypatch) -> None:
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    update = evaluate_node({"tool_results": ["SUCCESS: order found"]})
    assert update["evaluation_result"] == "success"


def test_terminal_nodes_do_not_overwrite_classified_route() -> None:
    assert "route" not in dead_letter_node({"attempt": 2, "max_attempts": 2})
    assert "route" not in finalize_node({"final_answer": "done"})


def test_clarification_and_risky_action_produce_current_values() -> None:
    clarification = ask_clarification_node({"query": "fix it"})
    risky = risky_action_node({"query": "refund this customer"})
    assert clarification["pending_question"]
    assert clarification["final_answer"] == clarification["pending_question"]
    assert risky["proposed_action"]
