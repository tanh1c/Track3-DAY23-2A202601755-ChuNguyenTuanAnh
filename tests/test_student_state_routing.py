from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from langgraph_agent_lab.state import AgentState


def test_required_current_value_fields_are_declared() -> None:
    annotations = AgentState.__annotations__
    assert "evaluation_result" in annotations
    assert "pending_question" in annotations
    assert "proposed_action" in annotations
    assert "approval" in annotations


def test_classify_routing_table() -> None:
    assert route_after_classify({"route": "simple"}) == "answer"
    assert route_after_classify({"route": "tool"}) == "tool"
    assert route_after_classify({"route": "missing_info"}) == "clarify"
    assert route_after_classify({"route": "risky"}) == "risky_action"
    assert route_after_classify({"route": "error"}) == "retry"
    assert route_after_classify({"route": "unknown"}) == "answer"


def test_retry_boundary_is_fail_closed() -> None:
    assert route_after_retry({"attempt": 0, "max_attempts": 1}) == "tool"
    assert route_after_retry({"attempt": 1, "max_attempts": 1}) == "dead_letter"
    assert route_after_retry({"attempt": 2, "max_attempts": 1}) == "dead_letter"


def test_evaluate_and_approval_routing() -> None:
    assert route_after_evaluate({"evaluation_result": "needs_retry"}) == "retry"
    assert route_after_evaluate({"evaluation_result": "success"}) == "answer"
    assert route_after_approval({"approval": {"approved": True}}) == "tool"
    assert route_after_approval({"approval": {"approved": False}}) == "clarify"
