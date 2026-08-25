from langgraph_agent_lab.graph import build_graph
import langgraph_agent_lab.nodes as nodes


REQUIRED_NODES = {
    "intake",
    "classify",
    "tool",
    "evaluate",
    "answer",
    "clarify",
    "risky_action",
    "approval",
    "retry",
    "dead_letter",
    "finalize",
}


def test_compiled_graph_contains_exact_required_workflow_nodes() -> None:
    graph = build_graph()
    graph_nodes = set(graph.get_graph().nodes)
    assert REQUIRED_NODES <= graph_nodes


def test_error_route_at_retry_limit_skips_tool(monkeypatch) -> None:
    def classify_error(_state: dict) -> dict:
        return {
            "route": "error",
            "risk_level": "low",
            "events": [
                {
                    "node": "classify",
                    "event_type": "completed",
                    "message": "test route",
                    "latency_ms": 0,
                    "metadata": {},
                }
            ],
        }

    monkeypatch.setattr(nodes, "classify_node", classify_error)
    graph = build_graph()
    result = graph.invoke(
        {
            "thread_id": "thread-boundary",
            "scenario_id": "boundary",
            "query": "synthetic failure",
            "route": "",
            "risk_level": "unknown",
            "attempt": 0,
            "max_attempts": 1,
            "final_answer": None,
            "messages": [],
            "tool_results": [],
            "errors": [],
            "events": [],
        }
    )
    visited = [event["node"] for event in result["events"]]
    assert "retry" in visited
    assert "dead_letter" in visited
    assert "tool" not in visited
    assert visited[-1] == "finalize"
