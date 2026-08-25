from langgraph.types import Send

from langgraph_agent_lab.bonus_parallel import (
    build_parallel_bonus_graph,
    plan_tasks,
    verify_parallel_send,
)


def test_plan_tasks_returns_actual_send_objects() -> None:
    routes = plan_tasks({"tasks": ["account", "order", "policy"], "results": []})
    assert len(routes) == 3
    assert all(isinstance(route, Send) for route in routes)


def test_parallel_bonus_graph_fans_out_and_aggregates_deterministically() -> None:
    final = build_parallel_bonus_graph().invoke(
        {"tasks": ["order", "policy", "account"], "results": []}
    )
    assert final["aggregate"] == "account|order|policy"
    assert sorted(final["results"]) == ["account", "order", "policy"]


def test_parallel_send_verifier_records_runtime_evidence() -> None:
    evidence = verify_parallel_send(["order", "policy", "account"])
    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.used_send is True
    assert evidence.task_count == 3
    assert evidence.result_count == 3
    assert evidence.aggregate_deterministic is True
