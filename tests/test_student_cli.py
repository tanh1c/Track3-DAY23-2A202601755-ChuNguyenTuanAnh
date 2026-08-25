from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics


def test_metric_uses_measured_latency_and_real_interrupt_event() -> None:
    state = {
        "scenario_id": "synthetic",
        "route": "risky",
        "final_answer": "done",
        "approval": {"approved": True},
        "events": [
            {"node": "approval", "event_type": "completed", "metadata": {}},
            {"node": "approval", "event_type": "interrupted", "metadata": {}},
            {"node": "finalize", "event_type": "completed", "metadata": {}},
        ],
        "errors": [],
    }
    metric = metric_from_state(state, "risky", True, latency_ms=37)
    assert metric.latency_ms == 37
    assert metric.interrupt_count == 1


def test_ordinary_approval_visit_is_not_an_interrupt() -> None:
    state = {
        "scenario_id": "synthetic",
        "route": "risky",
        "final_answer": "done",
        "approval": {"approved": True},
        "events": [{"node": "approval", "event_type": "completed", "metadata": {}}],
        "errors": [],
    }
    assert metric_from_state(state, "risky", True).interrupt_count == 0


def test_resume_success_requires_explicit_evidence() -> None:
    metric = metric_from_state(
        {
            "scenario_id": "x",
            "route": "simple",
            "final_answer": "ok",
            "events": [],
            "errors": [],
        },
        "simple",
        False,
    )
    assert summarize_metrics([metric]).resume_success is False
    assert summarize_metrics([metric], resume_success=True).resume_success is True
