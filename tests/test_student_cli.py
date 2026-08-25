from pathlib import Path

import langgraph_agent_lab.cli as cli
from langgraph_agent_lab.metrics import MetricsReport, metric_from_state, summarize_metrics
from langgraph_agent_lab.state import Route, Scenario


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


def test_run_scenarios_records_measured_latency(tmp_path: Path, monkeypatch) -> None:
    scenario = Scenario(id="synthetic", query="How do I reset it?", expected_route=Route.SIMPLE)

    class FakeGraph:
        def invoke(self, state: dict, config: dict) -> dict:
            return {
                **state,
                "route": "simple",
                "final_answer": "ok",
                "events": [
                    {
                        "node": "finalize",
                        "event_type": "completed",
                        "message": "done",
                        "latency_ms": 0,
                        "metadata": {},
                    }
                ],
            }

    captured: list[MetricsReport] = []
    monkeypatch.setattr(cli, "load_scenarios", lambda _path: [scenario])
    monkeypatch.setattr(cli, "build_checkpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "build_graph", lambda checkpointer=None: FakeGraph())
    monkeypatch.setattr(cli, "write_metrics", lambda report, _path: captured.append(report))
    monkeypatch.setattr(cli, "perf_counter", iter([1.0, 1.025]).__next__, raising=False)

    config = tmp_path / "lab.yaml"
    config.write_text("scenarios_path: ignored.jsonl\ncheckpointer: memory\n", encoding="utf-8")
    cli.run_scenarios(config=config, output=tmp_path / "metrics.json")

    assert captured
    assert captured[0].scenario_metrics[0].latency_ms == 25
