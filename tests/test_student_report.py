from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report


def _sample_report() -> MetricsReport:
    return MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=1,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="synthetic",
                success=True,
                expected_route="error",
                actual_route="error",
                nodes_visited=4,
                retry_count=1,
                interrupt_count=0,
                approval_required=False,
                approval_observed=False,
                latency_ms=25,
                errors=["transient failure"],
            )
        ],
    )


def test_report_contains_required_evidence_sections() -> None:
    text = render_report(_sample_report())
    for heading in (
        "Architecture",
        "State Schema",
        "Scenario Results",
        "Failure Analysis",
        "Persistence and Recovery",
        "Extension Work",
        "Improvement Plan",
    ):
        assert heading in text
    assert "synthetic" in text
    assert "25" in text
    assert "resume_success" in text


def test_report_contains_student_repo_commit_and_date(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "abc123def456")
    text = render_report(_sample_report())
    assert "Chu Nguyen Tuan Anh" in text
    assert "tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh" in text
    assert "abc123def456" in text
    assert "Report date" in text


def test_report_does_not_claim_real_hitl_when_no_interrupts() -> None:
    report = MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=2.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=False,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="simple",
                success=True,
                expected_route="simple",
                actual_route="simple",
                nodes_visited=2,
            )
        ],
    )
    text = render_report(report).lower()
    assert "no real hitl interrupt was observed" in text
