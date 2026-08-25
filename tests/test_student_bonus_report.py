from langgraph_agent_lab.bonus_evidence import (
    BonusEvidence,
    HitlEvidence,
    ParallelSendEvidence,
    RecoveryEvidence,
    TimeTravelEvidence,
    UiEvidence,
)
from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report


def _metrics() -> MetricsReport:
    return MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="bonus-report",
                success=True,
                expected_route="simple",
                actual_route="simple",
                nodes_visited=4,
                latency_ms=10,
            )
        ],
    )


def _bonus() -> BonusEvidence:
    return BonusEvidence(
        llm_as_judge_verified=True,
        durable_recovery_verified=True,
        mermaid_export_verified=True,
        hitl=HitlEvidence(
            implemented=True,
            verified=True,
            interrupt_observed=True,
            same_thread_id=True,
            resume_success=True,
            rejection_verified=True,
            reviewer="ci-reviewer",
        ),
        recovery=RecoveryEvidence(
            implemented=True,
            verified=True,
            writer_pid=101,
            reader_pid=202,
            distinct_processes=True,
            same_thread_id=True,
            persisted_finalized=True,
            thread_id="recovery-thread",
        ),
        time_travel=TimeTravelEvidence(
            implemented=True,
            verified=True,
            replay_verified=True,
            fork_verified=True,
            original_history_preserved=True,
            checkpoint_id="cp-1",
        ),
        parallel_send=ParallelSendEvidence(
            implemented=True,
            verified=True,
            used_send=True,
            task_count=3,
            result_count=3,
            aggregate_deterministic=True,
        ),
        streamlit_ui=UiEvidence(
            implemented=True,
            verified=True,
            view_model_verified=True,
            secret_safe=True,
        ),
    )


def test_report_renders_all_official_extensions_from_evidence() -> None:
    text = render_report(_metrics(), bonus=_bonus())
    for extension in (
        "LLM-as-judge",
        "Real HITL",
        "SQLite recovery",
        "Time travel",
        "Parallel Send",
        "Streamlit UI",
        "Mermaid export",
    ):
        assert extension in text
    for heading in (
        "Baseline",
        "Implementation",
        "Verification",
        "Evidence",
        "Limitations",
    ):
        assert heading in text
    assert "timeout=20s" in text
    assert "max_retries=0" in text
    assert "writer PID 101 -> reader PID 202" in text
    assert "interrupt + same-thread" in text
    assert "replay + fork" in text
    assert "3 tasks -> 3 reducer results" in text
    assert "streamlit run src/langgraph_agent_lab/ui.py" in text
    assert "presentation smoke, not browser E2E" in text


def test_report_remains_backward_compatible_without_bonus() -> None:
    text = render_report(_metrics())
    assert "Extension Work" in text
    assert "no real hitl interrupt was observed" in text.lower()
