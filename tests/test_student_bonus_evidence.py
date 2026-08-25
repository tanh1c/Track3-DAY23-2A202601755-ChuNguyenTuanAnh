from pathlib import Path

import pytest

from langgraph_agent_lab.bonus_evidence import (
    BonusEvidence,
    HitlEvidence,
    ParallelSendEvidence,
    RecoveryEvidence,
    TimeTravelEvidence,
    UiEvidence,
    validate_bonus_evidence,
    write_bonus_evidence,
)


def _verified_recovery() -> RecoveryEvidence:
    return RecoveryEvidence(
        implemented=True,
        verified=True,
        writer_pid=101,
        reader_pid=202,
        distinct_processes=True,
        same_thread_id=True,
        persisted_finalized=True,
        thread_id="recovery-thread",
    )


def test_bonus_evidence_rejects_unproven_hitl() -> None:
    evidence = BonusEvidence(
        llm_as_judge_verified=True,
        durable_recovery_verified=True,
        recovery=_verified_recovery(),
        mermaid_export_verified=True,
        hitl=HitlEvidence(
            implemented=True,
            verified=True,
            interrupt_observed=False,
            same_thread_id=True,
            resume_success=True,
            rejection_verified=True,
        ),
    )
    with pytest.raises(ValueError, match="HITL"):
        validate_bonus_evidence(evidence)


def test_bonus_evidence_rejects_unproven_recovery() -> None:
    evidence = BonusEvidence(
        durable_recovery_verified=True,
        recovery=RecoveryEvidence(
            implemented=True,
            verified=True,
            writer_pid=101,
            reader_pid=101,
            distinct_processes=False,
            same_thread_id=True,
            persisted_finalized=True,
            thread_id="recovery-thread",
        ),
    )
    with pytest.raises(ValueError, match="recovery"):
        validate_bonus_evidence(evidence)


def test_bonus_evidence_rejects_unproven_time_travel() -> None:
    evidence = BonusEvidence(
        time_travel=TimeTravelEvidence(
            implemented=True,
            verified=True,
            replay_verified=True,
            fork_verified=False,
            original_history_preserved=True,
            checkpoint_id="cp-1",
        )
    )
    with pytest.raises(ValueError, match="time travel"):
        validate_bonus_evidence(evidence)


def test_bonus_evidence_rejects_invalid_parallel_send() -> None:
    evidence = BonusEvidence(
        parallel_send=ParallelSendEvidence(
            implemented=True,
            verified=True,
            used_send=True,
            task_count=3,
            result_count=2,
            aggregate_deterministic=True,
        )
    )
    with pytest.raises(ValueError, match="parallel Send"):
        validate_bonus_evidence(evidence)


def test_bonus_evidence_rejects_unsafe_ui() -> None:
    evidence = BonusEvidence(
        streamlit_ui=UiEvidence(
            implemented=True,
            verified=True,
            view_model_verified=True,
            secret_safe=False,
        )
    )
    with pytest.raises(ValueError, match="Streamlit UI"):
        validate_bonus_evidence(evidence)


def test_write_bonus_evidence_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "bonus.json"
    evidence = BonusEvidence(
        llm_as_judge_verified=True,
        durable_recovery_verified=True,
        recovery=_verified_recovery(),
        mermaid_export_verified=True,
    )
    write_bonus_evidence(evidence, output)
    loaded = BonusEvidence.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded.llm_as_judge_verified is True
    assert loaded.recovery.distinct_processes is True
