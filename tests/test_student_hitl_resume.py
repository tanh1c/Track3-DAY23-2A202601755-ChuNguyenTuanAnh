from pathlib import Path

from langgraph_agent_lab.hitl import verify_hitl_round_trip


def test_real_hitl_round_trip_verifies_approve_and_reject(tmp_path: Path) -> None:
    evidence = verify_hitl_round_trip(str(tmp_path / "hitl.sqlite"), thread_id="test-hitl")

    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.interrupt_observed is True
    assert evidence.same_thread_id is True
    assert evidence.resume_success is True
    assert evidence.rejection_verified is True
    assert evidence.reviewer == "ci-reviewer"


def test_hitl_verifier_restores_interrupt_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "sentinel")
    verify_hitl_round_trip(str(tmp_path / "restore.sqlite"), thread_id="restore-hitl")
    assert __import__("os").environ["LANGGRAPH_INTERRUPT"] == "sentinel"
