from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import langgraph_agent_lab.cli as cli
from langgraph_agent_lab.bonus_evidence import (
    HitlEvidence,
    ParallelSendEvidence,
    RecoveryEvidence,
    TimeTravelEvidence,
    UiEvidence,
)


runner = CliRunner()


def test_verify_bonus_writes_runtime_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "verify_hitl_round_trip",
        lambda _db: HitlEvidence(
            implemented=True,
            verified=True,
            interrupt_observed=True,
            same_thread_id=True,
            resume_success=True,
            rejection_verified=True,
            reviewer="ci-reviewer",
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_subprocess_recovery",
        lambda _db: RecoveryEvidence(
            implemented=True,
            verified=True,
            writer_pid=101,
            reader_pid=202,
            distinct_processes=True,
            same_thread_id=True,
            persisted_finalized=True,
            thread_id="recovery-thread",
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_time_travel",
        lambda _db: TimeTravelEvidence(
            implemented=True,
            verified=True,
            replay_verified=True,
            fork_verified=True,
            original_history_preserved=True,
            checkpoint_id="cp-1",
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_parallel_send",
        lambda: ParallelSendEvidence(
            implemented=True,
            verified=True,
            used_send=True,
            task_count=3,
            result_count=3,
            aggregate_deterministic=True,
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_ui_view_model",
        lambda: UiEvidence(
            implemented=True,
            verified=True,
            view_model_verified=True,
            secret_safe=True,
        ),
    )

    output = tmp_path / "bonus.json"
    result = runner.invoke(
        cli.app,
        [
            "verify-bonus",
            "--database",
            str(tmp_path / "bonus.sqlite"),
            "--output",
            str(output),
            "--llm-judge-verified",
            "--mermaid-verified",
        ],
    )
    assert result.exit_code == 0
    payload = output.read_text(encoding="utf-8")
    assert '"llm_as_judge_verified": true' in payload
    assert '"durable_recovery_verified": true' in payload
    assert '"distinct_processes": true' in payload


def test_time_travel_command_replays_exact_checkpoint(monkeypatch) -> None:
    snapshot = SimpleNamespace(config={"configurable": {"checkpoint_id": "cp-1"}})
    monkeypatch.setattr(cli, "build_checkpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "build_graph", lambda checkpointer=None: object())
    monkeypatch.setattr(cli, "find_checkpoint", lambda *_args: snapshot)
    monkeypatch.setattr(
        cli,
        "replay_checkpoint",
        lambda *_args: {
            "route": "error",
            "attempt": 2,
            "events": [{"node": "finalize"}],
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "time-travel",
            "--database",
            "ignored.sqlite",
            "--thread-id",
            "thread-x",
            "--checkpoint-id",
            "cp-1",
            "--mode",
            "replay",
        ],
    )
    assert result.exit_code == 0
    assert "checkpoint=cp-1" in result.stdout
    assert "mode=replay" in result.stdout
    assert "finalized=yes" in result.stdout


def test_time_travel_command_rejects_unknown_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_checkpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "build_graph", lambda checkpointer=None: object())

    def missing(*_args):
        raise ValueError("Unknown checkpoint: missing")

    monkeypatch.setattr(cli, "find_checkpoint", missing)
    result = runner.invoke(
        cli.app,
        [
            "time-travel",
            "--database",
            "ignored.sqlite",
            "--thread-id",
            "thread-x",
            "--checkpoint-id",
            "missing",
            "--mode",
            "replay",
        ],
    )
    assert result.exit_code == 2
    assert "Unknown checkpoint: missing" in result.output
