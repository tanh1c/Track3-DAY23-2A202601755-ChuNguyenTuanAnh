from pathlib import Path

from langgraph_agent_lab.recovery import verify_subprocess_recovery


def test_subprocess_recovery_uses_distinct_processes(tmp_path: Path) -> None:
    evidence = verify_subprocess_recovery(tmp_path / "recovery.sqlite")

    assert evidence.implemented is True
    assert evidence.verified is True
    assert evidence.writer_pid > 0
    assert evidence.reader_pid > 0
    assert evidence.writer_pid != evidence.reader_pid
    assert evidence.distinct_processes is True
    assert evidence.same_thread_id is True
    assert evidence.persisted_finalized is True
    assert evidence.thread_id
