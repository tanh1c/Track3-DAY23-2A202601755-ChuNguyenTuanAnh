import re
from pathlib import Path


def test_manual_workflow_remains_non_automatic_and_uploads_bonus_evidence() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^  workflow_dispatch:\s*$", workflow)
    for forbidden in ("pull_request", "push", "schedule"):
        assert not re.search(rf"(?m)^  {forbidden}:\s*$", workflow)
    assert "cancel-in-progress: true" in workflow
    assert ".[dev,sqlite,ui,openai]" in workflow
    assert "outputs/bonus_evidence.json" in workflow

    upload = workflow.split("- name: Upload grading evidence", maxsplit=1)[1]
    assert "outputs/bonus_evidence.json" in upload
    assert "outputs/bonus-checkpoints.sqlite" not in upload
    assert "outputs/checkpoints.sqlite" not in upload
