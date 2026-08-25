"""Typed, machine-readable evidence for optional Day 23 extensions."""

from pathlib import Path

from pydantic import BaseModel, Field


class HitlEvidence(BaseModel):
    """Proof that a real interrupt/resume approval round-trip was exercised."""

    implemented: bool = False
    verified: bool = False
    interrupt_observed: bool = False
    same_thread_id: bool = False
    resume_success: bool = False
    rejection_verified: bool = False
    reviewer: str = ""


class TimeTravelEvidence(BaseModel):
    """Proof that checkpoint replay and fork semantics were exercised."""

    implemented: bool = False
    verified: bool = False
    replay_verified: bool = False
    fork_verified: bool = False
    original_history_preserved: bool = False
    checkpoint_id: str = ""


class ParallelSendEvidence(BaseModel):
    """Proof that the isolated bonus graph used LangGraph Send fan-out."""

    implemented: bool = False
    verified: bool = False
    used_send: bool = False
    task_count: int = 0
    result_count: int = 0
    aggregate_deterministic: bool = False


class UiEvidence(BaseModel):
    """Proof that the optional Streamlit presentation layer is secret-safe."""

    implemented: bool = False
    verified: bool = False
    view_model_verified: bool = False
    secret_safe: bool = False


class BonusEvidence(BaseModel):
    """Combined verification state for every extension listed by the lab HTML."""

    llm_as_judge_verified: bool = False
    durable_recovery_verified: bool = False
    mermaid_export_verified: bool = False
    hitl: HitlEvidence = Field(default_factory=HitlEvidence)
    time_travel: TimeTravelEvidence = Field(default_factory=TimeTravelEvidence)
    parallel_send: ParallelSendEvidence = Field(default_factory=ParallelSendEvidence)
    streamlit_ui: UiEvidence = Field(default_factory=UiEvidence)


def validate_bonus_evidence(evidence: BonusEvidence) -> None:
    """Reject positive verification claims that lack the required runtime proof."""
    hitl = evidence.hitl
    if hitl.verified and not (
        hitl.implemented
        and hitl.interrupt_observed
        and hitl.same_thread_id
        and hitl.resume_success
        and hitl.rejection_verified
        and bool(hitl.reviewer.strip())
    ):
        raise ValueError("HITL verification is missing interrupt/resume evidence")

    time_travel = evidence.time_travel
    if time_travel.verified and not (
        time_travel.implemented
        and time_travel.replay_verified
        and time_travel.fork_verified
        and time_travel.original_history_preserved
        and bool(time_travel.checkpoint_id.strip())
    ):
        raise ValueError("time travel verification is missing replay/fork evidence")

    parallel = evidence.parallel_send
    if parallel.verified and not (
        parallel.implemented
        and parallel.used_send
        and parallel.task_count > 1
        and parallel.result_count == parallel.task_count
        and parallel.aggregate_deterministic
    ):
        raise ValueError("parallel Send verification is incomplete")

    ui = evidence.streamlit_ui
    if ui.verified and not (
        ui.implemented and ui.view_model_verified and ui.secret_safe
    ):
        raise ValueError("Streamlit UI verification is incomplete or unsafe")


def write_bonus_evidence(evidence: BonusEvidence, output: str | Path) -> None:
    """Validate and persist extension evidence as deterministic JSON."""
    validate_bonus_evidence(evidence)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
