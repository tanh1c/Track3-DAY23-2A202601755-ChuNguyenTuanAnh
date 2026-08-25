# Day 23 Full Bonus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every optional extension listed by the official Day 23 HTML with executable evidence while preserving the already validated 100-point core graph unchanged.

**Architecture:** Keep `build_graph()` as the required eleven-node support-ticket graph. Add isolated bonus helpers for real HITL, checkpoint time travel, a separate `Send()` fan-out graph, a pure Streamlit view-model layer, and generated bonus evidence. Integrate these helpers through CLI/report/CI only after their focused tests pass.

**Tech Stack:** Python 3.11+, LangGraph 1.x APIs (`interrupt`, `Command`, `Send`, checkpoint history/update), Pydantic 2, SQLite LangGraph checkpointer, Typer, Streamlit as an optional extra, Ruff, MyPy, Pytest, GitHub Actions manual `workflow_dispatch`.

**Spec:** `docs/superpowers/specs/2026-08-25-day23-full-bonus-design.md`

## Global Constraints

- Preserve exactly eleven registered nodes in `src/langgraph_agent_lab/graph.py`.
- Preserve bounded retry, approval gating, finalize-before-END, current LLM routing, and existing scenario behavior.
- Bonus `Send()` functionality must live in a separate graph/module.
- Use SQLite as the durable extension proof; Postgres is not required because the official extension item is SQLite/Postgres.
- Real HITL verification must use an actual `interrupt()` plus `Command(resume=...)` with the same stable `thread_id`.
- Time travel must use checkpoint configs: replay via `graph.invoke(None, checkpoint.config)` and fork via `graph.update_state(checkpoint.config, values=...)` followed by `graph.invoke(None, fork_config)`.
- `outputs/bonus_evidence.json` must be generated from executed checks, never hand-edited to set `verified=true`.
- Streamlit remains optional and must not be required for the normal core package install.
- Do not modify public tests to hide failures.
- Do not add hidden grading data, secrets, real destructive tools, or automatic push/PR/schedule workflow triggers.
- Final GitHub Actions validation remains manual-only and should be run once after the implementation batch is complete.

---

## File Structure

### New source files

- `src/langgraph_agent_lab/bonus_evidence.py` — Pydantic evidence models, semantic validation, JSON writer.
- `src/langgraph_agent_lab/hitl.py` — programmatic real interrupt/resume verification over the existing core graph.
- `src/langgraph_agent_lab/time_travel.py` — checkpoint selection, replay, and fork helpers.
- `src/langgraph_agent_lab/bonus_parallel.py` — isolated LangGraph graph using actual `Send()` fan-out.
- `src/langgraph_agent_lab/ui.py` — pure state-to-view-model logic plus optional Streamlit renderer/entrypoint.

### New tests

- `tests/test_student_bonus_evidence.py`
- `tests/test_student_hitl_resume.py`
- `tests/test_student_time_travel.py`
- `tests/test_student_parallel_send.py`
- `tests/test_student_ui.py`

### Modified files

- `src/langgraph_agent_lab/cli.py` — bonus verification/time-travel commands.
- `src/langgraph_agent_lab/report.py` — optional generated extension matrix.
- `pyproject.toml` — optional `ui` extra.
- `.github/workflows/ci.yml` — one final manual full-bonus verification path and artifact upload.
- `tests/test_student_cli.py` and `tests/test_student_report.py` — CLI/report regression coverage for bonus integration.

### Explicitly unchanged unless a compatibility defect is proven

- `src/langgraph_agent_lab/graph.py`
- `src/langgraph_agent_lab/routing.py`
- public starter tests

---

### Task 1: Define machine-readable bonus evidence contracts

**Files:**
- Create: `src/langgraph_agent_lab/bonus_evidence.py`
- Create: `tests/test_student_bonus_evidence.py`

**Interfaces:**
- Produces: `HitlEvidence`, `TimeTravelEvidence`, `ParallelSendEvidence`, `UiEvidence`, `BonusEvidence` Pydantic models.
- Produces: `write_bonus_evidence(evidence: BonusEvidence, output: str | Path) -> None`.
- Produces: `validate_bonus_evidence(evidence: BonusEvidence) -> None` that raises `ValueError` when an implemented extension claims verification without its required proof fields.
- Later tasks return these typed evidence models instead of loose dictionaries.

- [ ] **Step 1: Write failing evidence-schema tests**

```python
from pathlib import Path

import pytest

from langgraph_agent_lab.bonus_evidence import (
    BonusEvidence,
    HitlEvidence,
    ParallelSendEvidence,
    TimeTravelEvidence,
    UiEvidence,
    validate_bonus_evidence,
    write_bonus_evidence,
)


def test_bonus_evidence_rejects_unproven_hitl() -> None:
    evidence = BonusEvidence(
        llm_as_judge_verified=True,
        durable_recovery_verified=True,
        mermaid_export_verified=True,
        hitl=HitlEvidence(
            implemented=True,
            verified=True,
            interrupt_observed=False,
            same_thread_id=True,
            resume_success=True,
            rejection_verified=True,
        ),
        time_travel=TimeTravelEvidence(implemented=True),
        parallel_send=ParallelSendEvidence(implemented=True),
        streamlit_ui=UiEvidence(implemented=True),
    )
    with pytest.raises(ValueError, match="HITL"):
        validate_bonus_evidence(evidence)


def test_write_bonus_evidence_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "bonus.json"
    evidence = BonusEvidence(
        llm_as_judge_verified=True,
        durable_recovery_verified=True,
        mermaid_export_verified=True,
    )
    write_bonus_evidence(evidence, output)
    loaded = BonusEvidence.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded.llm_as_judge_verified is True
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest tests/test_student_bonus_evidence.py -q
```

Expected: import/module failure because `bonus_evidence.py` does not exist.

- [ ] **Step 3: Implement strict evidence models**

Use this public shape:

```python
from pathlib import Path

from pydantic import BaseModel, Field


class HitlEvidence(BaseModel):
    implemented: bool = False
    verified: bool = False
    interrupt_observed: bool = False
    same_thread_id: bool = False
    resume_success: bool = False
    rejection_verified: bool = False
    reviewer: str = ""


class TimeTravelEvidence(BaseModel):
    implemented: bool = False
    verified: bool = False
    replay_verified: bool = False
    fork_verified: bool = False
    original_history_preserved: bool = False
    checkpoint_id: str = ""


class ParallelSendEvidence(BaseModel):
    implemented: bool = False
    verified: bool = False
    used_send: bool = False
    task_count: int = 0
    result_count: int = 0
    aggregate_deterministic: bool = False


class UiEvidence(BaseModel):
    implemented: bool = False
    verified: bool = False
    view_model_verified: bool = False
    secret_safe: bool = False


class BonusEvidence(BaseModel):
    llm_as_judge_verified: bool = False
    durable_recovery_verified: bool = False
    mermaid_export_verified: bool = False
    hitl: HitlEvidence = Field(default_factory=HitlEvidence)
    time_travel: TimeTravelEvidence = Field(default_factory=TimeTravelEvidence)
    parallel_send: ParallelSendEvidence = Field(default_factory=ParallelSendEvidence)
    streamlit_ui: UiEvidence = Field(default_factory=UiEvidence)
```

`validate_bonus_evidence()` must require the following before a nested extension may have `verified=True`:

- HITL: interrupt observed, same thread, resume success, rejection verified.
- time travel: replay verified, fork verified, original history preserved, checkpoint ID non-empty.
- parallel send: actual `Send`, `task_count > 1`, result count equals task count, deterministic aggregate.
- UI: view model verified and secret-safe.

`write_bonus_evidence()` first calls the semantic validator, creates the parent directory, and writes `model_dump_json(indent=2)`.

- [ ] **Step 4: Run focused tests and confirm GREEN**

```bash
pytest tests/test_student_bonus_evidence.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run Ruff/MyPy on the new module**

```bash
ruff check src/langgraph_agent_lab/bonus_evidence.py tests/test_student_bonus_evidence.py
mypy src/langgraph_agent_lab/bonus_evidence.py
```

Expected: zero errors.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/bonus_evidence.py tests/test_student_bonus_evidence.py
git commit -m "feat: add typed bonus evidence contracts"
```

---

### Task 2: Add real HITL interrupt/resume verification

**Files:**
- Create: `src/langgraph_agent_lab/hitl.py`
- Create: `tests/test_student_hitl_resume.py`

**Interfaces:**
- Consumes: `build_graph(checkpointer)`, `build_checkpointer("sqlite", database_url)`, existing `approval_node()` interrupt behavior, and `HitlEvidence`.
- Produces: `verify_hitl_round_trip(database_url: str, *, thread_id: str = "bonus-hitl-approved") -> HitlEvidence`.
- Produces: internal `_run_review_case(..., approved: bool, reviewer: str, comment: str)` helper returning final state plus whether an interrupt was observed.

- [ ] **Step 1: Write RED tests for approval and rejection resume flows**

Tests must monkeypatch classifier/answer/evaluator LLM-dependent nodes where necessary so the tests exercise LangGraph interrupt semantics without consuming API quota.

Minimum assertions:

```python
assert approved.interrupt_observed is True
assert approved.same_thread_id is True
assert approved.resume_success is True
assert approved.reviewer == "ci-reviewer"
assert approved.rejection_verified is True
```

Also inspect the approved final state to ensure a `tool` event occurs after an `approval` event, and inspect the rejected final state to ensure `clarify` occurs and no `tool` event occurs.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
pytest tests/test_student_hitl_resume.py -q
```

Expected: import/function failure for missing HITL helper.

- [ ] **Step 3: Implement the HITL verifier using the real LangGraph protocol**

Use the current runtime contract:

```python
from langgraph.types import Command

initial = graph.invoke(state, config=config)
interrupts = list(initial.get("__interrupt__", []))
if not interrupts:
    raise RuntimeError("Expected approval interrupt")

resumed = graph.invoke(
    Command(
        resume={
            "approved": approved,
            "reviewer": reviewer,
            "comment": comment,
        }
    ),
    config=config,
)
```

Important implementation rules:

- Temporarily enable `LANGGRAPH_INTERRUPT=true` only inside the helper and restore the previous environment value in `finally`.
- Use a durable SQLite saver and stable thread ID.
- Use separate thread IDs for approval and rejection cases so one path cannot contaminate the other.
- Confirm the returned approval event has `metadata.real_interrupt == True`.
- Confirm the approved path contains `risky_action -> approval -> tool` and reaches `finalize`.
- Confirm the rejected path contains `risky_action -> approval -> clarify -> finalize` and does not contain `tool` after approval.
- Evidence must come from observed state/events, not assumptions.

- [ ] **Step 4: Run focused HITL tests and confirm GREEN**

```bash
pytest tests/test_student_hitl_resume.py -q
```

Expected: pass with no live API calls.

- [ ] **Step 5: Run existing approval/node regressions**

```bash
pytest tests/test_student_nodes.py tests/test_student_graph.py -q
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/hitl.py tests/test_student_hitl_resume.py
git commit -m "feat: verify real HITL interrupt and resume"
```

---

### Task 3: Implement checkpoint replay and fork time travel

**Files:**
- Create: `src/langgraph_agent_lab/time_travel.py`
- Create: `tests/test_student_time_travel.py`

**Interfaces:**
- Consumes: compiled graph with SQLite checkpointer and LangGraph `StateSnapshot.config` / `next`.
- Produces: `list_checkpoints(graph: Any, thread_id: str) -> list[CheckpointInfo]`.
- Produces: `find_checkpoint(graph: Any, thread_id: str, checkpoint_id: str) -> Any`.
- Produces: `replay_checkpoint(graph: Any, checkpoint: Any) -> dict[str, Any]`.
- Produces: `fork_checkpoint(graph: Any, checkpoint: Any, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]` where the first item is the new fork config and the second is the final state.
- Produces: `verify_time_travel(database_url: str) -> TimeTravelEvidence` using a deterministic offline fixture graph or monkeypatched core nodes so no live model is required.

- [ ] **Step 1: Write RED tests for history, unknown checkpoint, replay, and fork**

Required checks:

```python
history = list_checkpoints(graph, thread_id)
assert history
checkpoint_info = next(item for item in history if item.next_nodes)
selected_snapshot = find_checkpoint(graph, thread_id, checkpoint_info.checkpoint_id)

with pytest.raises(ValueError, match="Unknown checkpoint"):
    find_checkpoint(graph, thread_id, "missing-checkpoint")

replayed = replay_checkpoint(graph, selected_snapshot)
assert any(event["node"] == "finalize" for event in replayed["events"])

original_ids_before = {item.checkpoint_id for item in list_checkpoints(graph, thread_id)}
fork_config, forked = fork_checkpoint(graph, selected_snapshot, {"query": "forked request"})
original_ids_after = {item.checkpoint_id for item in list_checkpoints(graph, thread_id)}
assert original_ids_before <= original_ids_after
assert fork_config["configurable"]["checkpoint_id"] not in original_ids_before
assert any(event["node"] == "finalize" for event in forked["events"])
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
pytest tests/test_student_time_travel.py -q
```

Expected: missing module/functions.

- [ ] **Step 3: Implement checkpoint helpers using official replay/fork semantics**

`CheckpointInfo` should be a small Pydantic model:

```python
class CheckpointInfo(BaseModel):
    checkpoint_id: str
    thread_id: str
    next_nodes: tuple[str, ...]
    route: str = ""
    attempt: int = 0
```

Replay must call:

```python
result = graph.invoke(None, checkpoint.config)
```

Fork must call:

```python
fork_config = graph.update_state(checkpoint.config, values=values)
result = graph.invoke(None, fork_config)
```

Never replace a requested unknown `checkpoint_id` with the latest checkpoint.

For verification, select a non-final checkpoint whose `next` is non-empty so replay is not a no-op. Preserve and compare pre-fork history to prove the earlier execution remains readable.

- [ ] **Step 4: Run focused time-travel tests and confirm GREEN**

```bash
pytest tests/test_student_time_travel.py -q
```

- [ ] **Step 5: Run persistence regressions**

```bash
pytest tests/test_student_persistence.py tests/test_student_cli.py -q
```

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/time_travel.py tests/test_student_time_travel.py
git commit -m "feat: add checkpoint replay and fork time travel"
```

---

### Task 4: Add isolated parallel map-reduce graph using actual `Send()`

**Files:**
- Create: `src/langgraph_agent_lab/bonus_parallel.py`
- Create: `tests/test_student_parallel_send.py`

**Interfaces:**
- Produces: `ParallelState` TypedDict with reducer-managed `results`.
- Produces: `dispatch_node(state: ParallelState) -> dict[str, object]`.
- Produces: `plan_tasks(state: ParallelState) -> list[Send]`.
- Produces: `run_task(state: TaskState) -> dict[str, list[str]]`.
- Produces: `aggregate_results(state: ParallelState) -> dict[str, str]`.
- Produces: `build_parallel_bonus_graph() -> Any`.
- Produces: `verify_parallel_send(tasks: list[str] | None = None) -> ParallelSendEvidence`.

- [ ] **Step 1: Write RED tests proving the planner returns actual `Send` objects**

```python
from langgraph.types import Send

routes = plan_tasks({"tasks": ["account", "order", "policy"], "results": []})
assert len(routes) == 3
assert all(isinstance(route, Send) for route in routes)
```

Also compile/invoke the separate graph and verify one result per task and deterministic aggregate membership:

```python
final = build_parallel_bonus_graph().invoke({
    "tasks": ["order", "policy", "account"],
    "results": [],
})
assert final["aggregate"] == "account|order|policy"
assert sorted(final["results"]) == ["account", "order", "policy"]
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
pytest tests/test_student_parallel_send.py -q
```

- [ ] **Step 3: Implement the isolated graph**

Use the documented LangGraph map-reduce shape:

```python
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class ParallelState(TypedDict, total=False):
    tasks: list[str]
    results: Annotated[list[str], add]
    aggregate: str


class TaskState(TypedDict):
    task: str


def dispatch_node(state: ParallelState) -> dict[str, object]:
    return {}


def plan_tasks(state: ParallelState) -> list[Send]:
    return [Send("run_task", {"task": task}) for task in state["tasks"]]


def run_task(state: TaskState) -> dict[str, list[str]]:
    return {"results": [state["task"]]}


def aggregate_results(state: ParallelState) -> dict[str, str]:
    return {"aggregate": "|".join(sorted(state.get("results", [])))}
```

Build exactly this isolated topology:

```python
builder = StateGraph(ParallelState)
builder.add_node("dispatch", dispatch_node)
builder.add_node("run_task", run_task)
builder.add_node("aggregate", aggregate_results)
builder.add_edge(START, "dispatch")
builder.add_conditional_edges("dispatch", plan_tasks, ["run_task"])
builder.add_edge("run_task", "aggregate")
builder.add_edge("aggregate", END)
```

The verifier must inspect `plan_tasks()` output to set `used_send=True`, invoke the compiled bonus graph, and compare sorted result membership to sorted input membership.

- [ ] **Step 4: Confirm focused tests GREEN**

```bash
pytest tests/test_student_parallel_send.py -q
```

- [ ] **Step 5: Prove core graph remains eleven nodes**

```bash
pytest tests/test_student_graph.py -q
```

Expected: existing eleven-node graph assertions pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/bonus_parallel.py tests/test_student_parallel_send.py
git commit -m "feat: add isolated Send fan-out bonus graph"
```

---

### Task 5: Add optional Streamlit UI and secret-safe view model

**Files:**
- Create: `src/langgraph_agent_lab/ui.py`
- Create: `tests/test_student_ui.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `build_view_model(state: AgentState, *, checkpoint_id: str = "") -> dict[str, object]`.
- Produces: `verify_ui_view_model() -> UiEvidence`.
- Produces: `main() -> None` Streamlit entrypoint that imports Streamlit lazily.

- [ ] **Step 1: Write RED view-model and secret-safety tests**

Representative state:

```python
state = {
    "thread_id": "thread-ui",
    "query": "refund order",
    "route": "risky",
    "risk_level": "high",
    "proposed_action": "refund order",
    "approval": {"approved": True, "reviewer": "alice", "comment": "ok"},
    "final_answer": "completed",
    "events": [{"node": "finalize", "event_type": "completed", "message": "done"}],
}
```

Assert the view model contains only presentation fields and does not contain strings from `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` even when those environment variables are set during the test.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
pytest tests/test_student_ui.py -q
```

- [ ] **Step 3: Add Streamlit as optional dependency only**

Add:

```toml
ui = ["streamlit>=1.40"]
```

Do not add Streamlit to base dependencies.

- [ ] **Step 4: Implement pure view model and lazy renderer**

`build_view_model()` returns these keys only:

```text
thread_id
checkpoint_id
query
route
risk_level
proposed_action
approval
final_answer
events
```

Do not read environment variables inside the view model.

`main()` performs `import streamlit as st` inside the function, initializes a simple state object from form controls, and renders the view-model fields. It must not render `os.environ`, provider config, or `.env` content.

`verify_ui_view_model()` builds a representative state, verifies every required key exists, serializes the view model to text, compares it against any currently configured provider secret values, and returns `UiEvidence(implemented=True, verified=True, view_model_verified=True, secret_safe=True)` only when safe.

- [ ] **Step 5: Run focused UI tests and confirm GREEN**

```bash
pytest tests/test_student_ui.py -q
```

- [ ] **Step 6: Run packaging/type regressions**

```bash
mypy src
pytest tests/test_student_report.py tests/test_student_cli.py -q
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/langgraph_agent_lab/ui.py tests/test_student_ui.py
git commit -m "feat: add optional Streamlit evidence UI"
```

---

### Task 6: Integrate bonus CLI commands and generated evidence

**Files:**
- Modify: `src/langgraph_agent_lab/cli.py`
- Modify: `tests/test_student_cli.py`
- Modify: `src/langgraph_agent_lab/bonus_evidence.py`
- Modify: `tests/test_student_bonus_evidence.py`

**Interfaces:**
- Produces CLI command `time-travel`.
- Produces CLI command `verify-bonus`.
- `verify-bonus` writes `outputs/bonus_evidence.json` from executable verification helpers.

- [ ] **Step 1: Write RED CLI tests**

Test command registration and one deterministic invocation with helper monkeypatching:

```python
result = runner.invoke(cli.app, ["verify-bonus", "--output", str(output)])
assert result.exit_code == 0
assert output.exists()
```

For `time-travel`, ensure unknown checkpoint exits non-zero with an explicit error message and known checkpoint produces compact replay/fork output.

- [ ] **Step 2: Run CLI tests and confirm RED**

```bash
pytest tests/test_student_cli.py -q
```

- [ ] **Step 3: Implement `time-travel` command**

Required options:

```text
--database PATH
--thread-id TEXT
--checkpoint-id TEXT
--mode replay|fork
--set-query TEXT   # optional; used only for fork
```

Rules:

- Build a SQLite checkpointer from `--database`.
- Find the exact checkpoint; unknown ID is an error.
- Replay invokes from checkpoint without rewriting history.
- Fork requires at least one explicit state override; initial supported override is query.
- Print checkpoint ID, mode, route, attempt, and whether finalization was reached.

- [ ] **Step 4: Implement `verify-bonus` command**

Signature:

```text
agent-lab verify-bonus \
  --database outputs/bonus-checkpoints.sqlite \
  --output outputs/bonus_evidence.json
```

The command executes:

1. `verify_hitl_round_trip()`.
2. `verify_time_travel()`.
3. `verify_parallel_send()`.
4. `verify_ui_view_model()`.
5. Sets existing extension booleans for LLM-as-judge, durable recovery, and Mermaid only when their prerequisite artifacts/runtime checks are actually available.
6. Calls `validate_bonus_evidence()`.
7. Writes the JSON.

Do not mark LLM-as-judge verified from file existence alone. In CI, pass an explicit CLI option such as `--llm-judge-verified` only after the live scenario step completed successfully. Likewise use `--durable-recovery-verified` only after metrics show `resume_success=true`, and `--mermaid-verified` only after graph export validation.

- [ ] **Step 5: Confirm CLI tests GREEN**

```bash
pytest tests/test_student_cli.py tests/test_student_bonus_evidence.py -q
```

- [ ] **Step 6: Run all bonus-focused offline tests**

```bash
pytest \
  tests/test_student_bonus_evidence.py \
  tests/test_student_hitl_resume.py \
  tests/test_student_time_travel.py \
  tests/test_student_parallel_send.py \
  tests/test_student_ui.py \
  tests/test_student_cli.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/langgraph_agent_lab/cli.py src/langgraph_agent_lab/bonus_evidence.py \
  tests/test_student_cli.py tests/test_student_bonus_evidence.py
git commit -m "feat: integrate full bonus verification CLI"
```

---

### Task 7: Render evidence-backed extension matrix in the lab report

**Files:**
- Modify: `src/langgraph_agent_lab/report.py`
- Modify: `tests/test_student_report.py`
- Modify: `src/langgraph_agent_lab/cli.py`

**Interfaces:**
- Change: `render_report(metrics: MetricsReport, bonus: BonusEvidence | None = None) -> str`.
- Change: `write_report(metrics: MetricsReport, output_path: str | Path, bonus: BonusEvidence | None = None) -> None`.
- Existing callers remain compatible because `bonus` defaults to `None`.

- [ ] **Step 1: Write RED report tests**

Construct a fully verified `BonusEvidence` and assert the rendered report contains an extension matrix with all official extension names and evidence values.

Also test `bonus=None` retains the current core report behavior so existing calls are not broken.

- [ ] **Step 2: Run report tests and confirm RED**

```bash
pytest tests/test_student_report.py -q
```

- [ ] **Step 3: Implement report integration**

When bonus evidence is supplied, append a table under `## Extension Work`:

```text
| Extension | Implemented | Verified | Evidence |
|---|---:|---:|---|
| LLM-as-judge | yes | yes | structured evaluator exercised in live scenario gate |
| Real HITL | yes | yes | interrupt + same-thread Command(resume) + rejection path |
| SQLite recovery | yes | yes | fresh saver read completed stable thread |
| Time travel | yes | yes | replay + fork + original history preserved |
| Parallel Send | yes | yes | N tasks -> N reducer results using Send |
| Streamlit UI | yes | yes | import/view-model/secret-safety smoke |
| Mermaid export | yes | yes | compiled eleven-node core graph |
```

The evidence text must derive from the evidence object. Do not print `yes` when corresponding verification is false.

Update `verify-bonus` or a small `refresh-report` path so after `bonus_evidence.json` is generated, the report is re-rendered from the existing validated metrics plus bonus evidence.

- [ ] **Step 4: Confirm report tests GREEN**

```bash
pytest tests/test_student_report.py tests/test_student_cli.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/report.py src/langgraph_agent_lab/cli.py \
  tests/test_student_report.py tests/test_student_cli.py
git commit -m "feat: report full bonus runtime evidence"
```

---

### Task 8: Extend the manual CI gate without introducing automatic runs

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_student_cli.py` or add a focused workflow-contract test only if needed; do not modify public tests.

**Interfaces:**
- Full-live installs `.[dev,sqlite,ui,<selected-provider>]`.
- Full-live produces and validates `outputs/bonus_evidence.json`.
- Artifact upload contains exactly core evidence plus bonus JSON; SQLite files remain excluded.

- [ ] **Step 1: Add/extend a test that checks manual-only workflow semantics**

The test or CI hygiene script must still verify:

```text
workflow_dispatch present
push trigger absent
pull_request trigger absent
schedule trigger absent
cancel-in-progress true
```

Also assert `outputs/bonus_evidence.json` appears in the artifact path and no `*.sqlite` path is uploaded.

- [ ] **Step 2: Modify full-live dependency install**

Selected provider examples:

```bash
python -m pip install -e '.[dev,sqlite,ui,openai]'
```

Static mode may remain `.[dev,sqlite]` unless UI smoke is intentionally part of static validation. Because the full final gate exercises Streamlit, `ui` is mandatory only for `full-live`.

- [ ] **Step 3: Add bonus verification steps after core metrics and Mermaid are proven**

Recommended sequence after `Export compiled Mermaid graph`:

```yaml
- name: Verify full bonus extensions
  if: inputs.mode == 'full-live'
  env:
    OPENAI_API_KEY: ""
    GEMINI_API_KEY: ""
    ANTHROPIC_API_KEY: ""
    LANGGRAPH_INTERRUPT: "false"
  run: |
    python -m langgraph_agent_lab.cli verify-bonus \
      --database outputs/bonus-checkpoints.sqlite \
      --output outputs/bonus_evidence.json \
      --llm-judge-verified \
      --durable-recovery-verified \
      --mermaid-verified
```

Important: the bonus verification helper itself should not require live LLM calls. The `--llm-judge-verified` flag is valid only because the earlier live scenario step is a hard predecessor in the same job.

- [ ] **Step 4: Extend semantic evidence validation**

Validate at minimum:

```python
bonus = json.loads(Path("outputs/bonus_evidence.json").read_text())
assert bonus["llm_as_judge_verified"] is True
assert bonus["durable_recovery_verified"] is True
assert bonus["mermaid_export_verified"] is True
assert bonus["hitl"]["verified"] is True
assert bonus["hitl"]["interrupt_observed"] is True
assert bonus["hitl"]["same_thread_id"] is True
assert bonus["hitl"]["rejection_verified"] is True
assert bonus["time_travel"]["replay_verified"] is True
assert bonus["time_travel"]["fork_verified"] is True
assert bonus["time_travel"]["original_history_preserved"] is True
assert bonus["parallel_send"]["used_send"] is True
assert bonus["parallel_send"]["task_count"] == bonus["parallel_send"]["result_count"]
assert bonus["parallel_send"]["task_count"] > 1
assert bonus["streamlit_ui"]["verified"] is True
assert bonus["streamlit_ui"]["secret_safe"] is True
```

Also require the report to contain each official extension name.

- [ ] **Step 5: Extend artifact upload**

Artifact paths:

```text
outputs/metrics.json
outputs/graph.mmd
outputs/bonus_evidence.json
reports/lab_report.md
```

Do not upload `outputs/checkpoints.sqlite`, `outputs/bonus-checkpoints.sqlite`, WAL, or SHM files.

- [ ] **Step 6: Run all offline gates once before requesting CI**

```bash
make lint
make typecheck
make test
git diff --check origin/main...HEAD
```

Expected: zero failures.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml tests
git commit -m "ci: validate complete Day 23 bonus evidence"
```

---

### Task 9: Final review and one fresh full-live verification run

**Files:**
- No source changes unless verification finds a concrete defect.
- Generated evidence after a green run: `outputs/metrics.json`, `outputs/graph.mmd`, `outputs/bonus_evidence.json`, `reports/lab_report.md`.

**Interfaces:**
- Consumes the complete implementation from Tasks 1-8.
- Produces final review evidence and a submission-ready PR.

- [ ] **Step 1: Review core topology diff before running CI**

Confirm `src/langgraph_agent_lab/graph.py` is unchanged from the previously green implementation SHA unless a documented compatibility fix was necessary.

Run/inspect:

```bash
git diff d6ea5833e59c5ff349eda56b9139e07277ea30c0 -- src/langgraph_agent_lab/graph.py
```

Expected: empty diff.

- [ ] **Step 2: Run complete offline verification**

```bash
make lint
make typecheck
make test
python -m langgraph_agent_lab.cli export-graph --output outputs/graph.mmd
git diff --check origin/main...HEAD
```

Expected: all commands exit zero.

- [ ] **Step 3: Perform code review against the written spec**

Check each official extension:

```text
LLM-as-judge            implementation + live predecessor evidence
Real HITL               actual interrupt + Command(resume) + reject path
SQLite recovery         restart-style proof
Time travel             replay + fork + history preservation
Parallel Send           actual Send objects + reducer aggregation
Streamlit UI            optional extra + view model + secret safety
Mermaid export          compiled core graph
```

No Critical or Important review issues may remain before CI.

- [ ] **Step 4: Trigger exactly one fresh manual GitHub Actions run**

Use:

```text
workflow: manual-ci
branch: feat/day23-full-score-bonus
mode: full-live
provider: openai
```

Do not use `Re-run` on an older workflow run after new commits; create a new workflow dispatch so it uses the latest branch SHA.

- [ ] **Step 5: Verify every CI step and artifact**

Required green stages:

```text
install
secret preflight
lint
typecheck
offline tests
submission hygiene
live graph smoke
live seven scenarios
metrics validation
Mermaid export
full bonus verification
semantic evidence validation
artifact upload
```

Inspect the downloaded artifact and confirm:

- `metrics.json`: 7 scenarios, success rate 1.0, meaningful retries/approval/latencies, durable recovery true.
- `graph.mmd`: all eleven required core nodes.
- `bonus_evidence.json`: every official extension is verified under the strict evidence semantics.
- `lab_report.md`: core metrics and extension matrix match generated JSON evidence.

- [ ] **Step 6: Commit the final generated evidence snapshot only after the green run**

Commit the four text/JSON artifacts, but never SQLite files:

```bash
git add outputs/metrics.json outputs/graph.mmd outputs/bonus_evidence.json reports/lab_report.md
git commit -m "docs: record validated full bonus evidence"
```

After this evidence-only commit, compare it to the green implementation SHA and confirm no source/config/test files changed.

- [ ] **Step 7: Mark PR ready only after evidence review**

Update the PR body with the fresh run ID, verified implementation SHA, core score evidence, and all seven extension proofs. Do not merge until the user explicitly requests merge.
