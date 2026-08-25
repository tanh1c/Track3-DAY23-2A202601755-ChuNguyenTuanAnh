# Day 23 LangGraph Full-Score + Bonus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Day 23 Track 3 LangGraph support-ticket agent so the official Codelab core contract is fully satisfied, then add evidence-backed bonus extensions without destabilizing core behavior.

**Architecture:** Keep the starter `AgentState` + eleven-node LangGraph architecture intact, implement all node/routing contracts with real LLM classification/answer generation, then add durable SQLite recovery, measured metrics, report evidence, feature-gated HITL, LLM-as-judge, time travel, and Mermaid export. GitHub Actions is manual-only and is used as an explicit validation surface rather than an automatic per-commit gate.

**Tech Stack:** Python 3.11+, LangGraph, Pydantic v2, Typer, PyYAML, python-dotenv, optional LangChain Gemini/OpenAI/Anthropic adapters, `langgraph-checkpoint-sqlite`, pytest, Ruff, MyPy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-25-day23-langgraph-full-score-bonus-design.md`

## Global Constraints

- Official VLearn Codelab HTML is the highest-priority behavioral source of truth.
- Complete 100/100 core behavior before bonus work is considered done.
- Preserve exactly eleven graph nodes; the four routing helpers remain conditional-edge functions, not nodes.
- Classification priority is `risky > tool > missing_info > error > simple`.
- `classify_node` must use real structured-output LLM classification in production code.
- `answer_node` must use a real LLM and ground its response in state/tool/approval context.
- Never hard-code sample scenario IDs or exact sample query lookup tables.
- Only `retry_or_fallback_node` increments `attempt`, exactly once per visit.
- `attempt < max_attempts` routes to `tool`; otherwise route to `dead_letter`.
- Every terminal path must pass through `finalize` before `END`.
- `route` remains the original classified input route; terminal nodes do not overwrite it.
- Mock approval remains the default; real `interrupt()` is feature-gated and never blocks default CI.
- No API key, database credential, hidden grading data, or secret value may enter Git history, metrics, reports, screenshots, or logs.
- GitHub Actions must not use `push`, `pull_request`, or `schedule` triggers.
- Live LLM/API runs happen only through explicit local commands or manually dispatched `full-live` CI.
- Public tests, sample scenarios, and grading boundaries are contracts; do not edit them to hide defects.

## Repository discovery amendment

The approved design named `.github/workflows/manual-ci.yml` as the planned workflow path. Repository inspection after spec approval found an existing `.github/workflows/ci.yml` that currently triggers on pull requests and pushes to `main`. Reuse and replace that existing workflow instead of creating a second workflow. This changes only the file path, not the approved manual-only CI behavior.

## File Structure

### Existing files to modify

- `.github/workflows/ci.yml` — single manual-only CI workflow with `static` and `full-live` modes.
- `pyproject.toml` — add `python-dotenv`; keep provider extras; use SQLite extra in CI/runtime where required.
- `src/langgraph_agent_lab/state.py` — add current-value fields required by routing/approval/evaluation.
- `src/langgraph_agent_lab/routing.py` — implement all four decision tables.
- `src/langgraph_agent_lab/nodes.py` — implement ten TODO nodes and feature-gated LLM/HITL behavior.
- `src/langgraph_agent_lab/llm.py` — load `.env` once and preserve provider selection contract.
- `src/langgraph_agent_lab/graph.py` — register/wire/compile the eleven-node graph.
- `src/langgraph_agent_lab/persistence.py` — MemorySaver + SQLite durable saver.
- `src/langgraph_agent_lab/metrics.py` — real latency/retry/interrupt/recovery semantics.
- `src/langgraph_agent_lab/report.py` — deterministic evidence-based Markdown report.
- `src/langgraph_agent_lab/cli.py` — scenario timing, recovery evidence, graph export/time-travel commands where needed.
- `configs/lab.yaml` — switch final evidence run to durable SQLite only after memory-based core is stable.
- `reports/lab_report.md` — final generated/evidence-backed report.
- `outputs/metrics.json` — final real sample-run metrics.

### New files to create

- `src/langgraph_agent_lab/schemas.py` — Pydantic schemas for LLM classification/evaluation outputs.
- `tests/test_student_state_routing.py` — added state/routing contract tests.
- `tests/test_student_nodes.py` — deterministic node contract and mutation tests.
- `tests/test_student_llm_nodes.py` — fake-LLM tests for structured classification, grounding, evaluator fallback.
- `tests/test_student_graph.py` — graph topology and route termination tests with fake LLM.
- `tests/test_student_persistence.py` — SQLite restart-style recovery/state-history tests.
- `tests/test_student_report.py` — report section/data consistency tests.
- `tests/test_student_cli.py` — latency/recovery/export CLI tests where direct unit coverage is useful.

---

### Task 1: Make CI Manual-Only and Fix Environment Loading Dependency

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Test: inspect workflow YAML + dependency metadata; no live API call

**Interfaces:**
- Consumes: existing Make targets `lint`, `typecheck`, `test`, `run-scenarios`, `grade-local`.
- Produces: manual workflow input `mode` (`static`/`full-live`) and `provider` (`openai`/`google`/`anthropic`); dependency `python-dotenv>=1.0`.

- [ ] **Step 1: Replace automatic triggers with a failing safety check in review**

Before editing, confirm the current workflow contains both forbidden triggers:

```yaml
on:
  pull_request:
  push:
    branches: [main]
```

Treat the current file as failing the approved CI-safety contract.

- [ ] **Step 2: Replace `.github/workflows/ci.yml` with manual-only dispatch**

Use this structure:

```yaml
name: manual-ci

on:
  workflow_dispatch:
    inputs:
      mode:
        description: Validation depth
        required: true
        default: static
        type: choice
        options:
          - static
          - full-live
      provider:
        description: LLM provider for full-live
        required: true
        default: openai
        type: choice
        options:
          - openai
          - google
          - anthropic

permissions:
  contents: read

concurrency:
  group: day23-manual-ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  validate:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ inputs.provider == 'openai' && secrets.OPENAI_API_KEY || '' }}
      GEMINI_API_KEY: ${{ inputs.provider == 'google' && secrets.GEMINI_API_KEY || '' }}
      ANTHROPIC_API_KEY: ${{ inputs.provider == 'anthropic' && secrets.ANTHROPIC_API_KEY || '' }}
      LLM_MODEL: ${{ vars.LLM_MODEL }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip
      - name: Install static dependencies
        if: inputs.mode == 'static'
        run: python -m pip install -e '.[dev,sqlite]'
      - name: Install live dependencies
        if: inputs.mode == 'full-live'
        shell: bash
        run: |
          case "${{ inputs.provider }}" in
            openai) python -m pip install -e '.[dev,sqlite,openai]' ;;
            google) python -m pip install -e '.[dev,sqlite,google]' ;;
            anthropic) python -m pip install -e '.[dev,sqlite,anthropic]' ;;
          esac
      - run: make lint
      - run: make typecheck
      - run: make test
      - name: Verify selected provider secret exists
        if: inputs.mode == 'full-live'
        shell: bash
        run: |
          case "${{ inputs.provider }}" in
            openai) test -n "$OPENAI_API_KEY" ;;
            google) test -n "$GEMINI_API_KEY" ;;
            anthropic) test -n "$ANTHROPIC_API_KEY" ;;
          esac
      - name: Run live scenarios
        if: inputs.mode == 'full-live'
        run: make run-scenarios
      - name: Validate metrics
        if: inputs.mode == 'full-live'
        run: make grade-local
```

Do not add `push`, `pull_request`, or `schedule` anywhere in the workflow.

- [ ] **Step 3: Add `python-dotenv` to core dependencies**

In `pyproject.toml`, add:

```toml
"python-dotenv>=1.0",
```

inside `[project].dependencies` so `.env` loading is available regardless of provider extra.

- [ ] **Step 4: Verify the workflow is structurally safe**

Run locally if available:

```bash
python - <<'PY'
from pathlib import Path
text = Path('.github/workflows/ci.yml').read_text()
assert 'workflow_dispatch:' in text
assert 'pull_request:' not in text
assert '\n  push:' not in text
assert 'schedule:' not in text
assert 'cancel-in-progress: true' in text
print('manual-ci contract: ok')
PY
```

Expected: `manual-ci contract: ok`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml
git commit -m "ci: make validation manual only"
```

---

### Task 2: Add Required State Fields and Implement Routing Tables

**Files:**
- Modify: `src/langgraph_agent_lab/state.py`
- Modify: `src/langgraph_agent_lab/routing.py`
- Create: `tests/test_student_state_routing.py`

**Interfaces:**
- Consumes: `AgentState`, `ApprovalDecision`, `Route`.
- Produces: `evaluation_result`, `pending_question`, `proposed_action`, `approval`; four routing helpers returning registered node names.

- [ ] **Step 1: Write failing state/routing tests**

Create `tests/test_student_state_routing.py`:

```python
from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from langgraph_agent_lab.state import AgentState


def test_required_current_value_fields_are_declared() -> None:
    annotations = AgentState.__annotations__
    assert "evaluation_result" in annotations
    assert "pending_question" in annotations
    assert "proposed_action" in annotations
    assert "approval" in annotations


def test_classify_routing_table() -> None:
    assert route_after_classify({"route": "simple"}) == "answer"
    assert route_after_classify({"route": "tool"}) == "tool"
    assert route_after_classify({"route": "missing_info"}) == "clarify"
    assert route_after_classify({"route": "risky"}) == "risky_action"
    assert route_after_classify({"route": "error"}) == "retry"
    assert route_after_classify({"route": "unknown"}) == "answer"


def test_retry_boundary_is_fail_closed() -> None:
    assert route_after_retry({"attempt": 0, "max_attempts": 1}) == "tool"
    assert route_after_retry({"attempt": 1, "max_attempts": 1}) == "dead_letter"
    assert route_after_retry({"attempt": 2, "max_attempts": 1}) == "dead_letter"


def test_evaluate_and_approval_routing() -> None:
    assert route_after_evaluate({"evaluation_result": "needs_retry"}) == "retry"
    assert route_after_evaluate({"evaluation_result": "success"}) == "answer"
    assert route_after_approval({"approval": {"approved": True}}) == "tool"
    assert route_after_approval({"approval": {"approved": False}}) == "clarify"
```

- [ ] **Step 2: Run tests and verify they fail on TODO implementation**

```bash
pytest tests/test_student_state_routing.py tests/test_routing.py -q
```

Expected: failures from missing fields and/or `NotImplementedError`.

- [ ] **Step 3: Add state fields as overwrite/current-value fields**

Add to `AgentState` without `Annotated[..., add]`:

```python
evaluation_result: str
pending_question: str
proposed_action: str
approval: dict[str, object]
```

Keep only `messages`, `tool_results`, `errors`, and `events` as append reducers.

- [ ] **Step 4: Implement the routing helpers exactly**

```python
def route_after_classify(state: AgentState) -> str:
    return {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }.get(str(state.get("route", "")), "answer")


def route_after_evaluate(state: AgentState) -> str:
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 0))
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    approval = state.get("approval") or {}
    return "tool" if bool(approval.get("approved")) else "clarify"
```

- [ ] **Step 5: Run state/routing tests**

```bash
pytest tests/test_student_state_routing.py tests/test_routing.py tests/test_state.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/state.py src/langgraph_agent_lab/routing.py tests/test_student_state_routing.py
git commit -m "feat: implement state and routing contracts"
```

---

### Task 3: Implement Deterministic Core Nodes and Retry Semantics

**Files:**
- Modify: `src/langgraph_agent_lab/nodes.py`
- Create: `tests/test_student_nodes.py`

**Interfaces:**
- Consumes: `AgentState`, `ApprovalDecision`, `make_event`.
- Produces: tool/evaluation/clarification/risky-action/approval/retry/dead-letter/finalize partial updates.

- [ ] **Step 1: Write failing node-contract tests**

Create tests covering mutation safety and retry ownership:

```python
from copy import deepcopy

from langgraph_agent_lab.nodes import (
    approval_node,
    ask_clarification_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)


def test_tool_appends_only_new_result_without_mutating_input() -> None:
    state = {"route": "error", "attempt": 1, "query": "service failed", "tool_results": ["old"]}
    before = deepcopy(state)
    update = tool_node(state)
    assert state == before
    assert len(update["tool_results"]) == 1
    assert "ERROR" in update["tool_results"][0]


def test_retry_is_only_counter_owner() -> None:
    state = {"attempt": 0, "max_attempts": 1, "errors": []}
    update = retry_or_fallback_node(state)
    assert state["attempt"] == 0
    assert update["attempt"] == 1
    assert len(update["errors"]) == 1


def test_mock_approval_is_non_interactive_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LANGGRAPH_INTERRUPT", raising=False)
    update = approval_node({"proposed_action": "refund order"})
    assert update["approval"]["approved"] is True


def test_evaluator_heuristic_detects_error() -> None:
    update = evaluate_node({"tool_results": ["ERROR: transient failure"]})
    assert update["evaluation_result"] == "needs_retry"


def test_terminal_nodes_preserve_route_by_not_returning_it() -> None:
    assert "route" not in dead_letter_node({"attempt": 2, "max_attempts": 2})
    assert "route" not in finalize_node({"route": "error"})


def test_clarification_sets_question_and_final_answer() -> None:
    update = ask_clarification_node({"query": "Can you fix it?"})
    assert update["pending_question"]
    assert update["final_answer"] == update["pending_question"]


def test_risky_action_only_prepares_action() -> None:
    update = risky_action_node({"query": "refund customer"})
    assert update["proposed_action"]
    assert "tool_results" not in update
```

- [ ] **Step 2: Run and verify failures**

```bash
pytest tests/test_student_nodes.py -q
```

Expected: `NotImplementedError` failures.

- [ ] **Step 3: Implement deterministic node behavior**

Use partial updates only. The core logic should follow this shape:

```python
def tool_node(state: AgentState) -> dict:
    attempt = int(state.get("attempt", 0))
    route = str(state.get("route", ""))
    query = str(state.get("query", ""))
    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure at attempt {attempt}"
        event_type = "failed"
    else:
        result = f"Tool completed for request: {query[:80]}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result[:100], metadata={"attempt": attempt})],
    }


def evaluate_node(state: AgentState) -> dict:
    latest = str((state.get("tool_results") or [""])[-1])
    verdict = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": verdict,
        "events": [make_event("evaluate", "completed", verdict)],
    }


def ask_clarification_node(state: AgentState) -> dict:
    query = str(state.get("query", "")).strip()
    approval = state.get("approval") or {}
    if approval and not bool(approval.get("approved")):
        question = "The proposed action was not approved. What alternative outcome would you like?"
    else:
        question = f"What additional details should I use to handle this request: {query or 'your request'}?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    query = str(state.get("query", "")).strip()
    action = f"Proposed risky action requiring approval: {query}"
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "prepared", action[:100])],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    next_attempt = int(state.get("attempt", 0)) + 1
    message = f"Retry attempt {next_attempt} recorded after transient failure"
    return {
        "attempt": next_attempt,
        "errors": [message],
        "events": [make_event("retry", "recorded", message, metadata={"attempt": next_attempt})],
    }


def dead_letter_node(state: AgentState) -> dict:
    attempt = int(state.get("attempt", 0))
    maximum = int(state.get("max_attempts", 0))
    answer = f"Unable to complete the request after {attempt} retry attempts; escalation is required."
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "exhausted", answer, metadata={"max_attempts": maximum})],
    }


def finalize_node(state: AgentState) -> dict:
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
```

For `approval_node`, default to an `ApprovalDecision(approved=True, reviewer="mock-reviewer", comment="auto-approved for non-interactive lab run")` converted with `.model_dump()`; real interrupt behavior is added later in Task 9.

- [ ] **Step 4: Run deterministic node tests**

```bash
pytest tests/test_student_nodes.py tests/test_state.py tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/nodes.py tests/test_student_nodes.py
git commit -m "feat: implement deterministic workflow nodes"
```

---

### Task 4: Add LLM Schemas, `.env` Loading, Structured Classification, and Grounded Answering

**Files:**
- Create: `src/langgraph_agent_lab/schemas.py`
- Modify: `src/langgraph_agent_lab/llm.py`
- Modify: `src/langgraph_agent_lab/nodes.py`
- Create: `tests/test_student_llm_nodes.py`

**Interfaces:**
- Consumes: `get_llm(model: str | None = None, temperature: float = 0.0)` existing factory.
- Produces: `ClassificationDecision`, `EvaluationDecision`, production `classify_node`, production `answer_node`.

- [ ] **Step 1: Write failing fake-LLM tests**

Use deterministic fakes; do not call an external API:

```python
from types import SimpleNamespace

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.schemas import ClassificationDecision


class FakeStructuredLLM:
    def __init__(self, decision):
        self.decision = decision

    def invoke(self, prompt):
        return self.decision


class FakeLLM:
    def __init__(self, decision=None, answer="grounded answer"):
        self.decision = decision
        self.answer = answer
        self.last_prompt = None

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self.decision)

    def invoke(self, prompt):
        self.last_prompt = str(prompt)
        return SimpleNamespace(content=self.answer)


def test_classifier_uses_structured_output(monkeypatch) -> None:
    fake = FakeLLM(ClassificationDecision(route="risky", risk_level="high", reason="side effect"))
    monkeypatch.setattr(nodes, "get_llm", lambda **kwargs: fake)
    update = nodes.classify_node({"query": "Please refund the order and check its status"})
    assert update["route"] == "risky"
    assert update["risk_level"] == "high"


def test_answer_prompt_contains_tool_and_approval_context(monkeypatch) -> None:
    fake = FakeLLM(answer="The lookup completed successfully.")
    monkeypatch.setattr(nodes, "get_llm", lambda **kwargs: fake)
    update = nodes.answer_node({
        "query": "Where is order 123?",
        "tool_results": ["Order 123 is shipped"],
        "approval": {"approved": True, "reviewer": "mock-reviewer", "comment": "ok"},
    })
    assert update["final_answer"] == "The lookup completed successfully."
    assert "Order 123 is shipped" in fake.last_prompt
    assert "Where is order 123?" in fake.last_prompt
```

- [ ] **Step 2: Run and verify failures**

```bash
pytest tests/test_student_llm_nodes.py -q
```

Expected: import/classification/answer failures.

- [ ] **Step 3: Add explicit Pydantic schemas**

Create `schemas.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field


class ClassificationDecision(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"]
    reason: str = Field(min_length=1, max_length=240)


class EvaluationDecision(BaseModel):
    verdict: Literal["success", "needs_retry"]
    reason: str = Field(min_length=1, max_length=240)
```

- [ ] **Step 4: Load `.env` once in `llm.py`**

At module import boundary:

```python
from dotenv import load_dotenv

load_dotenv()
```

Keep provider selection order unchanged: Gemini, then OpenAI, then Anthropic.

- [ ] **Step 5: Implement structured `classify_node`**

Use a prompt that states semantic definitions and exact priority:

```python
CLASSIFICATION_PROMPT = """You classify support tickets for a LangGraph workflow.
Return only the structured schema requested by the caller.
Routes:
- risky: action with side effects or destructive/financial/account changes requiring approval
- tool: lookup or external tool action without risky side effects
- missing_info: request is too vague/incomplete to act safely
- error: request explicitly describes a system/processing failure that should enter retry handling
- simple: informational/help request that can be answered without a tool
Priority when multiple signals appear: risky > tool > missing_info > error > simple.
Set risk_level=high only for risky; otherwise low.
Ticket: {query}
"""


def classify_node(state: AgentState) -> dict:
    query = str(state.get("query", "")).strip()
    llm = get_llm(temperature=0.0)
    decision = llm.with_structured_output(ClassificationDecision).invoke(
        CLASSIFICATION_PROMPT.format(query=query)
    )
    parsed = ClassificationDecision.model_validate(decision)
    return {
        "route": parsed.route,
        "risk_level": parsed.risk_level,
        "events": [make_event("classify", "completed", parsed.reason, metadata={"route": parsed.route})],
    }
```

Do not read `scenario_id` in this function.

- [ ] **Step 6: Implement grounded `answer_node`**

Build a compact prompt from only relevant state:

```python
def answer_node(state: AgentState) -> dict:
    query = str(state.get("query", "")).strip()
    tool_results = list(state.get("tool_results") or [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")
    prompt = (
        "Answer the support request using only the workflow context below. "
        "Do not claim an action occurred if it was rejected or if the context does not prove it.\n"
        f"Query: {query}\n"
        f"Tool results: {tool_results}\n"
        f"Proposed action: {proposed_action}\n"
        f"Approval: {approval}\n"
    )
    response = get_llm(temperature=0.0).invoke(prompt)
    content = str(getattr(response, "content", response)).strip()
    return {
        "final_answer": content,
        "events": [make_event("answer", "completed", "grounded LLM response generated")],
    }
```

- [ ] **Step 7: Run fake-LLM tests and static checks**

```bash
pytest tests/test_student_llm_nodes.py tests/test_student_nodes.py -q
ruff check src tests
mypy src
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/langgraph_agent_lab/llm.py src/langgraph_agent_lab/schemas.py src/langgraph_agent_lab/nodes.py tests/test_student_llm_nodes.py
git commit -m "feat: add structured LLM classification and grounded answers"
```

---

### Task 5: Build and Prove the Eleven-Node LangGraph

**Files:**
- Modify: `src/langgraph_agent_lab/graph.py`
- Create: `tests/test_student_graph.py`

**Interfaces:**
- Consumes: all node functions and four routing helpers.
- Produces: `build_graph(checkpointer: Any | None = None)` returning a compiled graph.

- [ ] **Step 1: Write a graph topology test**

Create `tests/test_student_graph.py`:

```python
from langgraph_agent_lab.graph import build_graph


def test_graph_contains_required_nodes() -> None:
    graph = build_graph(checkpointer=None)
    mermaid = graph.get_graph().draw_mermaid()
    for node in (
        "intake", "classify", "answer", "tool", "evaluate", "clarify",
        "risky_action", "approval", "retry", "dead_letter", "finalize",
    ):
        assert node in mermaid
```

- [ ] **Step 2: Run and verify graph builder fails**

```bash
pytest tests/test_student_graph.py -q
```

Expected: `NotImplementedError` from `build_graph`.

- [ ] **Step 3: Implement graph registration and fixed/conditional edges**

`build_graph()` must register exactly these node names:

```python
builder.add_node("intake", intake_node)
builder.add_node("classify", classify_node)
builder.add_node("answer", answer_node)
builder.add_node("tool", tool_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("clarify", ask_clarification_node)
builder.add_node("risky_action", risky_action_node)
builder.add_node("approval", approval_node)
builder.add_node("retry", retry_or_fallback_node)
builder.add_node("dead_letter", dead_letter_node)
builder.add_node("finalize", finalize_node)
```

Then wire:

```python
builder.add_edge(START, "intake")
builder.add_edge("intake", "classify")
builder.add_conditional_edges("classify", route_after_classify)
builder.add_edge("tool", "evaluate")
builder.add_conditional_edges("evaluate", route_after_evaluate)
builder.add_edge("risky_action", "approval")
builder.add_conditional_edges("approval", route_after_approval)
builder.add_conditional_edges("retry", route_after_retry)
builder.add_edge("answer", "finalize")
builder.add_edge("clarify", "finalize")
builder.add_edge("dead_letter", "finalize")
builder.add_edge("finalize", END)
return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Add fake-LLM route termination tests**

Extend `tests/test_student_graph.py` with a fake LLM that returns a deterministic structured decision and answer. Invoke synthetic cases for `simple`, `tool`, `missing_info`, `risky`, and `error`, then assert every final event trail contains `finalize`. Include a generic error case with `max_attempts=1` and assert no `tool` event occurs after the first retry boundary.

Concrete assertion shape:

```python
assert any(event["node"] == "finalize" for event in result["events"])
assert result["route"] == expected_route
```

For the dead-letter boundary:

```python
nodes = [event["node"] for event in result["events"]]
assert nodes[:3] == ["intake", "classify", "retry"]
assert "dead_letter" in nodes
assert "tool" not in nodes
```

- [ ] **Step 5: Run graph and public routing/state tests**

```bash
pytest tests/test_student_graph.py tests/test_routing.py tests/test_state.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/graph.py tests/test_student_graph.py
git commit -m "feat: wire complete LangGraph workflow"
```

---

### Task 6: Add SQLite Durable Checkpointing and Recovery Evidence

**Files:**
- Modify: `src/langgraph_agent_lab/persistence.py`
- Create: `tests/test_student_persistence.py`
- Modify: `configs/lab.yaml` only after memory path remains green

**Interfaces:**
- Consumes: `build_checkpointer(kind, database_url)`.
- Produces: SQLite saver usable by `build_graph(checkpointer=...)`; restart-style state recovery proof.

- [ ] **Step 1: Write a failing SQLite construction test**

```python
from pathlib import Path

from langgraph_agent_lab.persistence import build_checkpointer


def test_sqlite_checkpointer_can_be_created(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    saver = build_checkpointer("sqlite", str(db))
    assert saver is not None
    assert db.exists()
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_student_persistence.py -q
```

Expected: starter `NotImplementedError`.

- [ ] **Step 3: Implement SQLite saver with WAL**

Use one SQLite connection owned by the saver:

```python
import sqlite3
from pathlib import Path
from typing import Any


def _sqlite_connection(database_url: str | None) -> sqlite3.Connection:
    path = Path(database_url or "outputs/checkpoints.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver(_sqlite_connection(database_url))
    if kind == "postgres":
        raise NotImplementedError("Postgres is not implemented in this submission")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
```

- [ ] **Step 4: Add restart-style recovery test**

Create a SQLite-backed graph, invoke a simple fake-LLM scenario using a fixed `thread_id`, close/drop the first graph/saver reference, construct a second SQLite saver/graph against the same database, and assert `graph.get_state({"configurable": {"thread_id": thread_id}})` returns non-empty state with events from the previous run.

Use a synthetic thread ID such as `thread-recovery-test`; do not rely on a public scenario ID.

- [ ] **Step 5: Add state-history/time-travel evidence test**

On the recovered graph:

```python
history = list(graph.get_state_history({"configurable": {"thread_id": thread_id}}))
assert len(history) >= 2
assert any(snapshot.values.get("events") for snapshot in history)
```

If the installed LangGraph API exposes checkpoint IDs in `snapshot.config`, also invoke a controlled replay using that checkpoint config and assert it yields a valid state without altering the original classified route.

- [ ] **Step 6: Run persistence tests**

```bash
pytest tests/test_student_persistence.py tests/test_student_graph.py -q
```

Expected: PASS.

- [ ] **Step 7: Switch final lab config to SQLite evidence mode**

Update `configs/lab.yaml` to:

```yaml
scenarios_path: data/sample/scenarios.jsonl
checkpointer: sqlite
database_url: outputs/checkpoints.sqlite
report_path: reports/lab_report.md
```

Memory support remains available through `build_checkpointer("memory")` for tests.

- [ ] **Step 8: Commit**

```bash
git add src/langgraph_agent_lab/persistence.py tests/test_student_persistence.py configs/lab.yaml
git commit -m "feat: add durable sqlite checkpoint recovery"
```

---

### Task 7: Instrument Real Metrics and Recovery Semantics

**Files:**
- Modify: `src/langgraph_agent_lab/metrics.py`
- Modify: `src/langgraph_agent_lab/cli.py`
- Create: `tests/test_student_cli.py`

**Interfaces:**
- Consumes: final graph state, expected route, approval requirement, measured `latency_ms`, recovery evidence boolean.
- Produces: truthful `ScenarioMetric` and `MetricsReport` values.

- [ ] **Step 1: Write failing metrics instrumentation tests**

Add assertions that `metric_from_state` accepts measured latency while preserving the old call shape through a default:

```python
from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics


def test_metric_uses_measured_latency_and_real_interrupt_event() -> None:
    state = {
        "scenario_id": "synthetic",
        "route": "risky",
        "final_answer": "done",
        "approval": {"approved": True},
        "events": [
            {"node": "approval", "event_type": "completed"},
            {"node": "approval", "event_type": "interrupted"},
            {"node": "finalize", "event_type": "completed"},
        ],
        "errors": [],
    }
    metric = metric_from_state(state, "risky", True, latency_ms=37)
    assert metric.latency_ms == 37
    assert metric.interrupt_count == 1


def test_resume_success_is_explicit_evidence() -> None:
    metric = metric_from_state({
        "scenario_id": "x", "route": "simple", "final_answer": "ok", "events": [], "errors": []
    }, "simple", False)
    assert summarize_metrics([metric], resume_success=True).resume_success is True
```

- [ ] **Step 2: Run and verify signature/semantics fail**

```bash
pytest tests/test_metrics.py tests/test_student_cli.py -q
```

Expected: failure because latency/resume parameters are not implemented.

- [ ] **Step 3: Extend metric helpers compatibly**

Change signatures to:

```python
def metric_from_state(
    state: dict[str, Any],
    expected_route: str,
    approval_required: bool,
    latency_ms: int = 0,
) -> ScenarioMetric:
    ...


def summarize_metrics(
    items: list[ScenarioMetric],
    resume_success: bool = False,
) -> MetricsReport:
    ...
```

Count retry visits from events where `node == "retry"`. Count real interrupts only when the approval event explicitly uses an interrupt event type; an ordinary approval visit is not an interrupt.

- [ ] **Step 4: Measure each scenario invocation in `cli.py`**

Use:

```python
from time import perf_counter

started = perf_counter()
final_state = graph.invoke(state, config=run_config)
latency_ms = max(1, round((perf_counter() - started) * 1000))
metrics.append(
    metric_from_state(
        final_state,
        scenario.expected_route.value,
        scenario.requires_approval,
        latency_ms=latency_ms,
    )
)
```

- [ ] **Step 5: Compute durable recovery evidence before summarizing**

When `checkpointer == "sqlite"`, use the same SQLite path/thread evidence to confirm at least one finished scenario can be read through a fresh saver/graph instance. Pass that verified boolean into `summarize_metrics(metrics, resume_success=recovery_verified)`.

If recovery verification fails, set `resume_success=False`; never infer true only from configuration.

- [ ] **Step 6: Run metrics/CLI tests**

```bash
pytest tests/test_metrics.py tests/test_student_cli.py tests/test_student_persistence.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/langgraph_agent_lab/metrics.py src/langgraph_agent_lab/cli.py tests/test_student_cli.py
git commit -m "feat: instrument truthful runtime metrics"
```

---

### Task 8: Implement Evidence-Based Markdown Report

**Files:**
- Modify: `src/langgraph_agent_lab/report.py`
- Create: `tests/test_student_report.py`

**Interfaces:**
- Consumes: `MetricsReport` only for numeric scenario/aggregate values.
- Produces: deterministic Markdown containing required architecture/state/failure/recovery/extension sections.

- [ ] **Step 1: Write failing report contract test**

```python
from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report


def test_report_contains_required_evidence_sections() -> None:
    report = MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=1,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=[ScenarioMetric(
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
        )],
    )
    text = render_report(report)
    for heading in (
        "Architecture", "State Schema", "Scenario Results", "Failure Analysis",
        "Persistence and Recovery", "Extension Work", "Improvement Plan",
    ):
        assert heading in text
    assert "synthetic" in text
    assert "25" in text
```

- [ ] **Step 2: Run and verify report TODO fails**

```bash
pytest tests/test_student_report.py -q
```

Expected: `NotImplementedError` from `render_report`.

- [ ] **Step 3: Implement deterministic renderer**

Build Markdown from `MetricsReport.model_dump()`; numeric scenario values must come directly from the report object. Include:

```markdown
# LangGraph Agentic Orchestration Lab Report

## Student
## Architecture
## State Schema
## Aggregate Metrics
## Scenario Results
## Failure Analysis
### Failure mode 1 — transient tool failure and bounded retry
### Failure mode 2 — risky side effect and approval gate
## Persistence and Recovery
## Extension Work
## Improvement Plan
```

The scenario table columns must include scenario ID, expected route, actual route, success, nodes, retries, interrupts, approval observed, and latency ms.

Persistence section must render `metrics.resume_success` explicitly. Do not state that real HITL occurred when `total_interrupts == 0`.

- [ ] **Step 4: Run report and metrics tests**

```bash
pytest tests/test_student_report.py tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/report.py tests/test_student_report.py
git commit -m "feat: generate evidence-based lab report"
```

---

### Task 9: Upgrade Evaluator to LLM-as-Judge with Deterministic Fallback and Cost Guard

**Files:**
- Modify: `src/langgraph_agent_lab/nodes.py`
- Modify: `tests/test_student_llm_nodes.py`

**Interfaces:**
- Consumes: latest tool result and `EvaluationDecision`.
- Produces: `evaluation_result` compatible with existing routing; audit metadata explaining primary or fallback decision.

- [ ] **Step 1: Add failing judge/fallback tests**

```python
class RaisingStructuredLLM:
    def invoke(self, prompt):
        raise RuntimeError("provider unavailable")


class JudgeFakeLLM(FakeLLM):
    def with_structured_output(self, schema):
        if schema.__name__ == "EvaluationDecision":
            return FakeStructuredLLM({"verdict": "success", "reason": "tool result is usable"})
        return super().with_structured_output(schema)


def test_evaluator_uses_structured_judge(monkeypatch) -> None:
    fake = JudgeFakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda **kwargs: fake)
    update = nodes.evaluate_node({"tool_results": ["Order found"]})
    assert update["evaluation_result"] == "success"


def test_evaluator_falls_back_deterministically(monkeypatch) -> None:
    class BrokenLLM:
        def with_structured_output(self, schema):
            return RaisingStructuredLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda **kwargs: BrokenLLM())
    update = nodes.evaluate_node({"tool_results": ["ERROR: timeout"]})
    assert update["evaluation_result"] == "needs_retry"
    assert update.get("errors")
```

- [ ] **Step 2: Run and verify fallback test fails**

```bash
pytest tests/test_student_llm_nodes.py -q
```

- [ ] **Step 3: Implement one-call judge with cost guard**

`evaluate_node` makes at most one judge call per node visit. It never invokes itself and never retries the LLM internally. Validate through `EvaluationDecision`; on any exception use the deterministic `ERROR` heuristic and append an auditable error entry.

Core shape:

```python
latest = str((state.get("tool_results") or [""])[-1])
try:
    decision = get_llm(temperature=0.0).with_structured_output(EvaluationDecision).invoke(
        f"Evaluate this tool result. Return success if usable, needs_retry only for a real failure. Result: {latest}"
    )
    parsed = EvaluationDecision.model_validate(decision)
    verdict = parsed.verdict
    reason = parsed.reason
    errors = []
except Exception as exc:
    verdict = "needs_retry" if "ERROR" in latest.upper() else "success"
    reason = "deterministic fallback after evaluator failure"
    errors = [f"evaluate fallback: {type(exc).__name__}"]
```

Return `errors` only when non-empty; append one evaluate event with reason/verdict metadata.

- [ ] **Step 4: Run LLM-node and retry tests**

```bash
pytest tests/test_student_llm_nodes.py tests/test_student_nodes.py tests/test_student_graph.py -q
```

Expected: PASS and retry remains bounded.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/nodes.py tests/test_student_llm_nodes.py
git commit -m "feat: add llm judge with bounded fallback"
```

---

### Task 10: Add Feature-Gated Real HITL Without Breaking Default Runs

**Files:**
- Modify: `src/langgraph_agent_lab/nodes.py`
- Modify: `tests/test_student_nodes.py`
- Modify: `tests/test_student_graph.py`

**Interfaces:**
- Consumes: `LANGGRAPH_INTERRUPT` environment variable and `proposed_action`.
- Produces: mock `ApprovalDecision` by default; real `interrupt()` payload/decision only when flag is true.

- [ ] **Step 1: Add failing feature-flag test**

Keep the existing default mock test and add a test that monkeypatches the LangGraph interrupt function when the flag is true:

```python
def test_real_hitl_mode_uses_interrupt(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    observed = {}

    def fake_interrupt(payload):
        observed.update(payload)
        return {"approved": False, "reviewer": "human", "comment": "choose another action"}

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt, raising=False)
    update = nodes.approval_node({"proposed_action": "delete account"})
    assert observed["proposed_action"] == "delete account"
    assert update["approval"]["approved"] is False
    assert update["events"][0]["event_type"] in {"interrupted", "resumed"}
```

- [ ] **Step 2: Run and verify feature path fails**

```bash
pytest tests/test_student_nodes.py -q
```

- [ ] **Step 3: Implement feature-gated interrupt**

Use a helper:

```python
def _interrupt_enabled() -> bool:
    return os.getenv("LANGGRAPH_INTERRUPT", "").strip().lower() in {"1", "true", "yes", "on"}
```

Inside `approval_node`, if disabled, keep mock approval. If enabled:

```python
from langgraph.types import interrupt

raw = interrupt({"proposed_action": proposed_action, "question": "Approve this action?"})
decision = ApprovalDecision.model_validate(raw)
```

Return the serialized decision and an approval event indicating the interrupt/resume path. Do not call `tool_node` from `approval_node`.

- [ ] **Step 4: Verify default tests remain non-interactive**

```bash
LANGGRAPH_INTERRUPT=false pytest tests/test_student_nodes.py tests/test_student_graph.py -q
```

Expected: PASS without waiting for input.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_agent_lab/nodes.py tests/test_student_nodes.py tests/test_student_graph.py
git commit -m "feat: add feature-gated human approval interrupt"
```

---

### Task 11: Add Mermaid Export and Controlled State-History CLI Evidence

**Files:**
- Modify: `src/langgraph_agent_lab/cli.py`
- Modify: `tests/test_student_cli.py`

**Interfaces:**
- Consumes: compiled graph and a configured SQLite checkpointer.
- Produces: non-destructive CLI evidence commands for graph architecture and state history.

- [ ] **Step 1: Add CLI tests using Typer `CliRunner`**

Test a graph-export command writes Mermaid text containing the eleven node names. Test a history command against a temporary SQLite database/thread returns at least one checkpoint after a synthetic run.

Command contracts:

```text
agent-lab export-graph --output outputs/graph.mmd
agent-lab state-history --database outputs/checkpoints.sqlite --thread-id <thread>
```

- [ ] **Step 2: Run tests and verify commands do not exist**

```bash
pytest tests/test_student_cli.py -q
```

- [ ] **Step 3: Implement `export-graph`**

```python
@app.command("export-graph")
def export_graph(output: Annotated[Path, typer.Option("--output")]) -> None:
    graph = build_graph(checkpointer=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    typer.echo(f"Wrote graph Mermaid to {output}")
```

- [ ] **Step 4: Implement read-only `state-history`**

Build a SQLite checkpointer/graph, create `config = {"configurable": {"thread_id": thread_id}}`, iterate `graph.get_state_history(config)`, and print checkpoint IDs plus compact state facts (`route`, `attempt`, finalization presence). Do not print prompts, secrets, or complete sensitive state payloads.

- [ ] **Step 5: Run CLI tests and export topology**

```bash
pytest tests/test_student_cli.py -q
python -m langgraph_agent_lab.cli export-graph --output outputs/graph.mmd
grep -E 'intake|classify|approval|dead_letter|finalize' outputs/graph.mmd
```

Expected: all required representative node names found.

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_agent_lab/cli.py tests/test_student_cli.py outputs/graph.mmd
git commit -m "feat: add graph and checkpoint evidence commands"
```

---

### Task 12: Run Core Quality Gates Before Any Live API Scenario Run

**Files:**
- Modify only files exposed by failures; do not alter public contracts to make tests pass.

**Interfaces:**
- Consumes: implementation from Tasks 1-11.
- Produces: static/unit green baseline with zero required API calls.

- [ ] **Step 1: Run Ruff**

```bash
make lint
```

Expected: exit 0.

- [ ] **Step 2: Run MyPy**

```bash
make typecheck
```

Expected: exit 0.

- [ ] **Step 3: Run the complete pytest suite with no provider key intentionally exposed**

```bash
GEMINI_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY= pytest -q
```

Expected: all deterministic/unit tests pass; public graph smoke may skip if it is explicitly designed to skip without a provider key.

- [ ] **Step 4: Scan required implementation paths for unfinished student code**

```bash
rg -n 'TODO\(student\)|NotImplementedError' src tests
```

Expected: no required core TODO/NotImplementedError remains. A deliberate Postgres-not-implemented branch is acceptable only if rewritten without the starter `TODO(student)` marker and clearly labeled as unsupported optional functionality.

- [ ] **Step 5: Scan for accidental secrets and hidden grading content**

```bash
rg -n '(sk-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{20,}|ANTHROPIC_API_KEY\s*=.+|OPENAI_API_KEY\s*=.+|GEMINI_API_KEY\s*=.+)' . --glob '!*.md'
test ! -d data/grading
```

Expected: no secret values; no `data/grading` directory.

- [ ] **Step 6: Commit any gate-only fixes**

```bash
git add -A
git commit -m "fix: satisfy static quality gates"
```

Skip this commit if no fixes were required.

---

### Task 13: Perform One Deliberate Live Validation Run and Generate Final Evidence

**Files:**
- Generate/update: `outputs/metrics.json`
- Generate/update: `reports/lab_report.md`
- Generate/update: `outputs/checkpoints.sqlite` locally/CI artifact as appropriate; do not commit database if repository policy excludes it.

**Interfaces:**
- Consumes: exactly one configured live provider secret.
- Produces: real scenario metrics/report/recovery evidence matching the official sample scenarios.

- [ ] **Step 1: Select exactly one provider**

Preferred repository-secret setup is one of:

```text
OPENAI_API_KEY
GEMINI_API_KEY
ANTHROPIC_API_KEY
```

Do not expose multiple keys to the process. If OpenAI is selected, install/run with the OpenAI extra only; analogous rule for Gemini/Anthropic.

- [ ] **Step 2: Run public graph smoke once with the selected provider**

```bash
pytest tests/test_graph_smoke.py -q
```

Expected: the five public route cases pass and every path contains a finalize event.

- [ ] **Step 3: Run all seven sample scenarios once**

```bash
make run-scenarios
```

Expected: writes `outputs/metrics.json` and `reports/lab_report.md`.

- [ ] **Step 4: Validate metrics**

```bash
make grade-local
```

Expected: schema valid with at least six scenarios; target sample run is seven scenarios.

- [ ] **Step 5: Inspect semantic evidence rather than only exit codes**

Use a script:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('outputs/metrics.json').read_text())
assert p['total_scenarios'] >= 7
assert p['success_rate'] == 1.0
assert p['resume_success'] is True
assert any(x['retry_count'] > 0 for x in p['scenario_metrics'])
assert any(x['approval_required'] and x['approval_observed'] for x in p['scenario_metrics'])
assert all(x['latency_ms'] > 0 for x in p['scenario_metrics'])
print('evidence metrics: ok')
PY
```

- [ ] **Step 6: Verify report/metrics consistency**

Confirm the report scenario count, route results, retries, recovery statement, and latency values are rendered from `outputs/metrics.json`; do not manually edit contradictory numbers.

- [ ] **Step 7: Commit final text/JSON evidence**

```bash
git add outputs/metrics.json reports/lab_report.md
git commit -m "docs: add final lab execution evidence"
```

Do not commit API keys or a SQLite database containing unnecessary runtime state.

---

### Task 14: Perform One Manual GitHub Actions `full-live` Verification and Review Artifacts

**Files:**
- No source change unless CI exposes a genuine defect.

**Interfaces:**
- Consumes: repository Actions secret + manual workflow dispatch.
- Produces: one deliberate CI run proving reproducibility on Ubuntu/Python 3.11 without per-commit spam.

- [ ] **Step 1: Add the selected provider secret in GitHub UI**

Repository path:

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

Add exactly the selected secret name/value. Optionally add `LLM_MODEL` as an Actions variable, not a secret, if a model override is required.

- [ ] **Step 2: Run `static` manually once**

In GitHub Actions, choose `manual-ci`, `Run workflow`, mode `static`.

Expected: install, Ruff, MyPy, and pytest pass without using a live provider.

- [ ] **Step 3: Run `full-live` manually once**

Select the same provider whose secret was configured.

Expected: static gates + provider preflight + live scenario run + metrics validation all pass.

Do not repeatedly rerun a successful workflow. If a run fails, inspect the failing step/log first and fix the cause before dispatching another run.

- [ ] **Step 4: Confirm anti-spam properties after the run**

Verify `.github/workflows/ci.yml` still has only `workflow_dispatch`, one Python version, no matrix, and `cancel-in-progress: true`.

- [ ] **Step 5: Commit only if CI-specific fixes were necessary**

```bash
git add -A
git commit -m "fix: address manual ci validation findings"
```

Skip if no changes were required.

---

### Task 15: Final Submission Hygiene and Review Gate

**Files:**
- Entire branch diff.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: review-ready branch with no hidden/test-cheating/secrets/CI-spam regressions.

- [ ] **Step 1: Run the official final command sequence**

```bash
make lint
make typecheck
make test
make run-scenarios
make grade-local
git status
git diff --check
```

Expected: all commands exit 0; `git diff --check` emits no whitespace errors.

- [ ] **Step 2: Verify submission checklist mechanically**

```bash
python - <<'PY'
from pathlib import Path
required = [
    'src/langgraph_agent_lab/state.py',
    'src/langgraph_agent_lab/nodes.py',
    'src/langgraph_agent_lab/routing.py',
    'src/langgraph_agent_lab/graph.py',
    'src/langgraph_agent_lab/llm.py',
    'src/langgraph_agent_lab/persistence.py',
    'src/langgraph_agent_lab/metrics.py',
    'src/langgraph_agent_lab/report.py',
    'outputs/metrics.json',
    'reports/lab_report.md',
]
for path in required:
    assert Path(path).exists(), path
workflow = Path('.github/workflows/ci.yml').read_text()
assert 'workflow_dispatch:' in workflow
assert 'pull_request:' not in workflow
assert '\n  push:' not in workflow
assert not Path('data/grading').exists()
print('submission structure: ok')
PY
```

- [ ] **Step 3: Review branch diff for hard-coding and report overclaims**

Search for sample IDs/query literals in production source:

```bash
rg -n 'S0[1-7]_|How do I reset my password|Refund this customer|Can you fix it\?' src
```

Expected: no matches in production source.

- [ ] **Step 4: Ensure branch contains no accidental runtime secret/database artifacts**

```bash
git status --short
```

If `outputs/checkpoints.sqlite`, `*.sqlite-wal`, or `*.sqlite-shm` appear as tracked/unwanted files, remove them from the commit and add the appropriate ignore rule without deleting the final metrics/report evidence.

- [ ] **Step 5: Final commit if hygiene changes were required**

```bash
git add -A
git commit -m "chore: finalize Day 23 submission hygiene"
```

- [ ] **Step 6: Request code review before merge**

Review must specifically check:

```text
1. HTML/Codelab checklist coverage
2. structured LLM classifier and grounded answer are genuine
3. no exact-sample hard-coding
4. bounded retry including max_attempts=1 dead-letter
5. risky side effect cannot occur before approval
6. SQLite recovery evidence is real
7. metrics/report claims match instrumentation
8. bonus extensions have tests/evidence
9. CI remains manual-only
10. no secret or hidden grading data
```

Only after this review should the branch be merged into `main`.
