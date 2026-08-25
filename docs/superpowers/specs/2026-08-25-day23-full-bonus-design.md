# Day 23 Full Bonus Design

**Date:** 2026-08-25
**Repository:** `tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh`
**Branch:** `feat/day23-full-score-bonus`
**Baseline implementation SHA:** `d6ea5833e59c5ff349eda56b9139e07277ea30c0`
**Baseline evidence head before this extension:** `1a7ae142ec31c4b8f2df336987deb50637e3392f`

## 1. Objective

Extend the already validated Day 23 LangGraph submission so that every optional extension listed in the official VLearn HTML has a concrete implementation plus testable evidence, while preserving the core 100-point contract unchanged.

The official extension list is:

1. LLM-as-judge.
2. Real HITL interrupt/resume.
3. SQLite/Postgres durable recovery.
4. Time travel.
5. Parallel fan-out with `Send()`.
6. Streamlit UI.
7. Mermaid graph export.

The current submission already has validated LLM-as-judge, SQLite durable recovery, and compiled Mermaid export. Real HITL is implemented behind a feature flag but lacks a true runtime interrupt/resume proof. State-history inspection exists but time travel does not yet provide replay/fork semantics. Parallel `Send()` and Streamlit UI are not yet implemented.

The extension work therefore completes the remaining proof surface rather than replacing the core agent.

## 2. Non-negotiable core invariants

The full-bonus work must not weaken any behavior that already passed the manual `full-live` gate.

The required core graph remains exactly eleven registered nodes:

- `intake`
- `classify`
- `tool`
- `evaluate`
- `answer`
- `clarify`
- `risky_action`
- `approval`
- `retry`
- `dead_letter`
- `finalize`

The following invariants remain unchanged:

- every terminal path reaches `finalize` before `END`;
- only the retry node increments `attempt`;
- retry remains bounded by `attempt < max_attempts`;
- risky side effects cannot execute before approval;
- rejected approval goes to clarification;
- structured LLM classification and grounded LLM answers remain the primary behavior;
- sample scenario IDs and exact sample queries are not used as production decision rules;
- state remains serializable;
- append reducers are used only for append-only history fields;
- the default CI path remains non-interactive;
- no hidden grading data or secrets are added to Git.

The full-bonus implementation must not add bonus-only nodes to `build_graph()`. Bonus graphs or demos are isolated from the required eleven-node graph.

## 3. Architecture strategy: isolated bonus pack

Bonus functionality is implemented as adapters, CLI commands, evidence helpers, and one separate bonus graph. This keeps the required graph stable and lets every extension be tested independently.

The extension surface is divided into four units:

### 3.1 HITL evidence runner

A dedicated helper/CLI command executes a risky request with `LANGGRAPH_INTERRUPT=true` using a durable SQLite checkpointer and a stable `thread_id`.

Expected flow:

```text
invoke risky request
  -> risky_action
  -> approval
  -> interrupt payload returned
  -> checkpoint persists
  -> resume with Command(resume=reviewer decision)
  -> same thread_id
  -> tool
  -> evaluate
  -> answer
  -> finalize
```

The runtime proof must record that an actual interrupt was observed, the same thread was resumed, the reviewer decision was stored, and the resumed graph reached `finalize`.

The default mock approval path remains unchanged when `LANGGRAPH_INTERRUPT` is false.

### 3.2 Time-travel service

State-history inspection is upgraded into executable time travel. A small service layer exposes operations over a SQLite-backed compiled graph:

- list checkpoints for a thread;
- select a checkpoint by `checkpoint_id`;
- replay from a selected checkpoint without rewriting prior history;
- fork from a selected checkpoint after an explicit state update, creating a new trajectory.

CLI operations should be explicit and auditable. A fork must not silently overwrite the original thread history.

The preferred interface is a new command such as:

```text
agent-lab time-travel \
  --database outputs/checkpoints.sqlite \
  --thread-id <thread> \
  --checkpoint-id <checkpoint> \
  --mode replay|fork
```

For `fork`, explicit override fields are supplied through safe CLI options or a small JSON object. The implementation must reject an unknown checkpoint instead of falling back to the latest checkpoint.

### 3.3 Parallel `Send()` bonus graph

Parallel fan-out is implemented in a separate module, not the core graph.

The bonus graph uses LangGraph `Send()` for map-style dispatch:

```text
START
  -> plan_tasks
     -> Send(independent_tool, task A)
     -> Send(independent_tool, task B)
     -> Send(independent_tool, task C)
  -> aggregate
  -> END
```

State uses a reducer-managed results list so parallel workers can merge output deterministically. The test fixture uses independent deterministic mock tasks and verifies that all requested tasks are represented exactly once in the aggregate regardless of execution order.

This extension demonstrates the LangGraph primitive without introducing concurrency into the required support-ticket graph.

### 3.4 Streamlit evidence UI

A small optional Streamlit UI is added as a presentation layer over the existing graph and state contracts. It must not contain separate agent logic.

The UI shows:

- ticket/query;
- selected route and risk level;
- proposed action;
- approval/rejection state;
- final answer;
- event trail;
- thread/checkpoint information when available.

The UI must never render API keys, environment dumps, or hidden grading content.

Streamlit is an optional dependency, for example an `ui` extra, so normal core install/tests do not need UI packages.

A smoke test verifies that the UI module imports and that its pure rendering/view-model helper can transform a representative state without starting a server.

## 4. Extension-specific behavior

### 4.1 LLM-as-judge

Existing behavior is retained:

- one structured evaluation call per tool result when a provider is configured;
- `success` or `needs_retry` verdict;
- short reason;
- deterministic fallback on provider/schema failure;
- no internal recursive judge loop.

Full-bonus evidence should explicitly mark this extension as implemented and reference the live scenario evidence plus student tests.

### 4.2 Real HITL interrupt/resume

The evidence runner must use LangGraph `Command(resume=...)` against a graph compiled with a durable checkpointer.

Acceptance evidence:

- actual interrupt payload captured;
- `real_interrupt=true` recorded in the approval event after resume;
- same stable `thread_id` used before and after resume;
- supplied reviewer identity/comment persisted;
- risky tool execution occurs only after positive approval;
- final state reaches `finalize`;
- a rejection case routes to clarification without tool side effect.

At least one automated test must exercise the resume contract without requiring a human at CI runtime.

### 4.3 Durable recovery

Existing SQLite restart-style recovery remains the default durable proof. Postgres is not required because the official extension wording allows SQLite/Postgres; implementing both would add operational risk without strengthening the grading contract.

The evidence pack must continue to report `resume_success=true` only from a fresh SQLite-backed graph instance reading a previously completed stable thread.

### 4.4 Time travel

Time travel must prove both supported semantics:

- **replay:** re-execute from an older checkpoint under the checkpoint contract;
- **fork:** create a new branch from an older checkpoint after an explicit state modification.

Evidence should include original checkpoint ID, selected mode, resulting thread/trajectory identifier, and a compact confirmation that the resulting trajectory reached a terminal state.

### 4.5 Parallel fan-out with `Send()`

The bonus graph must use the actual `Send` primitive, not `asyncio.gather()` or a manually simulated list loop.

Evidence must demonstrate:

- more than one dispatched task;
- one worker result per task;
- deterministic aggregate membership;
- no mutation of the core eleven-node graph.

### 4.6 Streamlit UI

The UI is intentionally small and grading-oriented. It is not a production customer-support frontend.

Evidence includes:

- optional dependency declared;
- import/smoke test;
- view model or rendering helper covers the required fields;
- report documents how to launch the UI;
- secret-safety check confirms no key values are displayed.

### 4.7 Mermaid export

Existing compiled-graph Mermaid export remains unchanged. Final evidence still verifies that all eleven required core nodes are present in the exported graph.

## 5. Evidence model

Add `outputs/bonus_evidence.json` as machine-readable evidence for extension status.

Recommended shape:

```json
{
  "llm_as_judge": {"implemented": true, "verified": true},
  "real_hitl": {
    "implemented": true,
    "verified": true,
    "interrupt_observed": true,
    "same_thread_id": true,
    "resume_success": true
  },
  "durable_recovery": {"implemented": true, "verified": true},
  "time_travel": {
    "implemented": true,
    "verified": true,
    "replay_verified": true,
    "fork_verified": true
  },
  "parallel_send": {
    "implemented": true,
    "verified": true,
    "task_count": 3,
    "result_count": 3
  },
  "streamlit_ui": {"implemented": true, "verified": true},
  "mermaid_export": {"implemented": true, "verified": true}
}
```

The exact schema may be represented by Pydantic models so CI can validate semantics rather than only checking file existence.

The file must be generated by executable verification logic. It must not be hand-edited to claim a feature that the verification path did not exercise.

## 6. Report changes

`reports/lab_report.md` gains a compact extension matrix with:

- extension name;
- baseline state;
- implementation change;
- verification method;
- runtime evidence;
- limitations.

The report must distinguish:

- core success metrics from `outputs/metrics.json`;
- actual HITL interrupt evidence from ordinary mock approval;
- durable recovery from time-travel replay/fork;
- core graph from separate bonus parallel graph;
- UI smoke evidence from a full browser end-to-end test.

No capability may be described as verified unless the corresponding generated evidence is true.

## 7. Expected file changes

Likely new files:

- `src/langgraph_agent_lab/bonus_evidence.py`
- `src/langgraph_agent_lab/bonus_parallel.py`
- `src/langgraph_agent_lab/time_travel.py`
- `src/langgraph_agent_lab/ui.py`
- `tests/test_student_hitl_resume.py`
- `tests/test_student_time_travel.py`
- `tests/test_student_parallel_send.py`
- `tests/test_student_ui.py`

Likely modified files:

- `src/langgraph_agent_lab/cli.py`
- `src/langgraph_agent_lab/report.py`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- possibly `.gitignore` for new transient evidence/database files if required.

`src/langgraph_agent_lab/graph.py` and the required routing topology should remain unchanged unless a narrowly scoped compatibility fix is proven necessary by tests.

## 8. Testing strategy

Implementation follows TDD for each remaining extension.

### HITL tests

- graph interrupts for risky approval when feature flag is enabled;
- positive `Command(resume=...)` reaches the tool only after approval;
- rejected resume reaches clarification;
- thread ID is unchanged across interrupt/resume;
- evidence writer records actual interrupt semantics.

### Time-travel tests

- checkpoint listing is non-empty after a persisted run;
- unknown checkpoint fails clearly;
- replay from an older checkpoint succeeds;
- fork applies the requested state update;
- original checkpoint/history remains readable after fork.

### Parallel `Send()` tests

- planner returns actual `Send` objects;
- at least three independent tasks fan out;
- reducer combines one result per task;
- aggregate is deterministic by task identity;
- core graph node list remains eleven.

### UI tests

- optional UI module imports when the `ui` extra is installed;
- pure view-model transformation produces required fields;
- no environment secret values are included in the view model;
- Streamlit server startup is not required for unit tests.

### Regression tests

The existing routing, node, graph, persistence, metrics, CLI, report, lint, and MyPy tests remain mandatory.

## 9. Manual CI design

The workflow remains `workflow_dispatch` only with the current anti-spam controls.

The final `full-live` sequence becomes:

1. install selected live provider, SQLite, development dependencies, and UI extra;
2. verify the selected provider secret exists;
3. Ruff lint;
4. MyPy typecheck;
5. offline unit tests;
6. submission hygiene and `git diff --check`;
7. live core graph smoke;
8. live seven-scenario run;
9. core metrics validation;
10. real HITL interrupt/resume verification using durable SQLite and non-interactive programmed reviewer decisions;
11. time-travel replay/fork verification;
12. parallel `Send()` verification;
13. Streamlit import/view-model smoke verification;
14. compiled Mermaid export;
15. generate `outputs/bonus_evidence.json`;
16. semantic validation of core + bonus evidence;
17. upload grading evidence.

The workflow must not wait for manual reviewer input. HITL is proven by programmatically resuming the interrupted graph with an explicit reviewer payload, which tests the real LangGraph interrupt/resume mechanism without making CI interactive.

Artifact upload should include:

- `outputs/metrics.json`;
- `outputs/graph.mmd`;
- `outputs/bonus_evidence.json`;
- `reports/lab_report.md`.

SQLite databases remain excluded from uploaded grading artifacts and Git.

## 10. Security and hygiene

- Repository secret values remain step-scoped to live steps.
- No secret value is written to JSON, Markdown, SQLite evidence text, screenshots, or logs.
- UI code reads only the state/evidence objects necessary to display the workflow.
- `data/grading` remains absent.
- Public tests remain unmodified.
- Bonus tests must not encode hidden/sample-specific grading logic.
- Evidence files contain no API responses that could leak credentials or provider metadata unnecessarily.

## 11. Acceptance criteria

The full-bonus implementation is accepted when all of the following are true in one fresh manual `full-live` run:

### Core

- all previously green core gates still pass;
- seven sample scenarios remain successful;
- core success rate remains `1.0`;
- bounded retry/dead-letter behavior remains verified;
- risky approval behavior remains verified;
- durable recovery remains true;
- compiled Mermaid still contains all eleven core nodes.

### Extensions

- LLM-as-judge is marked implemented/verified from executable evidence;
- real HITL produces an actual interrupt and successful same-thread resume;
- durable SQLite recovery remains verified;
- time-travel replay succeeds;
- time-travel fork succeeds without deleting original history;
- separate bonus graph uses actual LangGraph `Send()` and aggregates all dispatched tasks;
- Streamlit UI module and view model pass smoke/security checks;
- Mermaid export remains generated from the compiled core graph;
- `outputs/bonus_evidence.json` validates semantically;
- report extension matrix matches generated evidence.

No merge occurs until the fresh full-bonus run is green and its artifact evidence has been reviewed.

## 12. Non-goals

- changing the required core topology to showcase bonus features;
- implementing both SQLite and Postgres merely for quantity;
- adding a production authentication system to the Streamlit UI;
- using real destructive external tools;
- running interactive human approval inside CI;
- recreating hidden grading scenarios;
- adding automatic push/PR/schedule workflow triggers;
- claiming visual browser end-to-end coverage when only Streamlit smoke evidence exists.
