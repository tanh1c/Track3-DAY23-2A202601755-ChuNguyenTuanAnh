# LangGraph Agentic Orchestration Lab Report

## Student

- Name: Chu Nguyễn Tuấn Anh
- MSSV: `2A202601755`
- Repository: `tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh`
- Commit: `35ede0e6051f00de8043be2349069ea77c0da83c`
- Report date: 2026-08-25
- Runtime numbers below are rendered from validated evidence objects, not retyped.

## Architecture

The workflow contains eleven registered nodes: `intake`, `classify`, `tool`, `evaluate`, `answer`, `clarify`, `risky_action`, `approval`, `retry`, `dead_letter`, and `finalize`. Four routing functions choose conditional edges.
Every terminal path reaches `finalize` before `END`. The error path enters `retry` before a tool call, and only the retry node increments the bounded attempt counter.
Bonus demonstrations are isolated helpers or separate graphs and do not modify this required eleven-node topology.

## State Schema

| Field group | Update semantics | Purpose |
|---|---|---|
| `messages`, `tool_results`, `errors`, `events` | append reducer | ordered audit/history |
| `route`, `risk_level` | overwrite | current classified intent and risk |
| `attempt`, `max_attempts` | overwrite | bounded retry state |
| `evaluation_result` | overwrite | evaluate routing gate |
| `pending_question` | overwrite | current clarification output |
| `proposed_action`, `approval` | overwrite | approval-gated side effect state |
| `final_answer` | overwrite | terminal user-facing result |

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.57 |
| Total retries | 4 |
| Total real interrupts | 0 |
| `resume_success` | `true` |

## Scenario Results

| Scenario | Expected route | Actual route | Success | Nodes | Retries | Interrupts | Approval observed | Latency ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S01_simple | simple | simple | yes | 4 | 0 | 0 | no | 2758 |
| S02_tool | tool | tool | yes | 6 | 0 | 0 | no | 2290 |
| S03_missing | missing_info | missing_info | yes | 4 | 0 | 0 | no | 786 |
| S04_risky | risky | risky | yes | 8 | 0 | 0 | yes | 1725 |
| S05_error | error | error | yes | 11 | 3 | 0 | no | 2195 |
| S06_delete | risky | risky | yes | 8 | 0 | 0 | yes | 2023 |
| S07_dead_letter | error | error | yes | 5 | 1 | 0 | no | 805 |

## Failure Analysis

### Failure mode 1 — transient tool failure and bounded retry

The mock tool deliberately returns an `ERROR` result for early attempts on the error route. `evaluate` converts unusable evidence into `needs_retry`; `retry` is the only counter owner. Once `attempt >= max_attempts`, routing fails closed into `dead_letter`, which can only continue to `finalize`. Residual risk: a real tool would need provider-specific timeout and idempotency controls.

### Failure mode 2 — risky side effect and approval gate

A risky request first creates `proposed_action`; it does not execute a side effect. The approval decision gates the only path to `tool`. The tool itself also fails closed when a risky route lacks affirmative approval, providing defense in depth. A rejected decision routes to clarification rather than unauthorized execution.

## Persistence and Recovery

- `resume_success`: `true`.
- Recovery evidence indicates a fresh SQLite-backed graph instance could read a previously completed thread by its stable `thread_id`.

## Extension Work

- LLM-as-judge: live provider runs use one structured evaluation call per tool result; provider/schema failures fall back deterministically without an internal retry loop.
- SQLite persistence: durable checkpointer support uses WAL and stable thread IDs; state-history inspection is read-only.
- The seven core scenarios run non-interactively; no real HITL interrupt was observed in that core scenario batch.

### Official extension matrix

| Extension | Baseline | Implementation | Verification | Verified | Evidence | Limitations |
|---|---|---|---|---:|---|---|
| LLM-as-judge | deterministic evaluator fallback | structured verdict with one bounded live judge call | live provider gate plus evaluator tests | yes | structured evaluator exercised in live verification | provider failure still falls back deterministically |
| Real HITL | mock approval in non-interactive core runs | real interrupt and Command(resume) helper | approve and reject round-trips on durable SQLite | yes | interrupt + same-thread resume + rejection path; reviewer=ci-reviewer | CI uses programmed reviewer decisions rather than a waiting human |
| SQLite recovery | memory checkpointer available for lightweight tests | durable SQLite saver with stable thread IDs | fresh saver reads a previously completed thread | yes | resume_success from restart-style recovery proof | Postgres is intentionally not required by the SQLite/Postgres option |
| Time travel | read-only state-history inspection | exact checkpoint replay and explicit fork | replay + fork + original history preservation checks | yes | replay + fork verified from checkpoint 1f1a0798-09b7-6198-8001-94a413bc766e | verification uses a deterministic fixture; CLI supports persisted core threads |
| Parallel Send | single-path required support graph | separate map-reduce graph using LangGraph Send | actual Send objects plus reducer aggregation | yes | 3 tasks -> 3 reducer results using Send | kept separate so the required eleven-node graph is unchanged |
| Streamlit UI | CLI/report evidence only | optional presentation layer over the existing state contract | import, view-model, and secret-safety smoke | yes | launch: `streamlit run src/langgraph_agent_lab/ui.py` | presentation smoke, not browser E2E |
| Mermaid export | target topology documented in the lab | export generated from the compiled core graph | semantic gate checks all eleven required node names | yes | compiled eleven-node core graph exported to outputs/graph.mmd | diagram evidence does not replace runtime graph tests |

- Mermaid graph export is derived from the compiled graph rather than a hand-written diagram.

## Improvement Plan

The next production priority is replacing the deterministic mock tool with idempotent provider adapters that have explicit timeout/retry budgets, while keeping the current approval boundary and checkpoint/audit contracts unchanged.
