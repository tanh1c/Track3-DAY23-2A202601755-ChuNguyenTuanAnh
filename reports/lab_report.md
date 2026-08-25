# LangGraph Agentic Orchestration Lab Report

## Student

- Name: Chu Nguyen Tuan Anh
- Repository: `tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh`
- Commit: `d6ea5833e59c5ff349eda56b9139e07277ea30c0`
- Report date: 2026-08-25
- Runtime numbers below are rendered from the validated metrics object, not retyped.

## Architecture

The workflow contains eleven registered nodes: `intake`, `classify`, `tool`, `evaluate`, `answer`, `clarify`, `risky_action`, `approval`, `retry`, `dead_letter`, and `finalize`. Four routing functions choose conditional edges.
Every terminal path reaches `finalize` before `END`. The error path enters `retry` before a tool call, and only the retry node increments the bounded attempt counter.

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
| S01_simple | simple | simple | yes | 4 | 0 | 0 | no | 2363 |
| S02_tool | tool | tool | yes | 6 | 0 | 0 | no | 1837 |
| S03_missing | missing_info | missing_info | yes | 4 | 0 | 0 | no | 1121 |
| S04_risky | risky | risky | yes | 8 | 0 | 0 | yes | 3370 |
| S05_error | error | error | yes | 11 | 3 | 0 | no | 2100 |
| S06_delete | risky | risky | yes | 8 | 0 | 0 | yes | 1866 |
| S07_dead_letter | error | error | yes | 5 | 1 | 0 | no | 530 |

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
- Real HITL is feature-gated for interactive runs; no real HITL interrupt was observed in this run.
- Mermaid graph export is derived from the compiled graph rather than a hand-written diagram.

## Improvement Plan

The next production priority is replacing the deterministic mock tool with idempotent provider adapters that have explicit timeout/retry budgets, while keeping the current approval boundary and checkpoint/audit contracts unchanged.
