"""Deterministic Markdown report generation from validated runtime metrics."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .metrics import MetricsReport


STUDENT_NAME = "Chu Nguyen Tuan Anh"
REPOSITORY = "tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _commit_label() -> str:
    """Use the CI commit when available without invoking Git from the renderer."""
    return os.getenv("GITHUB_SHA", "not recorded in this runtime")


def render_report(metrics: MetricsReport) -> str:
    """Render the complete evidence-based lab report from one MetricsReport."""
    lines = [
        "# LangGraph Agentic Orchestration Lab Report",
        "",
        "## Student",
        "",
        f"- Name: {STUDENT_NAME}",
        f"- Repository: `{REPOSITORY}`",
        f"- Commit: `{_commit_label()}`",
        f"- Report date: {date.today().isoformat()}",
        "- Runtime numbers below are rendered from the validated metrics object, not retyped.",
        "",
        "## Architecture",
        "",
        "The workflow contains eleven registered nodes: `intake`, `classify`, `tool`, "
        "`evaluate`, `answer`, `clarify`, `risky_action`, `approval`, `retry`, "
        "`dead_letter`, and `finalize`. Four routing functions choose conditional edges.",
        "Every terminal path reaches `finalize` before `END`. The error path enters `retry` "
        "before a tool call, and only the retry node increments the bounded attempt counter.",
        "",
        "## State Schema",
        "",
        "| Field group | Update semantics | Purpose |",
        "|---|---|---|",
        "| `messages`, `tool_results`, `errors`, `events` | append reducer | ordered audit/history |",
        "| `route`, `risk_level` | overwrite | current classified intent and risk |",
        "| `attempt`, `max_attempts` | overwrite | bounded retry state |",
        "| `evaluation_result` | overwrite | evaluate routing gate |",
        "| `pending_question` | overwrite | current clarification output |",
        "| `proposed_action`, `approval` | overwrite | approval-gated side effect state |",
        "| `final_answer` | overwrite | terminal user-facing result |",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total scenarios | {metrics.total_scenarios} |",
        f"| Success rate | {metrics.success_rate:.2%} |",
        f"| Average nodes visited | {metrics.avg_nodes_visited:.2f} |",
        f"| Total retries | {metrics.total_retries} |",
        f"| Total real interrupts | {metrics.total_interrupts} |",
        f"| `resume_success` | `{str(metrics.resume_success).lower()}` |",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Expected route | Actual route | Success | Nodes | Retries | Interrupts | Approval observed | Latency ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in metrics.scenario_metrics:
        lines.append(
            "| "
            f"{item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | "
            f"{_yes_no(item.success)} | {item.nodes_visited} | {item.retry_count} | "
            f"{item.interrupt_count} | {_yes_no(item.approval_observed)} | {item.latency_ms} |"
        )

    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            "### Failure mode 1 — transient tool failure and bounded retry",
            "",
            "The mock tool deliberately returns an `ERROR` result for early attempts on the "
            "error route. `evaluate` converts unusable evidence into `needs_retry`; `retry` is "
            "the only counter owner. Once `attempt >= max_attempts`, routing fails closed into "
            "`dead_letter`, which can only continue to `finalize`. Residual risk: a real tool "
            "would need provider-specific timeout and idempotency controls.",
            "",
            "### Failure mode 2 — risky side effect and approval gate",
            "",
            "A risky request first creates `proposed_action`; it does not execute a side effect. "
            "The approval decision gates the only path to `tool`. The tool itself also fails "
            "closed when a risky route lacks affirmative approval, providing defense in depth. "
            "A rejected decision routes to clarification rather than unauthorized execution.",
            "",
            "## Persistence and Recovery",
            "",
            f"- `resume_success`: `{str(metrics.resume_success).lower()}`.",
        ]
    )
    if metrics.resume_success:
        lines.append(
            "- Recovery evidence indicates a fresh SQLite-backed graph instance could read a "
            "previously completed thread by its stable `thread_id`."
        )
    else:
        lines.append(
            "- Durable recovery was not proven in this metrics run; configuration alone is not "
            "counted as recovery evidence."
        )

    lines.extend(["", "## Extension Work", ""])
    lines.append(
        "- LLM-as-judge: live provider runs use one structured evaluation call per tool result; "
        "provider/schema failures fall back deterministically without an internal retry loop."
    )
    lines.append(
        "- SQLite persistence: durable checkpointer support uses WAL and stable thread IDs; "
        "state-history inspection is read-only."
    )
    if metrics.total_interrupts > 0:
        lines.append(
            f"- Real HITL: {metrics.total_interrupts} real interrupt event(s) were observed in "
            "this run."
        )
    else:
        lines.append(
            "- Real HITL is feature-gated for interactive runs; no real HITL interrupt was "
            "observed in this run."
        )
    lines.extend(
        [
            "- Mermaid graph export is derived from the compiled graph rather than a hand-written "
            "diagram.",
            "",
            "## Improvement Plan",
            "",
            "The next production priority is replacing the deterministic mock tool with "
            "idempotent provider adapters that have explicit timeout/retry budgets, while keeping "
            "the current approval boundary and checkpoint/audit contracts unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a UTF-8 Markdown file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
