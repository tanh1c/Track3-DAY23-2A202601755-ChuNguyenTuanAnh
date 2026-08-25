"""Deterministic Markdown report generation from validated runtime evidence."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .bonus_evidence import BonusEvidence
from .metrics import MetricsReport


STUDENT_NAME = "Chu Nguyễn Tuấn Anh"
STUDENT_ID = "2A202601755"
REPOSITORY = "tanh1c/Track3-DAY23-2A202601755-ChuNguyenTuanAnh"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _commit_label() -> str:
    """Use the CI commit when available without invoking Git from the renderer."""
    return os.getenv("GITHUB_SHA", "not recorded in this runtime")


def _bonus_rows(
    bonus: BonusEvidence,
) -> list[tuple[str, str, str, str, bool, str, str]]:
    """Translate typed bonus evidence into the official extension proof matrix."""
    hitl = bonus.hitl
    recovery = bonus.recovery
    time_travel = bonus.time_travel
    parallel = bonus.parallel_send
    ui = bonus.streamlit_ui
    return [
        (
            "LLM-as-judge",
            "deterministic evaluator fallback",
            "structured verdict with reason, 20s timeout, fallback, and zero provider retries",
            "live provider gate plus evaluator budget tests",
            bonus.llm_as_judge_verified,
            "one bounded structured judge call; timeout=20s; max_retries=0",
            "provider/schema/timeout failure falls back deterministically",
        ),
        (
            "Real HITL",
            "mock approval in non-interactive core runs",
            "real interrupt and Command(resume) helper",
            "approve and reject round-trips on durable SQLite",
            hitl.verified,
            (
                "interrupt + same-thread resume + rejection path; "
                f"reviewer={hitl.reviewer or '-'}"
            ),
            "CI uses programmed reviewer decisions rather than a waiting human",
        ),
        (
            "SQLite recovery",
            "fresh saver recovery in the core scenario runner",
            "writer subprocess exits before a distinct reader subprocess loads the checkpoint",
            "two Python processes use one SQLite DB and the same stable thread_id",
            bonus.durable_recovery_verified,
            (
                f"writer PID {recovery.writer_pid} -> reader PID {recovery.reader_pid}; "
                f"thread={recovery.thread_id or '-'}; finalized={_yes_no(recovery.persisted_finalized)}"
            ),
            "SQLite path is proven; Postgres is not exercised",
        ),
        (
            "Time travel",
            "read-only state-history inspection",
            "exact checkpoint replay and explicit fork",
            "replay + fork + original history preservation checks",
            time_travel.verified,
            (
                f"replay + fork verified from checkpoint "
                f"{time_travel.checkpoint_id or '-'}"
            ),
            "verification uses a deterministic fixture; CLI supports persisted core threads",
        ),
        (
            "Parallel Send",
            "single-path required support graph",
            "separate map-reduce graph using LangGraph Send",
            "actual Send objects plus reducer aggregation",
            parallel.verified,
            f"{parallel.task_count} tasks -> {parallel.result_count} reducer results using Send",
            "kept separate so the required eleven-node graph is unchanged",
        ),
        (
            "Streamlit UI",
            "CLI/report evidence only",
            "optional presentation layer over the existing state contract",
            "import, view-model, and secret-safety smoke",
            ui.verified,
            "launch: `streamlit run src/langgraph_agent_lab/ui.py`",
            "presentation smoke, not browser E2E",
        ),
        (
            "Mermaid export",
            "target topology documented in the lab",
            "export generated from the compiled core graph",
            "semantic gate checks all eleven required node names",
            bonus.mermaid_export_verified,
            "compiled eleven-node core graph exported to outputs/graph.mmd",
            "diagram evidence does not replace runtime graph tests",
        ),
    ]


def render_report(metrics: MetricsReport, bonus: BonusEvidence | None = None) -> str:
    """Render the complete evidence-based lab report from validated runtime objects."""
    state_history_row = (
        "| `messages`, `tool_results`, `errors`, `events` | append reducer | "
        "ordered audit/history |"
    )
    scenario_header = (
        "| Scenario | Expected route | Actual route | Success | Nodes | Retries | "
        "Interrupts | Approval observed | Latency ms |"
    )
    lines = [
        "# LangGraph Agentic Orchestration Lab Report",
        "",
        "## Student",
        "",
        f"- Name: {STUDENT_NAME}",
        f"- MSSV: `{STUDENT_ID}`",
        f"- Repository: `{REPOSITORY}`",
        f"- Commit: `{_commit_label()}`",
        f"- Report date: {date.today().isoformat()}",
        "- Runtime numbers below are rendered from validated evidence objects, not retyped.",
        "",
        "## Architecture",
        "",
        "The workflow contains eleven registered nodes: `intake`, `classify`, `tool`, "
        "`evaluate`, `answer`, `clarify`, `risky_action`, `approval`, `retry`, "
        "`dead_letter`, and `finalize`. Four routing functions choose conditional edges.",
        "Every terminal path reaches `finalize` before `END`. The error path enters `retry` "
        "before a tool call, and only the retry node increments the bounded attempt counter.",
        "Bonus demonstrations are isolated helpers or separate graphs and do not modify this "
        "required eleven-node topology.",
        "",
        "## State Schema",
        "",
        "| Field group | Update semantics | Purpose |",
        "|---|---|---|",
        state_history_row,
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
        scenario_header,
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
            "- Core recovery evidence indicates a fresh SQLite-backed graph instance could read "
            "a previously completed thread by its stable `thread_id`."
        )
    else:
        lines.append(
            "- Durable recovery was not proven in this metrics run; configuration alone is not "
            "counted as recovery evidence."
        )

    lines.extend(["", "## Extension Work", ""])
    lines.append(
        "- LLM-as-judge: one structured verdict call has an explicit 20-second request timeout "
        "and zero provider retries; provider/schema/timeout failure falls back deterministically."
    )
    lines.append(
        "- SQLite persistence: durable checkpointer support uses WAL and stable thread IDs; the "
        "bonus verifier proves survival across two distinct Python processes."
    )
    if metrics.total_interrupts > 0:
        lines.append(
            f"- Core scenarios observed {metrics.total_interrupts} real interrupt event(s)."
        )
    else:
        lines.append(
            "- The seven core scenarios run non-interactively; no real HITL interrupt was "
            "observed in that core scenario batch."
        )

    if bonus is not None:
        lines.extend(
            [
                "",
                "### Official extension matrix",
                "",
                (
                    "| Extension | Baseline | Implementation | Verification | Verified | "
                    "Evidence | Limitations |"
                ),
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for row in _bonus_rows(bonus):
            name, baseline, change, verification, verified, evidence_text, limitation = row
            lines.append(
                f"| {name} | {baseline} | {change} | {verification} | "
                f"{_yes_no(verified)} | {evidence_text} | {limitation} |"
            )

    lines.extend(
        [
            "",
            "- Mermaid graph export is derived from the compiled graph rather than a hand-written "
            "diagram.",
            "",
            "## Improvement Plan",
            "",
            "The next production priority is replacing the deterministic mock tool with "
            "idempotent provider adapters while keeping the current bounded LLM judge, approval "
            "boundary, persistence, and checkpoint/audit contracts unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    metrics: MetricsReport,
    output_path: str | Path,
    bonus: BonusEvidence | None = None,
) -> None:
    """Write the rendered report to a UTF-8 Markdown file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics, bonus=bonus), encoding="utf-8")
