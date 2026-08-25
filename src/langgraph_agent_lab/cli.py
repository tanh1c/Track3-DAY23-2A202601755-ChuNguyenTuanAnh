"""CLI for running, validating, and inspecting the lab workflow."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer
import yaml

from .bonus_evidence import BonusEvidence, write_bonus_evidence
from .bonus_parallel import verify_parallel_send
from .graph import build_graph
from .hitl import verify_hitl_round_trip
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state
from .time_travel import find_checkpoint, fork_checkpoint, replay_checkpoint, verify_time_travel
from .ui import verify_ui_view_model

app = typer.Typer(no_args_is_help=True)


def _verify_recovery(kind: str, database_url: str | None, thread_id: str | None) -> bool:
    """Read a finished thread through a fresh SQLite saver/graph instance."""
    if kind != "sqlite" or not thread_id:
        return False
    try:
        fresh = build_checkpointer("sqlite", database_url)
        graph = build_graph(checkpointer=fresh)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        values: dict[str, Any] = dict(snapshot.values or {})
        events = list(values.get("events", []) or [])
        return bool(events) and any(event.get("node") == "finalize" for event in events)
    except Exception:
        return False


def _finalized(state: dict[str, Any]) -> bool:
    return any(event.get("node") == "finalize" for event in state.get("events", []) or [])


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all configured scenarios and write evidence-backed metrics/report output."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind = str(cfg.get("checkpointer", "memory"))
    database_url = cfg.get("database_url")
    checkpointer = build_checkpointer(checkpointer_kind, database_url)
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    first_thread_id: str | None = None

    for scenario in scenarios:
        state = initial_state(scenario)
        if first_thread_id is None:
            first_thread_id = state["thread_id"]
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
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

    recovery_verified = _verify_recovery(
        checkpointer_kind,
        str(database_url) if database_url is not None else None,
        first_thread_id,
    )
    report = summarize_metrics(metrics, resume_success=recovery_verified)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("export-graph")
def export_graph(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Export Mermaid from the actual compiled graph."""
    graph = build_graph(checkpointer=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    typer.echo(f"Wrote graph Mermaid to {output}")


@app.command("state-history")
def state_history(
    database: Annotated[Path, typer.Option("--database")],
    thread_id: Annotated[str, typer.Option("--thread-id")],
) -> None:
    """Print compact, read-only checkpoint facts for one SQLite-backed thread."""
    graph = build_graph(checkpointer=build_checkpointer("sqlite", str(database)))
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = list(graph.get_state_history(config))
    if not snapshots:
        typer.echo("No checkpoints found for that thread.")
        return

    for index, snapshot in enumerate(snapshots):
        snapshot_config = dict(snapshot.config or {})
        configurable = dict(snapshot_config.get("configurable", {}) or {})
        checkpoint_id = configurable.get("checkpoint_id", "unknown")
        values = dict(snapshot.values or {})
        events = list(values.get("events", []) or [])
        finalized = any(event.get("node") == "finalize" for event in events)
        route = values.get("route", "unknown")
        attempt = values.get("attempt", 0)
        typer.echo(
            f"{index}: checkpoint={checkpoint_id} route={route} "
            f"attempt={attempt} finalized={'yes' if finalized else 'no'}"
        )


@app.command("time-travel")
def time_travel(
    database: Annotated[Path, typer.Option("--database")],
    thread_id: Annotated[str, typer.Option("--thread-id")],
    checkpoint_id: Annotated[str, typer.Option("--checkpoint-id")],
    mode: Annotated[str, typer.Option("--mode")],
    set_query: Annotated[str | None, typer.Option("--set-query")] = None,
) -> None:
    """Replay or fork from one exact persisted core-graph checkpoint."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"replay", "fork"}:
        raise typer.BadParameter("--mode must be replay or fork")

    graph = build_graph(checkpointer=build_checkpointer("sqlite", str(database)))
    try:
        checkpoint = find_checkpoint(graph, thread_id, checkpoint_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if normalized_mode == "replay":
        result = replay_checkpoint(graph, checkpoint)
    else:
        if set_query is None or not set_query.strip():
            raise typer.BadParameter("fork mode requires --set-query")
        _fork_config, result = fork_checkpoint(
            graph,
            checkpoint,
            {"query": set_query.strip()},
        )

    typer.echo(
        f"checkpoint={checkpoint_id} mode={normalized_mode} "
        f"route={result.get('route', 'unknown')} attempt={result.get('attempt', 0)} "
        f"finalized={'yes' if _finalized(result) else 'no'}"
    )


@app.command("verify-bonus")
def verify_bonus(
    database: Annotated[Path, typer.Option("--database")] = Path(
        "outputs/bonus-checkpoints.sqlite"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/bonus_evidence.json"),
    llm_judge_verified: Annotated[bool, typer.Option("--llm-judge-verified")] = False,
    durable_recovery_verified: Annotated[
        bool, typer.Option("--durable-recovery-verified")
    ] = False,
    mermaid_verified: Annotated[bool, typer.Option("--mermaid-verified")] = False,
) -> None:
    """Execute every offline bonus proof and write strict machine-readable evidence."""
    evidence = BonusEvidence(
        llm_as_judge_verified=llm_judge_verified,
        durable_recovery_verified=durable_recovery_verified,
        mermaid_export_verified=mermaid_verified,
        hitl=verify_hitl_round_trip(str(database)),
        time_travel=verify_time_travel(str(database)),
        parallel_send=verify_parallel_send(),
        streamlit_ui=verify_ui_view_model(),
    )
    write_bonus_evidence(evidence, output)
    typer.echo(f"Wrote bonus evidence to {output}")


if __name__ == "__main__":
    app()
