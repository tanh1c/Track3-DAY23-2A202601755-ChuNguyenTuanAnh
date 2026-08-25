"""CLI for running, validating, and inspecting the lab workflow."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

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


if __name__ == "__main__":
    app()
