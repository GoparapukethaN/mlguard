"""CLI for mlguard. Three commands: check, baseline, report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from mlguard.baseline import load_baseline, save_baseline
from mlguard.drift import check_drift
from mlguard.latency import check_latency
from mlguard.regression import check_regression, load_model
from mlguard.report import generate_report
from mlguard.verdict import decide

app = typer.Typer(help="mlguard: pre-deployment safety checks for ML models.")
console = Console()


@app.command()
def check(
    model: str = typer.Option(..., help="Path to model file (.pkl, .joblib, .pt)"),
    ref: str = typer.Option(..., help="Path to reference dataset (CSV)"),
    current: str = typer.Option(..., help="Path to current dataset (CSV)"),
    baseline_path: str = typer.Option(
        "./mlguard_baseline.json", help="Path to baseline metrics JSON"
    ),
    target: str = typer.Option("target", help="Name of target column in CSV"),
    task: str = typer.Option("classification", help="Task type: classification or regression"),
    output: str = typer.Option("./mlguard_report.md", help="Output report path"),
    latency_runs: int = typer.Option(100, help="Number of inference calls for latency check"),
):
    """Run all three checks: drift, regression, latency."""
    console.print("\n[bold]mlguard[/bold] — pre-deployment safety checks\n")

    # load data
    ref_df = pd.read_csv(ref)
    cur_df = pd.read_csv(current)
    console.print(f"  Reference: {len(ref_df)} rows, Current: {len(cur_df)} rows")

    # load model
    mdl = load_model(model)
    console.print(f"  Model: {Path(model).name}")

    # load baseline
    try:
        bl = load_baseline(baseline_path)
        console.print(f"  Baseline: {baseline_path}")
    except FileNotFoundError:
        console.print(
            f"  [yellow]No baseline found at {baseline_path}. "
            "Run `mlguard baseline` first, or using defaults.[/yellow]"
        )
        bl = {}

    # separate features and target
    feature_cols = [c for c in ref_df.columns if c != target]
    ref_features = ref_df[feature_cols]
    cur_features = cur_df[feature_cols]

    # 1. drift check
    console.print("\n  [1/3] Checking data drift...")
    drift_results = check_drift(ref_features, cur_features)
    for d in drift_results:
        icon = "[green]PASS[/green]" if d.status == "pass" else (
            "[yellow]WARN[/yellow]" if d.status == "warn" else "[red]FAIL[/red]"
        )
        console.print(f"    {d.feature}: PSI={d.psi:.4f} {icon}")

    # 2. performance regression check
    console.print("  [2/3] Checking performance regression...")
    X_cur = cur_features.values
    y_cur = cur_df[target].values
    perf_baseline = bl.get("performance", {})
    regression_results = check_regression(mdl, X_cur, y_cur, perf_baseline, task=task)
    for r in regression_results:
        icon = "[green]PASS[/green]" if r.status == "pass" else (
            "[yellow]WARN[/yellow]" if r.status == "warn" else "[red]FAIL[/red]"
        )
        console.print(
            f"    {r.metric}: {r.baseline_value:.4f} → {r.current_value:.4f} "
            f"({r.change_pct:+.1f}%) {icon}"
        )

    # 3. latency check
    console.print("  [3/3] Checking inference latency...")
    latency_baseline = bl.get("latency_p95_ms")
    latency_result = check_latency(mdl, X_cur, n_runs=latency_runs,
                                   baseline_p95_ms=latency_baseline)
    lat_icon = "[green]PASS[/green]" if latency_result.status == "pass" else (
        "[yellow]WARN[/yellow]" if latency_result.status == "warn" else "[red]FAIL[/red]"
    )
    console.print(
        f"    p95={latency_result.p95_ms:.2f}ms "
        f"(baseline={latency_result.baseline_p95_ms:.2f}ms, "
        f"{latency_result.change_pct:+.1f}%) {lat_icon}"
    )

    # verdict
    verdict = decide(
        [d.status for d in drift_results],
        [r.status for r in regression_results],
        latency_result.status,
    )

    generate_report(verdict, drift_results, regression_results,
                             latency_result, output)

    if verdict.overall == "PASS":
        console.print(f"\n  [bold green]{verdict.overall}[/bold green] — {verdict.summary}")
    elif verdict.overall == "WARN":
        console.print(f"\n  [bold yellow]{verdict.overall}[/bold yellow] — {verdict.summary}")
    else:
        console.print(f"\n  [bold red]{verdict.overall}[/bold red] — {verdict.summary}")

    console.print(f"  Report saved to {output}\n")

    if verdict.overall == "FAIL":
        raise typer.Exit(1)


@app.command("baseline")
def create_baseline(
    model: str = typer.Option(..., help="Path to model file"),
    data: str = typer.Option(..., help="Path to dataset (CSV)"),
    target: str = typer.Option("target", help="Target column name"),
    task: str = typer.Option("classification", help="classification or regression"),
    output: str = typer.Option("./mlguard_baseline.json", help="Output baseline path"),
    latency_runs: int = typer.Option(100, help="Number of runs for latency baseline"),
):
    """Create a baseline from the current model + data."""
    console.print("\n[bold]mlguard baseline[/bold]\n")

    df = pd.read_csv(data)
    mdl = load_model(model)

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = df[target].values

    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

    predictions = mdl.predict(X)

    if task == "classification":
        perf = {
            "accuracy": round(float(accuracy_score(y, predictions)), 4),
            "f1": round(float(f1_score(y, predictions, average="weighted",
                                       zero_division=0)), 4),
        }
    else:
        perf = {
            "rmse": round(float(np.sqrt(mean_squared_error(y, predictions))), 4),
        }

    latency_result = check_latency(mdl, X, n_runs=latency_runs)

    baseline = {
        "performance": perf,
        "latency_p95_ms": latency_result.p95_ms,
    }

    save_baseline(baseline, output)
    console.print(f"  Performance: {perf}")
    console.print(f"  Latency p95: {latency_result.p95_ms:.2f}ms")
    console.print(f"  Saved to {output}\n")


if __name__ == "__main__":
    app()
