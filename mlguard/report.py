"""Generate a markdown report from check results."""

from __future__ import annotations

from pathlib import Path

from mlguard.drift import DriftResult
from mlguard.latency import LatencyResult
from mlguard.regression import RegressionResult
from mlguard.verdict import Verdict

STATUS_ICONS = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def generate_report(
    verdict: Verdict,
    drift_results: list[DriftResult],
    regression_results: list[RegressionResult],
    latency_result: LatencyResult | None,
    output_path: str | None = None,
) -> str:
    """Generate a concise markdown report."""
    lines = [
        f"# MLGuard Report — {verdict.overall}",
        "",
        f"**Verdict:** {verdict.overall} — {verdict.summary}",
        "",
    ]

    # drift section
    lines.append("## Data Drift (PSI)")
    if drift_results:
        lines.append("")
        lines.append("| Feature | PSI | Status |")
        lines.append("|---------|-----|--------|")
        for d in drift_results:
            lines.append(
                f"| {d.feature} | {d.psi:.4f} | {STATUS_ICONS[d.status]} |"
            )
    else:
        lines.append("No numeric features to check.")
    lines.append("")

    # regression section
    lines.append("## Performance Regression")
    if regression_results:
        lines.append("")
        lines.append("| Metric | Baseline | Current | Change | Status |")
        lines.append("|--------|----------|---------|--------|--------|")
        for r in regression_results:
            sign = "+" if r.change_pct > 0 else ""
            lines.append(
                f"| {r.metric} | {r.baseline_value:.4f} | "
                f"{r.current_value:.4f} | {sign}{r.change_pct:.1f}% "
                f"| {STATUS_ICONS[r.status]} |"
            )
    else:
        lines.append("No regression data.")
    lines.append("")

    # latency section
    lines.append("## Latency")
    if latency_result:
        lines.append("")
        lines.append(f"- **p50:** {latency_result.p50_ms:.2f}ms")
        lines.append(f"- **p95:** {latency_result.p95_ms:.2f}ms")
        lines.append(f"- **p99:** {latency_result.p99_ms:.2f}ms")
        lines.append(
            f"- **Baseline p95:** {latency_result.baseline_p95_ms:.2f}ms"
        )
        sign = "+" if latency_result.change_pct > 0 else ""
        lines.append(
            f"- **Change:** {sign}{latency_result.change_pct:.1f}% "
            f"— {STATUS_ICONS[latency_result.status]}"
        )
    else:
        lines.append("Latency check skipped.")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report
