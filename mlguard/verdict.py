"""Combine check results into a final PASS / WARN / FAIL verdict."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verdict:
    overall: str  # "PASS", "WARN", "FAIL"
    drift_status: str
    regression_status: str
    latency_status: str
    summary: str


def decide(
    drift_statuses: list[str],
    regression_statuses: list[str],
    latency_status: str,
) -> Verdict:
    """Simple worst-case aggregation. If anything fails, the whole thing fails.

    I considered weighted scoring but honestly this is better — you don't want
    to deploy a model with drifted features just because latency looks fine.
    """
    all_statuses = drift_statuses + regression_statuses + [latency_status]

    if "fail" in all_statuses:
        overall = "FAIL"
    elif "warn" in all_statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    # per-section status (worst in each group)
    drift_overall = _worst(drift_statuses) if drift_statuses else "pass"
    regression_overall = _worst(regression_statuses) if regression_statuses else "pass"

    # build summary
    parts = []
    if drift_overall == "fail":
        n_fail = sum(1 for s in drift_statuses if s == "fail")
        parts.append(f"{n_fail} feature(s) with significant drift")
    if regression_overall == "fail":
        parts.append("performance regression detected")
    if latency_status == "fail":
        parts.append("latency regression detected")

    if not parts:
        if overall == "WARN":
            parts.append("minor issues detected, review recommended")
        else:
            parts.append("all checks passed")

    summary = "; ".join(parts)

    return Verdict(
        overall=overall,
        drift_status=drift_overall,
        regression_status=regression_overall,
        latency_status=latency_status,
        summary=summary,
    )


def _worst(statuses: list[str]) -> str:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"
