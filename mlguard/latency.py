"""Inference latency check.

Times N prediction calls and compares p50/p95/p99 against a saved baseline.
A 20%+ latency increase usually means something changed in the model or
the preprocessing pipeline that needs investigation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    baseline_p95_ms: float
    change_pct: float
    status: str  # "pass", "warn", "fail"


def check_latency(
    model,
    X: np.ndarray,
    n_runs: int = 100,
    baseline_p95_ms: float | None = None,
    warn_pct: float = 15.0,
    fail_pct: float = 30.0,
) -> LatencyResult:
    """Time n_runs single-sample predictions and check against baseline.

    Uses single-sample predictions because that's the production pattern
    for most real-time serving. Batch latency is a different thing.
    """
    if len(X) == 0:
        raise ValueError("X is empty, need at least 1 sample for latency check")

    # warm up — first few calls are always slower (JIT, caching, etc)
    for _ in range(min(5, len(X))):
        model.predict(X[:1])

    latencies = []
    for i in range(n_runs):
        idx = i % len(X)
        sample = X[idx : idx + 1]

        t0 = time.perf_counter()
        model.predict(sample)
        dt = (time.perf_counter() - t0) * 1000  # ms

        latencies.append(dt)

    latencies_arr = np.array(latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))

    if baseline_p95_ms is not None and baseline_p95_ms > 0:
        change_pct = ((p95 - baseline_p95_ms) / baseline_p95_ms) * 100
    else:
        change_pct = 0.0
        baseline_p95_ms = p95  # no baseline, use current

    if change_pct >= fail_pct:
        status = "fail"
    elif change_pct >= warn_pct:
        status = "warn"
    else:
        status = "pass"

    return LatencyResult(
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        baseline_p95_ms=round(baseline_p95_ms, 2),
        change_pct=round(change_pct, 1),
        status=status,
    )
