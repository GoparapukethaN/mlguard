"""Data drift detection using Population Stability Index (PSI).

PSI compares two distributions by binning values and measuring how much
the distribution has shifted. It's the standard metric in credit risk
and has held up well for ML feature monitoring too.

Interpretation:
  PSI < 0.1  → no significant drift
  PSI 0.1-0.2 → moderate drift, worth investigating
  PSI > 0.2  → significant drift, likely need to retrain
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    feature: str
    psi: float
    status: str  # "pass", "warn", "fail"


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate PSI between reference and current distributions.

    Uses equal-width bins based on the reference distribution.
    Clips zero-count bins to a small epsilon to avoid log(0).
    """
    eps = 1e-4

    # use reference distribution to define bin edges
    bin_edges = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # convert to proportions
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    warn_threshold: float = 0.1,
    fail_threshold: float = 0.2,
) -> list[DriftResult]:
    """Run PSI check on every numeric column.

    Only checks columns present in both dataframes.
    Skips non-numeric columns silently.
    """
    results = []
    common_cols = set(ref_df.columns) & set(cur_df.columns)

    for col in sorted(common_cols):
        try:
            if not np.issubdtype(ref_df[col].dtype, np.number):
                continue
            if not np.issubdtype(cur_df[col].dtype, np.number):
                continue
        except TypeError:
            # pandas extension types (StringDtype etc) aren't numpy dtypes
            continue

        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            # not enough data to compute meaningful PSI
            continue

        psi = compute_psi(ref_vals, cur_vals)

        if psi >= fail_threshold:
            status = "fail"
        elif psi >= warn_threshold:
            status = "warn"
        else:
            status = "pass"

        results.append(DriftResult(feature=col, psi=round(psi, 4), status=status))

    return results
