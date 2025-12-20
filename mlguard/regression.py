"""Performance regression check.

Loads a model, runs predictions on a holdout set, and compares metrics
against a saved baseline. Supports sklearn and pytorch models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


@dataclass
class RegressionResult:
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    status: str  # "pass", "warn", "fail"


def load_model(model_path: str | Path):
    """Load a model from disk. Tries joblib first, then pickle, then torch."""
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()

    if suffix in (".pkl", ".joblib"):
        return joblib.load(model_path)
    elif suffix in (".pt", ".pth"):
        try:
            import torch
            return torch.load(model_path, map_location="cpu", weights_only=False)
        except ImportError:
            raise ImportError("Install torch to load .pt/.pth models")
    else:
        # try joblib as default
        return joblib.load(model_path)


def check_regression(
    model,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    baseline: dict,
    task: str = "classification",
    warn_pct: float = 5.0,
    fail_pct: float = 10.0,
) -> list[RegressionResult]:
    """Compare current model performance against baseline metrics.

    Args:
        model: fitted model with .predict() method
        X: feature matrix for holdout set
        y: true labels for holdout set
        baseline: dict of {metric_name: value} from previous run
        task: "classification" or "regression"
        warn_pct: percentage drop that triggers a warning
        fail_pct: percentage drop that triggers a failure
    """
    predictions = model.predict(X)
    results = []

    if task == "classification":
        current_metrics = {
            "accuracy": accuracy_score(y, predictions),
            "f1": f1_score(y, predictions, average="weighted", zero_division=0),
        }
    else:
        current_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        }

    for metric_name, current_val in current_metrics.items():
        baseline_val = baseline.get(metric_name)
        if baseline_val is None:
            continue

        # for error metrics (rmse), increase is bad. for accuracy/f1, decrease is bad.
        if metric_name in ("rmse", "mse", "mae"):
            # higher is worse
            if baseline_val > 0:
                change_pct = ((current_val - baseline_val) / baseline_val) * 100
            else:
                change_pct = 0.0
            is_worse = change_pct > 0
        else:
            # higher is better
            if baseline_val > 0:
                change_pct = ((baseline_val - current_val) / baseline_val) * 100
            else:
                change_pct = 0.0
            is_worse = change_pct > 0

        if is_worse and abs(change_pct) >= fail_pct:
            status = "fail"
        elif is_worse and abs(change_pct) >= warn_pct:
            status = "warn"
        else:
            status = "pass"

        results.append(
            RegressionResult(
                metric=metric_name,
                baseline_value=round(baseline_val, 4),
                current_value=round(current_val, 4),
                change_pct=round(change_pct, 2),
                status=status,
            )
        )

    return results
