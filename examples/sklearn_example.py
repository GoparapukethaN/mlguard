"""End-to-end example: train a model, save baseline, simulate drift, run mlguard.

Run this to see mlguard in action:
    python examples/sklearn_example.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlguard.baseline import save_baseline
from mlguard.drift import check_drift
from mlguard.latency import check_latency
from mlguard.regression import check_regression
from mlguard.report import generate_report
from mlguard.verdict import decide


def main():
    print("=== MLGuard Example ===\n")

    # 1. generate synthetic data and train a model
    print("1. Training a RandomForest on synthetic data...")
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # save model
    joblib.dump(model, "/tmp/example_model.pkl")

    # save reference data
    ref_df = pd.DataFrame(X_test, columns=feature_names)
    ref_df["target"] = y_test
    ref_df.to_csv("/tmp/ref_data.csv", index=False)

    # 2. create baseline
    print("2. Creating baseline metrics...")
    from sklearn.metrics import accuracy_score, f1_score

    preds = model.predict(X_test)
    perf = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
    }
    lat = check_latency(model, X_test, n_runs=50)
    baseline = {"performance": perf, "latency_p95_ms": lat.p95_ms}
    save_baseline(baseline, "/tmp/mlguard_baseline.json")
    print(f"   Accuracy: {perf['accuracy']}, F1: {perf['f1']}, p95: {lat.p95_ms:.2f}ms")

    # 3. simulate drift: shift 3 features
    print("\n3. Simulating data drift (shifting 3 features)...")
    X_drifted = X_test.copy()
    X_drifted[:, 0] += 2.0   # shift feature_0
    X_drifted[:, 1] *= 3.0   # scale feature_1
    X_drifted[:, 4] += 1.5   # shift feature_4

    cur_df = pd.DataFrame(X_drifted, columns=feature_names)
    cur_df["target"] = y_test
    cur_df.to_csv("/tmp/cur_data.csv", index=False)

    # 4. run checks
    print("4. Running mlguard checks...\n")

    # drift
    drift_results = check_drift(
        ref_df[feature_names], cur_df[feature_names]
    )
    for d in drift_results:
        status = "PASS" if d.status == "pass" else d.status.upper()
        print(f"   Drift {d.feature}: PSI={d.psi:.4f} [{status}]")

    # regression
    regression_results = check_regression(
        model, X_drifted, y_test, perf
    )
    for r in regression_results:
        print(f"   {r.metric}: {r.baseline_value} → {r.current_value} ({r.change_pct:+.1f}%)")

    # latency
    latency_result = check_latency(
        model, X_drifted, n_runs=50, baseline_p95_ms=lat.p95_ms
    )
    print(f"   Latency p95: {latency_result.p95_ms:.2f}ms [{latency_result.status.upper()}]")

    # verdict
    verdict = decide(
        [d.status for d in drift_results],
        [r.status for r in regression_results],
        latency_result.status,
    )
    print(f"\n   === VERDICT: {verdict.overall} ===")
    print(f"   {verdict.summary}")

    # save report
    report = generate_report(
        verdict, drift_results, regression_results,
        latency_result, "/tmp/mlguard_report.md"
    )
    print(f"\n   Report saved to /tmp/mlguard_report.md")


if __name__ == "__main__":
    main()
