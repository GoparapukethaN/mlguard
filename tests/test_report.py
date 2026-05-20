from pathlib import Path

from mlguard.drift import DriftResult
from mlguard.latency import LatencyResult
from mlguard.regression import RegressionResult
from mlguard.report import generate_json_report, generate_report
from mlguard.verdict import Verdict


def _sample_inputs():
    verdict = Verdict(
        overall="FAIL",
        drift_status="fail",
        regression_status="warn",
        latency_status="pass",
        summary="1 feature(s) with significant drift",
    )
    drift = [DriftResult(feature="feature_0", psi=0.42, status="fail")]
    regression = [
        RegressionResult(
            metric="accuracy",
            baseline_value=0.9,
            current_value=0.84,
            change_pct=-6.7,
            status="warn",
        )
    ]
    latency = LatencyResult(
        p50_ms=1.0,
        p95_ms=2.0,
        p99_ms=3.0,
        baseline_p95_ms=2.1,
        change_pct=-4.8,
        status="pass",
    )
    return verdict, drift, regression, latency


def test_markdown_report_includes_release_gate_sections() -> None:
    verdict, drift, regression, latency = _sample_inputs()

    report = generate_report(verdict, drift, regression, latency)

    assert "# MLGuard Report — FAIL" in report
    assert "## Data Drift" in report
    assert "## Performance Regression" in report
    assert "## Latency" in report


def test_json_report_writes_machine_readable_release_gate(tmp_path: Path) -> None:
    output_path = tmp_path / "mlguard_report.json"
    verdict, drift, regression, latency = _sample_inputs()

    payload = generate_json_report(verdict, drift, regression, latency, str(output_path))

    assert output_path.exists()
    assert payload["verdict"]["overall"] == "FAIL"
    assert payload["data_drift"][0]["feature"] == "feature_0"
    assert payload["performance_regression"][0]["metric"] == "accuracy"
    assert payload["latency"]["status"] == "pass"


def test_json_report_includes_summary_counts() -> None:
    verdict, drift, regression, latency = _sample_inputs()

    payload = generate_json_report(verdict, drift, regression, latency)

    assert payload["summary"] == {
        "checks_total": 3,
        "checks_failed": 1,
        "checks_warned": 1,
        "drift_features_failed": 1,
        "drift_features_warned": 0,
        "regression_metrics_failed": 0,
        "regression_metrics_warned": 1,
        "latency_failed": 0,
        "latency_warned": 0,
    }
