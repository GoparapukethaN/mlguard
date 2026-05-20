from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from typer.testing import CliRunner

from mlguard.cli import app

runner = CliRunner()


def _write_model_and_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = pd.DataFrame(
        {
            "feature_0": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "feature_1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    model = DummyClassifier(strategy="most_frequent")
    model.fit(data[["feature_0", "feature_1"]].values, data["target"].values)

    model_path = tmp_path / "model.pkl"
    ref_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    joblib.dump(model, model_path)
    data.to_csv(ref_path, index=False)
    data.to_csv(current_path, index=False)
    return model_path, ref_path, current_path


def test_check_fails_fast_when_baseline_is_missing(tmp_path: Path) -> None:
    model_path, ref_path, current_path = _write_model_and_data(tmp_path)
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "check",
            "--model",
            str(model_path),
            "--ref",
            str(ref_path),
            "--current",
            str(current_path),
            "--baseline-path",
            str(tmp_path / "missing-baseline.json"),
            "--target",
            "target",
            "--output",
            str(report_path),
            "--json-output",
            str(json_path),
            "--latency-runs",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "No baseline found" in result.output
    assert "baseline" in result.output
    assert not report_path.exists()
    assert not json_path.exists()


def test_check_can_explicitly_allow_missing_baseline(tmp_path: Path) -> None:
    model_path, ref_path, current_path = _write_model_and_data(tmp_path)
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "check",
            "--model",
            str(model_path),
            "--ref",
            str(ref_path),
            "--current",
            str(current_path),
            "--baseline-path",
            str(tmp_path / "missing-baseline.json"),
            "--target",
            "target",
            "--output",
            str(report_path),
            "--json-output",
            str(json_path),
            "--latency-runs",
            "3",
            "--allow-missing-baseline",
        ],
    )

    assert result.exit_code == 0
    assert "No baseline found" in result.output
    assert json_path.exists()
