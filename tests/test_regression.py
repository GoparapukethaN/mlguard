"""Tests for performance regression checks."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlguard.regression import check_regression


class TestCheckRegression:
    def _make_model_and_data(self, n_samples=200):
        X, y = make_classification(
            n_samples=n_samples, n_features=5, random_state=42
        )
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_no_regression_passes(self):
        model, X, y = self._make_model_and_data()
        # baseline matches current (same model, same data)
        predictions = model.predict(X)
        from sklearn.metrics import accuracy_score, f1_score

        baseline = {
            "accuracy": accuracy_score(y, predictions),
            "f1": f1_score(y, predictions, average="weighted"),
        }
        results = check_regression(model, X, y, baseline)
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)

    def test_regression_detected(self):
        model, X, y = self._make_model_and_data()
        # fake a high baseline that current model can't match
        baseline = {"accuracy": 0.99, "f1": 0.99}
        results = check_regression(model, X, y, baseline, fail_pct=5.0)
        # model accuracy is around 0.85, so 0.99 -> 0.85 is >10% drop
        has_fail = any(r.status == "fail" for r in results)
        assert has_fail

    def test_handles_missing_baseline_metric(self):
        model, X, y = self._make_model_and_data()
        baseline = {"accuracy": 0.85}  # no f1 in baseline
        results = check_regression(model, X, y, baseline)
        # should only report on accuracy, skip f1
        assert len(results) == 1
        assert results[0].metric == "accuracy"
