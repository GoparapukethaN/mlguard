"""Tests for PSI drift detection."""

import numpy as np
import pandas as pd

from mlguard.drift import check_drift, compute_psi


class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.05  # should be very close to 0

    def test_shifted_distribution_high_psi(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # shifted mean
        psi = compute_psi(ref, cur)
        assert psi > 0.2  # should detect significant drift

    def test_slightly_shifted_moderate_psi(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0.5, 1, 1000)  # slight shift
        psi = compute_psi(ref, cur)
        # should be somewhere between 0 and high
        assert 0 <= psi < 1.0

    def test_psi_is_non_negative(self):
        ref = np.random.uniform(0, 10, 500)
        cur = np.random.uniform(2, 12, 500)
        psi = compute_psi(ref, cur)
        assert psi >= 0


class TestCheckDrift:
    def test_no_drift_all_pass(self):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.normal(5, 2, 500),
        })
        # same distribution
        cur = pd.DataFrame({
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.normal(5, 2, 500),
        })
        results = check_drift(ref, cur)
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)

    def test_drift_detected(self):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": rng.normal(5, 1, 500)})  # big shift
        results = check_drift(ref, cur)
        assert len(results) == 1
        assert results[0].status == "fail"

    def test_skips_non_numeric(self):
        ref = pd.DataFrame({"name": ["a", "b", "c"] * 10, "val": range(30)})
        cur = pd.DataFrame({"name": ["x", "y", "z"] * 10, "val": range(30)})
        results = check_drift(ref, cur)
        # should only check 'val', not 'name'
        assert len(results) == 1
        assert results[0].feature == "val"

    def test_skips_small_columns(self):
        ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})  # only 3 rows
        cur = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
        results = check_drift(ref, cur)
        assert len(results) == 0  # too few samples
