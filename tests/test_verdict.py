"""Tests for verdict logic."""

from mlguard.verdict import decide


class TestDecide:
    def test_all_pass(self):
        v = decide(["pass", "pass"], ["pass"], "pass")
        assert v.overall == "PASS"

    def test_any_fail_means_fail(self):
        v = decide(["pass", "fail"], ["pass"], "pass")
        assert v.overall == "FAIL"

    def test_warn_without_fail(self):
        v = decide(["pass", "warn"], ["pass"], "pass")
        assert v.overall == "WARN"

    def test_latency_fail(self):
        v = decide(["pass"], ["pass"], "fail")
        assert v.overall == "FAIL"
        assert "latency" in v.summary

    def test_empty_drift_still_works(self):
        v = decide([], ["pass"], "pass")
        assert v.overall == "PASS"

    def test_multiple_failures(self):
        v = decide(["fail", "fail"], ["fail"], "fail")
        assert v.overall == "FAIL"
        assert "drift" in v.summary
