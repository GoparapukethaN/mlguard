from pathlib import Path


def test_composite_action_exposes_json_report_output() -> None:
    action_yaml = Path("action/action.yml").read_text(encoding="utf-8")

    assert "json-output-path" in action_yaml
    assert "--json-output" in action_yaml
    assert "mlguard_report.json" in action_yaml


def test_composite_action_does_not_advertise_unsupported_torch_artifacts() -> None:
    action_yaml = Path("action/action.yml").read_text(encoding="utf-8")

    assert ".pt" not in action_yaml
    assert ".pth" not in action_yaml
