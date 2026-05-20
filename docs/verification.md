# Verification

This is the local checklist I use before treating MLGuard changes as ready to show.
The goal is to keep the release-gate claims tied to commands that run without external
services.

## Local CI

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make verify
```

Current local result from 2026-05-20:

- `24 passed`
- `ruff` clean
- sklearn example produced Markdown and JSON reports
- JSON report includes machine-readable summary counts

`make verify` runs Ruff, the pytest suite, and the included sklearn example. The example
trains a small model, records baseline metrics, simulates feature drift, runs the release
gate, and checks for the expected `FAIL` verdict.

## CLI Surface

The public CLI behavior covered by tests includes:

- missing baseline fails fast by default
- `--allow-missing-baseline` is explicit drift-only mode
- JSON reports include summary counts for failed and warned checks
- PyTorch `.pt`/`.pth` artifacts are rejected with a clear unsupported-model message

## Trust Boundary

MLGuard loads local model artifacts through joblib/pickle-style deserialization. That is
appropriate for release artifacts produced by the same trusted pipeline, but it should not
be pointed at untrusted model files.
