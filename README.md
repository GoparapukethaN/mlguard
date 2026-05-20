# MLGuard

Pre-deployment safety checks for ML models. Three checks, one command, pass or fail.

## Why

Monitoring is useful after a model is live, but release decisions need checks before a
new model reaches production. This project is a small release gate for that moment: it
compares a candidate model and current data against a known baseline, then returns a
clear pass, warn, or fail verdict.

It runs three checks:

1. **Data drift** — are the input features still distributed the same way? (PSI)
2. **Performance regression** — did accuracy/F1 drop compared to baseline?
3. **Latency regression** — is inference slower than before?

If any check fails, the CLI exits non-zero so it can block a CI/CD pipeline.

## Quick start

```bash
git clone https://github.com/GoparapukethaN/mlguard.git
cd mlguard
pip install -e .

# create a baseline from your current model + data
mlguard baseline --model model.pkl --data reference.csv --target label

# check a new model/data against the baseline
mlguard check --model model.pkl --ref reference.csv --current new_data.csv --target label
```

`mlguard check` fails fast when the baseline file is missing. If you intentionally want
a drift-only run while bootstrapping a baseline, pass `--allow-missing-baseline`.

Model files are loaded as local joblib/pickle-style artifacts. Only run MLGuard against
trusted model files that you created or reviewed.

Output:
```
mlguard — pre-deployment safety checks

  Reference: 300 rows, Current: 300 rows
  Model: model.pkl
  Baseline: ./mlguard_baseline.json

  [1/3] Checking data drift...
    feature_0: PSI=0.4521 FAIL
    feature_1: PSI=0.8234 FAIL
    feature_2: PSI=0.0089 PASS
    feature_3: PSI=0.0124 PASS
    feature_4: PSI=0.2891 FAIL

  [2/3] Checking performance regression...
    accuracy: 0.9467 → 0.7900 (-16.6%) FAIL
    f1: 0.9467 → 0.7884 (-16.7%) FAIL

  [3/3] Checking inference latency...
    p95=1.65ms (baseline=1.97ms, -16.2%) PASS

  FAIL — 3 feature(s) with significant drift; performance regression detected

  Report saved to ./mlguard_report.md
  JSON report saved to ./mlguard_report.json
```

Exit code 1 on FAIL, 0 on PASS/WARN. The Markdown report is useful for review, and the
JSON report is useful for CI annotations, dashboards, or release records.

## The three checks

### Data drift (PSI)

Population Stability Index compares the distribution of each feature between your reference data and the current data. If a feature's distribution shifted significantly (PSI > 0.2), it means the model is seeing data it wasn't trained on.

- PSI < 0.1: no drift
- PSI 0.1-0.2: moderate (WARN)
- PSI > 0.2: significant (FAIL)

### Performance regression

Loads the model, runs predictions on the current data, and compares accuracy/F1 against the saved baseline. If accuracy dropped more than 10%, something is wrong.

- Drop < 5%: PASS
- Drop 5-10%: WARN
- Drop > 10%: FAIL

### Latency regression

Times 100 single-sample predictions and compares p95 latency against the baseline. A jump in latency usually means something changed in preprocessing or the model architecture got bigger.

- Increase < 15%: PASS
- Increase 15-30%: WARN
- Increase > 30%: FAIL

## GitHub Actions

You can run the CLI directly in a deployment workflow:

```yaml
- name: ML safety check
  run: |
    pip install -e .
    mlguard check \
      --model ./model.pkl \
      --ref ./data/reference.csv \
      --current ./data/latest.csv \
      --target label
```

The exit code blocks the pipeline on FAIL.

The included composite action accepts custom Markdown/JSON output paths and uploads both
reports as a single `mlguard-reports` artifact.

```yaml
- uses: GoparapukethaN/mlguard/action@main
  with:
    model-path: ./models/model.pkl
    reference-data: ./data/reference.csv
    current-data: ./data/current.csv
    baseline-path: ./mlguard_baseline.json
    target-column: target
```

## Example

```bash
# run the included example (trains a model, simulates drift, runs checks)
pip install -e .
python examples/sklearn_example.py
```

## Verified Locally

```bash
make verify
```

Current local verification: `ruff` clean, `25 passed`, and the sklearn example produces
Markdown and JSON release-gate reports with summary counts when drift and performance regression are
simulated.

Sample report: [docs/example-report.md](docs/example-report.md)
Verification checklist: [docs/verification.md](docs/verification.md)

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Limitations

- Tested with sklearn-style estimators; other model wrappers can work if they expose
  `.predict()`
- PyTorch `.pt`/`.pth` files are not supported yet; adding that should include a tested
  adapter from tensor inputs to model outputs
- Model deserialization is for trusted local artifacts only
- PSI needs at least 10 samples per feature to be meaningful
- Latency check measures single-sample prediction time, not batched
- No GPU-specific latency profiling (CPU only for now)
- Baselines are JSON files — no database, no dashboard

## License

MIT
