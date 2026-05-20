# MLGuard Case Study

MLGuard is my small release-gate utility for checking a model candidate before it reaches
deployment. The project focuses on three signals that are easy to understand during a
release review: feature drift, performance regression, and latency regression.

It is intentionally not a full monitoring platform. The goal is to create a local,
scriptable gate that can stop a bad release early and leave behind a Markdown/JSON report
for review.

## Problem

Model monitoring usually happens after a model is live. That is useful, but a release
decision also needs a pre-deployment check:

- Has the current data shifted away from the reference distribution?
- Did the candidate model lose accuracy or F1 against the baseline?
- Did p95 inference latency get worse?
- Can the gate write an artifact that another engineer can review?

I wanted the release decision to be inspectable instead of hidden inside a notebook or a
one-off script.

## What I Built

The CLI has two main paths:

- `mlguard baseline` records reference metrics for a trusted local model and dataset.
- `mlguard check` compares a candidate model/current dataset against that baseline.

The check path returns `PASS`, `WARN`, or `FAIL`, exits nonzero on failure, and writes:

- a Markdown report for human review
- a JSON report for automation, dashboards, or release records

The included composite action can run the same check in a workflow and upload both report
artifacts.

## Current Example Result

The sklearn example trains a small RandomForest on synthetic data, records a baseline,
then shifts several features to simulate drift. The expected result is a blocked release:

```text
VERDICT: FAIL
3 feature(s) with significant drift; performance regression detected
```

The latest local example reports:

| Signal | Result |
| --- | --- |
| Feature drift | 3 failed PSI checks |
| Performance | accuracy and F1 both failed |
| Latency | passed |
| Report artifacts | Markdown and JSON |

This is a demo artifact for exercising the gate. It is not a claim about a production
model.

## Design Choices

### Fail fast when the baseline is missing

The default check refuses to run without a baseline because a release gate should not
silently downgrade itself. Drift-only mode still exists, but it requires
`--allow-missing-baseline` so the weaker mode is explicit.

### Keep model loading inside a trusted boundary

MLGuard loads local joblib/pickle-style artifacts. That is acceptable for artifacts
created by the same trusted pipeline, but the README and verification docs call out that
untrusted model files should not be loaded.

### Write both human and machine reports

Markdown is easy to review during a release discussion. JSON is easier to turn into CI
annotations, dashboards, or release records. Keeping both outputs makes the tool useful
without requiring a server.

### Keep unsupported artifacts honest

The CLI and action metadata describe trusted sklearn/joblib-style artifacts. PyTorch
`.pt`/`.pth` support is listed as a limitation until there is a tested adapter.

## Verification

Local verification currently includes:

- `26` pytest tests
- Ruff checks
- the sklearn example producing Markdown and JSON reports
- JSON summary counts for failed and warned checks
- CLI/action metadata checks around the supported model-artifact boundary

Command:

```bash
PYTHON=.venv/bin/python make verify
```

The repeatable proof is tracked in [verification.md](verification.md) and
[example-report.md](example-report.md).

## What I Would Improve Next

- Add a persisted release-history store instead of standalone report files.
- Add a small dashboard for comparing release attempts.
- Add tested adapters for non-sklearn model types.
- Add richer drift checks for categorical and text features.
- Add configurable ownership metadata so a report can name the model, dataset, and release
  candidate more clearly.

