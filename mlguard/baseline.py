"""Save and load baseline metrics as JSON files.

Baselines live in the repo alongside the model — version controlled,
auditable, no external database needed.
"""

from __future__ import annotations

import json
from pathlib import Path


def save_baseline(metrics: dict, path: str | Path) -> None:
    """Save baseline metrics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_baseline(path: str | Path) -> dict:
    """Load baseline metrics from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No baseline found at {path}. "
            "Run `mlguard baseline` first to create one."
        )
    with open(path) as f:
        return json.load(f)
