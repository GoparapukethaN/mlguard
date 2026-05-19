#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  python_cmd=".venv/bin/python"
else
  python_cmd="${PYTHON:-python3}"
fi

"$python_cmd" -m ruff check .
"$python_cmd" -m pytest tests -q
"$python_cmd" examples/sklearn_example.py >/tmp/mlguard-example.txt

grep -q "VERDICT: FAIL" /tmp/mlguard-example.txt
grep -q "performance regression detected" /tmp/mlguard-example.txt
test -f /tmp/mlguard_report.md

echo "local verification passed"
