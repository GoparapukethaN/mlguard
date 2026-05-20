PYTHON ?= python3

.PHONY: install lint test example verify ci-local

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests -q

example:
	$(PYTHON) examples/sklearn_example.py

verify:
	./scripts/verify-local.sh

ci-local: verify
