SHELL := /bin/bash

ENV := ./conda_env
CONDA_RUN := conda run -p $(ENV)
PYTEST := $(CONDA_RUN) pytest
PRE_COMMIT := $(CONDA_RUN) pre-commit

.PHONY: setup setup-yaml lock maintain test qa

## setup: Create/update local Conda env and install dev tooling + hooks
setup:
	@echo "[setup] Creating/updating conda env at $(ENV) …"
	@if [ ! -d "$(ENV)" ]; then conda create -y -p $(ENV) python=3.10; fi
	@echo "[setup] Installing runtime + dev dependencies …"
	@conda install -y -p $(ENV) -c conda-forge \
		pytest numpy h5py sqlalchemy pydantic loguru rich \
		pre-commit black ruff isort mypy interrogate
	@echo "[setup] Installing git hooks via pre-commit …"
	@$(PRE_COMMIT) install
	@echo "[setup] Done. To use tools directly: conda run -p $(ENV) <tool>"

## setup-yaml: Create/update env from environment.yml and install hooks
setup-yaml:
	@echo "[setup-yaml] Creating/updating env from environment.yml at $(ENV) …"
	@if [ ! -d "$(ENV)" ]; then \
		conda env create -p $(ENV) -f environment.yml; \
	else \
		conda env update -p $(ENV) -f environment.yml; \
	fi
	@echo "[setup-yaml] Installing git hooks via pre-commit …"
	@$(PRE_COMMIT) install
	@echo "[setup-yaml] Done."

## lock: Generate conda-lock.yml for pinned environments (multi-platform)
lock:
	@echo "[lock] Generating conda-lock.yml …"
	@$(CONDA_RUN) conda-lock lock -f environment.yml -p osx-64 -p linux-64 -p win-64
	@echo "[lock] Wrote conda-lock.yml"

## maintain: Run pre-commit on all files (format, lint, type-check, docs coverage)
maintain:
	@$(PRE_COMMIT) run --all-files

## test: Run the test suite with local sources on PYTHONPATH
## Note: We avoid installing the package; tests import from the working tree.
test:
	@$(CONDA_RUN) env PYTHONPATH=. $(PYTEST) -q

## qa: Run maintain and tests
qa: maintain test
