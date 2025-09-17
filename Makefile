SHELL := /bin/bash

ENV := ./conda_env
CONDA_RUN := conda run -p $(ENV)
PYTEST := $(CONDA_RUN) pytest
PRE_COMMIT := $(CONDA_RUN) pre-commit

.PHONY: setup maintain test

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

## maintain: Run pre-commit on all files (format, lint, type-check, docs coverage)
maintain:
	@$(PRE_COMMIT) run --all-files

## test: Run the test suite with local sources on PYTHONPATH
## Note: We avoid installing the package; tests import from the working tree.
test:
	@$(CONDA_RUN) env PYTHONPATH=. $(PYTEST) -q
