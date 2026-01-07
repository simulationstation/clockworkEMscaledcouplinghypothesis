.PHONY: all install install-dev install-all fetch run report clean clean-all lint format typecheck test test-fast test-slow ci help np-fast np-full np-report np-all doctor smoke

# Default target
all: help

# Python virtual environment
PYTHON ?= python3
VENV ?= .venv
PIP = $(VENV)/bin/pip
RANK1 = $(VENV)/bin/rank1

# Installation targets
install: $(VENV)
	$(PIP) install -e .

install-dev: $(VENV)
	$(PIP) install -e ".[dev]"

install-all: $(VENV)
	$(PIP) install -e ".[all]"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Data fetching
fetch: install
	$(RANK1) fetch all

fetch-higgs: install
	$(RANK1) fetch higgs

fetch-elastic: install
	$(RANK1) fetch elastic

fetch-diffractive: install
	$(RANK1) fetch diffractive

# Analysis runs
run: install
	$(RANK1) run all

run-higgs: install
	$(RANK1) run higgs

run-elastic: install
	$(RANK1) run elastic

run-diffractive: install
	$(RANK1) run diffractive

# Report generation
report: install
	$(RANK1) report all

# Cleaning
clean: install
	$(RANK1) clean

clean-all:
	rm -rf data/ outputs/
	rm -rf $(VENV)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Code quality
lint: $(VENV)
	$(VENV)/bin/ruff check src/ tests/

format: $(VENV)
	$(VENV)/bin/black src/ tests/
	$(VENV)/bin/ruff check --fix src/ tests/

typecheck: $(VENV)
	$(VENV)/bin/mypy src/

# Testing
test: install-dev
	$(VENV)/bin/pytest tests/ -v

test-fast: install-dev
	$(VENV)/bin/pytest tests/ -v -m "not slow and not network"

test-slow: install-dev
	$(VENV)/bin/pytest tests/ -v -m "slow"

test-network: install-dev
	$(VENV)/bin/pytest tests/ -v -m "network"

# CI target (runs all checks)
ci: lint typecheck test-fast

# Full reproducibility run
reproduce: install-all fetch run report
	@echo "Full analysis complete. See outputs/REPORT.md"

# NP (New Physics Sensitive) mode
np-fast: install
	$(RANK1) np run all --fast

np-full: install
	$(RANK1) np run all --full

np-report: install
	$(RANK1) np report all

np-all: install-all fetch np-full np-report
	@echo "NP analysis complete. See outputs/NP_REPORT.md"

np-higgs: install
	$(RANK1) np higgs

np-elastic: install
	$(RANK1) np elastic

np-diffractive: install
	$(RANK1) np diffractive

# Doctor / Smoke Test
doctor: install
	$(RANK1) doctor

smoke: install
	$(RANK1) smoke

doctor-full: install
	$(RANK1) doctor --no-skip-large --full

# Help
help:
	@echo "Rank-1 Factorization Analysis Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package in development mode"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make install-all   Install with all optional dependencies (incl. PDF extraction)"
	@echo ""
	@echo "Data:"
	@echo "  make fetch         Download all datasets"
	@echo "  make fetch-higgs   Download Higgs ATLAS data only"
	@echo "  make fetch-elastic Download elastic scattering data only"
	@echo "  make fetch-diffractive  Download diffractive DIS data only"
	@echo ""
	@echo "Analysis:"
	@echo "  make run           Run all analyses"
	@echo "  make run-higgs     Run Higgs rank test only"
	@echo "  make run-elastic   Run elastic scattering rank test only"
	@echo "  make run-diffractive  Run diffractive DIS rank test only"
	@echo "  make report        Generate markdown report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove processed data and outputs"
	@echo "  make clean-all     Remove everything including venv and caches"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with black and ruff"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run fast tests only (no network/slow)"
	@echo "  make ci            Run all CI checks"
	@echo ""
	@echo "NP (New Physics Sensitive) Mode:"
	@echo "  make np-fast       Run NP analysis (fast mode) on all datasets"
	@echo "  make np-full       Run NP analysis (full mode) on all datasets"
	@echo "  make np-report     Generate NP analysis report"
	@echo "  make np-all        Full NP pipeline: fetch, np-full, np-report"
	@echo "  make np-higgs      Run NP analysis on Higgs data only"
	@echo "  make np-elastic    Run NP analysis on elastic data only"
	@echo "  make np-diffractive Run NP analysis on diffractive data only"
	@echo ""
	@echo "Doctor / Smoke Test:"
	@echo "  make doctor        Run smoke test with auto-remediation"
	@echo "  make smoke         Alias for doctor"
	@echo "  make doctor-full   Full doctor with large downloads"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make reproduce     Run entire baseline pipeline"
	@echo "  make np-all        Run entire NP analysis pipeline"
	@echo "  make doctor        Quick smoke test to verify everything works"
