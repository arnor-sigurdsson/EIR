.PHONY: help install install-dev update-deps test test-cov test-group lint lint-fix \
	format format-check type-check docs pre-commit pre-commit-install clean build \
	check-all fix-and-check-all ci

GROUP ?= 1
SPLITS ?= 5

help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package
	uv sync

install-dev: ## Install the package with development dependencies
	uv sync --group dev

update-deps: ## Update dependencies to their latest versions
	uv sync --upgrade

test: ## Run the full test suite (slow, takes hours)
	uv run pytest tests/

test-group: ## Run one test split, e.g. make test-group GROUP=2 SPLITS=5
	uv run pytest tests/ \
		--splits $(SPLITS) \
		--group $(GROUP) \
		--splitting-algorithm=least_duration \
		--durations-path=tests/.test_durations

test-cov: ## Run tests with coverage
	uv run pytest tests/ \
		--cov-config=.coveragerc \
		--cov=eir \
		--cov-report=html \
		--cov-report=term

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check . --fix --unsafe-fixes

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check if code is formatted
	uv run ruff format --check .

type-check: ## Run mypy type checking
	uv run mypy

build-docs: ## Build the documentation
	uv run sphinx-build docs docs/_build

open-docs: ## Open the documentation in a web browser
	open docs/_build/index.html

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv build

check-all: lint format-check type-check docs ## Run all checks except the test suite

fix-and-check-all: lint-fix format type-check docs ## Fix issues, then run all checks except tests

ci: ## Run the CI checks the way GitHub Actions does (tox)
	uv run tox -e py
