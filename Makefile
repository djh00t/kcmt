.PHONY: help install install-dev format ruff-fix lint test test-verbose test-strict coverage check typecheck clean clean-build clean-cache clean-pyc clean-test build version bump-patch bump-minor bump-major release release-test dev-setup dev-check quick-patch quick-minor quick-major
# Default target
help:
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo ""
	@echo "Development:"
	@echo "  install       Install package for development"
	@echo "  install-dev   Install package and development dependencies"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Lint code with ruff"
	@echo "  test          Run tests"
	@echo "  test-verbose  Run tests with verbose output"
	@echo "  coverage      Run tests with coverage report"
	@echo "  check         Run all fixes and checks (ruff --fix, format, lint, typecheck, test)"
	@echo ""
	@echo "Build and Release:"
	@echo "  clean         Clean all build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  version       Show current version"
	@echo "  bump-patch    Bump patch version (0.0.X)"
	@echo "  bump-minor    Bump minor version (0.X.0)"
	@echo "  bump-major    Bump major version (X.0.0)"
	@echo "  release-test  Upload to TestPyPI"
	@echo "  release       Upload to PyPI"

# Variables
PACKAGE_NAME = kcmt
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest

# Get current version
VERSION := $(shell python -c "import kcmt; print(kcmt.__version__)")

# Installation targets
install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"
	$(UV) pip install black isort ruff pytest pytest-cov twine build

# Formatting and linting
format:
	@echo "Sorting imports with isort..."
	isort ./$(PACKAGE_NAME) tests
	@echo "Formatting code with black..."
	black ./$(PACKAGE_NAME) tests

lint:
	@echo "Linting with ruff..."
	ruff check ./$(PACKAGE_NAME) tests
	@echo "Checking import order with isort..."
	isort --check-only ./$(PACKAGE_NAME) tests
	@echo "Checking format with black..."
	black --check ./$(PACKAGE_NAME) tests

ruff-fix:
	@echo "Auto-fixing lint with ruff..."
	ruff check --fix ./$(PACKAGE_NAME) tests

# Testing
test:
	$(PYTEST) -q

test-verbose:
	$(PYTEST) -v

test-strict:
	$(PYTEST) -ra -vv -W default -W error::DeprecationWarning -W error::ResourceWarning --strict-config --strict-markers tests

coverage:
	$(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

# Type checking
typecheck:
	@echo "Type checking with mypy..."
	$(UV) run mypy ./$(PACKAGE_NAME)

# All checks
check: ruff-fix format lint typecheck test-strict
	@echo "All checks passed!"

# Cleaning
clean: clean-build clean-cache clean-pyc clean-test
	@echo "Cleaned all artifacts"

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-cache:
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .*_cache/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

# Version management
version:
	@echo "Current version: $(VERSION)"

bump-patch:
	@echo "Bumping patch version..."
	@current_version=$$(python -c "import kcmt; print(kcmt.__version__)"); \
	new_version=$$(echo $$current_version | awk -F. '{print $$1"."$$2"."$$3+1}'); \
	sed -i '' "s/__version__ = \"$$current_version\"/__version__ = \"$$new_version\"/" $(PACKAGE_NAME)/__init__.py; \
	echo "Version bumped from $$current_version to $$new_version"

bump-minor:
	@echo "Bumping minor version..."
	@current_version=$$(python -c "import kcmt; print(kcmt.__version__)"); \
	new_version=$$(echo $$current_version | awk -F. '{print $$1"."$$2+1".0"}'); \
	sed -i '' "s/__version__ = \"$$current_version\"/__version__ = \"$$new_version\"/" $(PACKAGE_NAME)/__init__.py; \
	echo "Version bumped from $$current_version to $$new_version"

bump-major:
	@echo "Bumping major version..."
	@current_version=$$(python -c "import kcmt; print(kcmt.__version__)"); \
	new_version=$$(echo $$current_version | awk -F. '{print $$1+1".0.0"}'); \
	sed -i '' "s/__version__ = \"$$current_version\"/__version__ = \"$$new_version\"/" $(PACKAGE_NAME)/__init__.py; \
	echo "Version bumped from $$current_version to $$new_version"

# Build
build: clean
	@echo "Building distribution packages..."
	$(PYTHON) -m build

# Release
release-test: build
	@echo "Uploading to TestPyPI..."
	@echo "Make sure you have TWINE_USERNAME and TWINE_PASSWORD set for TestPyPI"
	PYPI_USER_AGENT="$(TEST_PYPI_USER_AGENT)" twine upload --repository testpypi dist/*

release: build
	@echo "Uploading to PyPI..."
	@echo "Make sure you have TWINE_USERNAME and TWINE_PASSWORD set for PyPI"
	@read -p "Are you sure you want to release version $(VERSION) to PyPI? (y/N) " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		PYPI_USER_AGENT="$(PYPI_USER_AGENT)" twine upload dist/*; \
		echo "Released version $(VERSION) to PyPI"; \
	else \
		echo "Release cancelled"; \
	fi

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment setup complete!"

dev-check: format lint test
	@echo "Development checks complete - ready to commit!"

# Quick release workflow
quick-patch: bump-patch build release
quick-minor: bump-minor build release
quick-major: bump-major build release
