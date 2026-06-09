.PHONY: help install install-python install-dev format ruff-fix lint test test-ink test-rust test-llm-matrix test-verbose test-strict coverage check quality-gates typecheck clean clean-build clean-cache clean-pyc clean-test build version bump-patch bump-minor bump-major release release-test homebrew-sync publish dev-setup dev-check quick-patch quick-minor quick-major
# Default target
help:
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo ""
	@echo "Development:"
	@echo "  install       Install the Rust CLI binaries"
	@echo "  install-python Install the Python package for development"
	@echo "  install-dev   Install package and development dependencies"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Lint code with ruff"
	@echo "  test          Run tests"
	@echo "  test-rust     Run Rust workspace tests"
	@echo "  test-llm-matrix Run live Rust LLM provider matrix"
	@echo "  test-verbose  Run tests with verbose output"
	@echo "  coverage      Run tests with coverage report"
	@echo "  check         Run all fixes and checks (ruff --fix, format, lint, typecheck, test)"
	@echo "  quality-gates Run full Python, Rust, and UI validation"
	@echo ""
	@echo "Build and Release:"
	@echo "  clean         Clean all build artifacts"
	@echo "  build         Build and package Rust release binaries"
	@echo "  version       Show current Rust release version"
	@echo "  bump-patch    Bump patch version (0.0.X)"
	@echo "  bump-minor    Bump minor version (0.X.0)"
	@echo "  bump-major    Bump major version (X.0.0)"
	@echo "  release-test  Build release artifacts locally"
	@echo "  release       Tag and push a semver release"
	@echo "  homebrew-sync Update the kcmt-homebrew formula from release checksums"
	@echo "  publish       Alias for release"

# Variables
PACKAGE_NAME = kcmt_python
PACKAGE_DIR = legacy/kcmt-python/kcmt_python
PYTHON = python3
UV = uv
RUST_MANIFEST = rust/Cargo.toml
RUST_BIN_DIR = rust/target/release
RUST_DIST_DIR = dist
HOMEBREW_TAP_REPO ?= ../kcmt-homebrew
HOMEBREW_FORMULA = Formula/kcmt.rb
PYTEST = $(UV) run pytest
TEST_ENV = KCMT_DISABLE_KEYCHAIN=1
PYPI_TOKEN ?=
TEST_PYPI_TOKEN ?=
TWINE_USER ?= __token__
TWINE_TEST_USER ?= __token__

# Get current version
VERSION = $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' $(RUST_MANIFEST) | head -n 1)

# Installation targets
install:
	cargo install --locked --force --path rust/crates/kcmt-cli

install-python:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"
	$(UV) pip install black isort ruff pytest pytest-cov twine build

# Formatting and linting
format:
	@echo "Sorting imports with isort..."
	$(UV) run isort ./$(PACKAGE_DIR) tests
	@echo "Formatting code with black..."
	$(UV) run black ./$(PACKAGE_DIR) tests

lint:
	@echo "Linting with ruff..."
	$(UV) run ruff check ./$(PACKAGE_DIR) tests
	@echo "Checking import order with isort..."
	$(UV) run isort --check-only ./$(PACKAGE_DIR) tests
	@echo "Checking format with black..."
	$(UV) run black --check ./$(PACKAGE_DIR) tests

ruff-fix:
	@echo "Auto-fixing lint with ruff..."
	$(UV) run ruff check --fix ./$(PACKAGE_DIR) tests

# Testing
test:
	$(TEST_ENV) $(PYTEST) -q

test-ink:
	npm --prefix legacy/kcmt-python/kcmt_python/ui/ink test

test-rust:
	$(TEST_ENV) cargo test --locked --manifest-path rust/Cargo.toml --workspace --no-fail-fast

test-llm-matrix:
	KCMT_LIVE_LLM_MATRIX=1 cargo test --locked --manifest-path rust/Cargo.toml -p kcmt-cli --test live_llm_matrix -- --ignored --nocapture

test-verbose:
	$(TEST_ENV) $(PYTEST) -v

test-strict:
	$(TEST_ENV) $(PYTEST) -ra -vv -W default -W error::DeprecationWarning -W error::ResourceWarning --strict-config --strict-markers tests

coverage:
	$(TEST_ENV) $(PYTEST) --cov=kcmt_python.main --cov=kcmt_python --cov-report=html --cov-report=term

# Type checking
typecheck:
	@echo "Type checking with mypy..."
	$(UV) run mypy ./$(PACKAGE_DIR)

# All checks
check: ruff-fix format lint typecheck test-strict test-rust test-ink
	@echo "All checks passed!"

quality-gates: check coverage
	@echo "Quality gates passed!"

# Cleaning
clean: clean-build clean-cache clean-pyc clean-test
	@echo "Cleaned all artifacts"

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf rust/target/
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
	@current_version="$(VERSION)"; \
	new_version=$$(printf '%s\n' "$$current_version" | awk -F. '{printf "%s.%s.%d", $$1, $$2, $$3 + 1}'); \
	perl -0pi -e 's/\[workspace\.package\]\nversion = "[^"]*"/[workspace.package]\nversion = "'"$$new_version"'"/' $(RUST_MANIFEST); \
	echo "Version bumped from $$current_version to $$new_version"

bump-minor:
	@echo "Bumping minor version..."
	@current_version="$(VERSION)"; \
	new_version=$$(printf '%s\n' "$$current_version" | awk -F. '{printf "%s.%d.0", $$1, $$2 + 1}'); \
	perl -0pi -e 's/\[workspace\.package\]\nversion = "[^"]*"/[workspace.package]\nversion = "'"$$new_version"'"/' $(RUST_MANIFEST); \
	echo "Version bumped from $$current_version to $$new_version"

bump-major:
	@echo "Bumping major version..."
	@current_version="$(VERSION)"; \
	new_version=$$(printf '%s\n' "$$current_version" | awk -F. '{printf "%d.0.0", $$1 + 1}'); \
	perl -0pi -e 's/\[workspace\.package\]\nversion = "[^"]*"/[workspace.package]\nversion = "'"$$new_version"'"/' $(RUST_MANIFEST); \
	echo "Version bumped from $$current_version to $$new_version"

# Build
build: clean
	@echo "Building Rust release binaries..."
	cargo build --locked --release --manifest-path $(RUST_MANIFEST) -p kcmt-cli
	@mkdir -p $(RUST_DIST_DIR)
	@archive_dir="$(RUST_DIST_DIR)/kcmt-$(VERSION)-$$(uname -s | tr '[:upper:]' '[:lower:]')-$$(uname -m)"; \
	mkdir -p "$$archive_dir"; \
	cp $(RUST_BIN_DIR)/kcmt $(RUST_BIN_DIR)/commit $(RUST_BIN_DIR)/kc "$$archive_dir"/; \
	[ -f README.md ] && cp README.md "$$archive_dir"/ || true; \
	[ -f LICENSE ] && cp LICENSE "$$archive_dir"/ || true; \
	tar -C $(RUST_DIST_DIR) -czf "$(RUST_DIST_DIR)/kcmt-$(VERSION)-$$(uname -s | tr '[:upper:]' '[:lower:]')-$$(uname -m).tar.gz" "kcmt-$(VERSION)-$$(uname -s | tr '[:upper:]' '[:lower:]')-$$(uname -m)"; \
	git archive --format=tar.gz --prefix="kcmt-$(VERSION)/" HEAD -o "$(RUST_DIST_DIR)/kcmt-$(VERSION)-source.tar.gz"; \
	sha256sum "$(RUST_DIST_DIR)/kcmt-$(VERSION)-$$(uname -s | tr '[:upper:]' '[:lower:]')-$$(uname -m).tar.gz" "$(RUST_DIST_DIR)/kcmt-$(VERSION)-source.tar.gz" > "$(RUST_DIST_DIR)/SHA256SUMS"; \
	rm -rf "$$archive_dir"

# Release
release-test: build
	@echo "Built release archive: $(RUST_DIST_DIR)/kcmt-$(VERSION)-$$(uname -s | tr '[:upper:]' '[:lower:]')-$$(uname -m).tar.gz"

release: build
	@echo "Tagging release v$(VERSION)..."
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@git push origin v$(VERSION)
	@echo "Pushed release tag v$(VERSION)"
	@$(MAKE) homebrew-sync

homebrew-sync:
	@test -f "$(RUST_DIST_DIR)/SHA256SUMS" || (echo "Run make build or make release-test first." && exit 1)
	@test -d "$(HOMEBREW_TAP_REPO)/.git" || (echo "Set HOMEBREW_TAP_REPO to a cloned kcmt-homebrew repo." && exit 1)
	@echo "Syncing kcmt-homebrew formula from $(RUST_DIST_DIR)/SHA256SUMS..."
	$(PYTHON) scripts/sync_homebrew_formula.py --tap-repo "$(HOMEBREW_TAP_REPO)" --version "$(VERSION)" --sums-file "$(RUST_DIST_DIR)/SHA256SUMS"
	@if git -C "$(HOMEBREW_TAP_REPO)" diff --quiet -- "$(HOMEBREW_FORMULA)"; then \
		echo "kcmt-homebrew formula already up to date."; \
	else \
		git -C "$(HOMEBREW_TAP_REPO)" add "$(HOMEBREW_FORMULA)"; \
		git -C "$(HOMEBREW_TAP_REPO)" commit -m "build(homebrew): update kcmt formula for v$(VERSION)"; \
		git -C "$(HOMEBREW_TAP_REPO)" push origin HEAD; \
	fi

publish: release

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment setup complete!"

dev-check: format lint test
	@echo "Development checks complete - ready to commit!"

# Quick release workflow
quick-patch: bump-patch build release
quick-minor: bump-minor build release
quick-major: bump-major build release
