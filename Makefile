.PHONY: default setup simulate clean help zip test test-cov lint fix format ensure-venv commit update rebuild freeze test-one security complexity push

# Common variables
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := $(PYTHON) -m pytest
RUFF := .venv/bin/ruff
BLACK := .venv/bin/black
ISORT := .venv/bin/isort

BANDIT := .venv/bin/bandit
RADON := .venv/bin/radon

# Default target when running plain `make`
default: setup

setup:
	@if [ -d .venv ]; then \
		echo "⚠️  Virtualenv .venv already exists. Skipping creation."; \
	else \
		python3 -m venv .venv; \
	fi
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel
	if [ -f requirements.txt ]; then \
		. .venv/bin/activate && pip install -r requirements.txt; \
	fi
	@echo "\033[32m✅ Setup complete.\033[0m"
	@echo "Interpreter path: $$(. .venv/bin/activate && which python)"
	@echo "👉 Remember to select \033[1m.venv/bin/python\033[0m in VS Code."

simulate:
	@if [ ! -d .venv ]; then \
		echo "❌ Virtual environment not found. Please run 'make setup' first."; \
		exit 1; \
	fi
	@echo "🚀 Starting simulation..."
	. .venv/bin/activate && python -B -m runner.run_simulation

# -----------------
# Testing & Quality
# -----------------

# Run unit tests (quiet)
test: ensure-venv
	@echo "🧪 Running tests..."
	$(PYTEST) -q -vv

# Run tests with coverage (requires pytest-cov)
test-cov: ensure-venv
	@echo "🧪 Running tests with coverage..."
	$(PYTEST) -q -vv --cov=. --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html
	@echo "📊 Coverage reports: htmlcov/index.html and coverage.xml"

# Lint (Ruff)
lint: ensure-venv
	@echo "🔍 Linting with Ruff..."
	$(RUFF) check .

# Auto-fix lint (Ruff --fix)
fix: ensure-venv
	@echo "🛠  Ruff fix..."
	$(RUFF) check . --fix

# Format code (Black + isort)
format: ensure-venv
	@echo "🧹 Formatting with Black + isort..."
	$(BLACK) .
	$(ISORT) .


# Ensure virtual environment exists
ensure-venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi

clean:
	@if [ -d .venv ]; then \
		rm -rf .venv; \
	fi
	@find . -type d -name __pycache__ -exec rm -rf {} + || true
	@find . -type f -name '*.pyc' -delete || true
	@find . -type f -name '*.log' -delete || true
	@find . -type f -name '*~' -delete || true
	@echo "\033[32m🧹 Cleanup complete. Removed virtual environment, cache, logs, and temporary files.\033[0m"

zip:
	@echo "📦 Creating RoadSimulator3.zip archive..."
	@zip -r RoadSimulator3.zip . -x ".venv/*" -x "__pycache__/*" -x "*.pyc" -x "*.log" -x "RoadSimulator3.zip"
	@echo "\033[32m✅ Archive RoadSimulator3.zip created successfully.\033[0m"

commit:
	@git add .
	@git commit -m "Release 1.0"
	@git tag -a v1.0 -m "Version 1.0"
	@echo "\033[32m✅ Commit and tag v1.0 created successfully.\033[0m"

# ----------------------
# Maintenance & Releases
# ----------------------

# Update dependencies from requirements.txt
update: ensure-venv
	@echo "⬆️  Updating dependencies..."
	@if [ -f requirements.txt ]; then \
		$(PIP) install --upgrade -r requirements.txt; \
	else \
		echo "ℹ️  No requirements.txt found. Skipping."; \
	fi

# Clean everything and setup again
rebuild: clean setup
	@echo "🔄 Environment rebuilt."

# Freeze current environment to requirements.txt
freeze: ensure-venv
	@echo "📜 Freezing dependencies to requirements.txt..."
	@$(PIP) freeze > requirements.txt
	@echo "✅ requirements.txt updated."

# Run a single test (usage: make test-one FILE=tests/test_x.py::TestC::test_y)
test-one: ensure-venv
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please provide FILE=path::TestClass::test_method"; \
		exit 1; \
	fi
	$(PYTEST) -vv $(FILE)

# Security scan with bandit (installs if missing)
security: ensure-venv
	@echo "🛡  Running security scan with bandit..."
	@if [ ! -x "$(BANDIT)" ]; then $(PIP) install bandit >/dev/null 2>&1; fi
	@$(BANDIT) -r . || true

# Complexity analysis with radon (installs if missing)
complexity: ensure-venv
	@echo "📈 Analyzing code complexity with radon..."
	@if [ ! -x "$(RADON)" ]; then $(PIP) install radon >/dev/null 2>&1; fi
	@$(RADON) cc -s -a . || true

# Push current branch and tags to origin
push:
	@echo "🚀 Pushing current branch and tags to origin..."
	@git push origin HEAD --tags
	@echo "✅ Pushed."

# List available targets
help:
	@echo "Available targets:"
	@echo "  make (default) -> setup"
	@echo "  make setup      -> create venv, upgrade pip, install requirements"
	@echo "  make simulate   -> run the simulation using the virtual environment"
	@echo "  make clean      -> remove virtual environment, caches, logs, and temp files"
	@echo "  make zip        -> create a zip archive of the project excluding venv, caches, logs, and the archive itself"
	@echo "  make test       -> run unit tests"
	@echo "  make test-cov   -> run tests with coverage"
	@echo "  make lint       -> lint code using Ruff"
	@echo "  make fix        -> auto-fix lint issues using Ruff"
	@echo "  make format     -> format code with Black and isort"
	@echo "  make ensure-venv-> ensure virtual environment exists"
	@echo "  make commit     -> commit all changes with message 'Release 1.0' and tag as v1.0"
	@echo "  make update     -> update dependencies from requirements.txt"
	@echo "  make rebuild    -> clean and setup a fresh environment"
	@echo "  make freeze     -> write current env to requirements.txt"
	@echo "  make test-one   -> run a single test (FILE=...)"
	@echo "  make security   -> run bandit security scan"
	@echo "  make complexity -> run radon complexity analysis"
	@echo "  make push       -> push current branch and tags to origin"
	@echo "  make help       -> show this help"