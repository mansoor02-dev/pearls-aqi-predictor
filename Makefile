.PHONY: install sync test lint format clean run-feature run-train run-app

# Setup
install:
	uv sync

sync:
	uv sync --no-dev

# Development
test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint:
	uv run flake8 src/ tests/ app/
	uv run mypy src/

format:
	uv run black src/ tests/ app/
	uv run isort src/ tests/ app/

format-check:
	uv run black --check src/ tests/ app/
	uv run isort --check-only src/ tests/ app/

# Pipelines
run-feature:
	uv run python scripts/run_feature_pipeline.py

run-train:
	uv run python scripts/run_training_pipeline.py

run-backfill:
	uv run python scripts/run_backfill.py

# App
run-app:
	uv run streamlit run app/dashboard.py

run-api:
	uv run uvicorn app.api:app --reload --port 8000

# Cleanup
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete