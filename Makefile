.PHONY: setup test lint
setup:
	uv sync --locked
test:
	uv run pytest -m "not integration"
lint:
	uv run ruff check
	uv run ruff format --check
