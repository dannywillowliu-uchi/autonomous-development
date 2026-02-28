# Usage: make setup && make traces && make run
.PHONY: setup test traces dashboard run clean

setup:
	uv venv && uv sync --extra dev --extra tracing

test:
	.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/

traces:
	docker compose up -d jaeger

dashboard:
	.venv/bin/python -m mission_control.cli live --port 8080

run:
	.venv/bin/python -m mission_control.cli mission --config mission-control.toml --workers 2

clean:
	docker compose down
