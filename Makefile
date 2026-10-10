.PHONY: install test lint check run api

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

check: lint test

run:
	python -m evidenceflow

api:
	python -m uvicorn evidenceflow.api.app:app --reload
