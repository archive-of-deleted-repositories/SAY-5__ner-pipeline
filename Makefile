.PHONY: help install dev migrate seed test lint typecheck eval eval-smoke serve up down build clean

PY ?= python
PIP ?= pip
DC ?= docker compose

help:
	@echo "targets:"
	@echo "  install     - install runtime + dev deps via pip (uses pyproject.toml)"
	@echo "  dev         - install dev environment (alias for install)"
	@echo "  up          - bring up postgres via docker compose"
	@echo "  down        - tear down docker compose"
	@echo "  migrate     - run alembic upgrade head"
	@echo "  seed        - seed 100 synthetic documents (use SEED_NO_EXTRACT=1 to skip the model)"
	@echo "  test        - run unit tests"
	@echo "  test-int    - run integration tests (requires postgres)"
	@echo "  lint        - ruff + black --check"
	@echo "  format      - ruff --fix + black"
	@echo "  typecheck   - mypy strict on src/ner_pipeline"
	@echo "  eval        - run the full CoNLL-2003 test split eval"
	@echo "  eval-smoke  - run the 100-sentence smoke eval"
	@echo "  serve       - start the FastAPI app on :8000"
	@echo "  build       - docker build the image"
	@echo "  clean       - remove caches and build artefacts"

install dev:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]" || $(PIP) install \
		torch transformers datasets fastapi "uvicorn[standard]" \
		"sqlalchemy>=2" alembic "psycopg[binary]" pydantic pydantic-settings \
		structlog prometheus-client faker seqeval \
		pytest pytest-asyncio pytest-cov factory-boy ruff black mypy httpx \
		"testcontainers[postgres]"

up:
	$(DC) up -d postgres
	@echo "postgres should be reachable on :5432"

down:
	$(DC) down -v

migrate:
	alembic upgrade head

seed:
ifeq ($(SEED_NO_EXTRACT),1)
	$(PY) scripts/seed.py --count 100 --no-extract
else
	$(PY) scripts/seed.py --count 100
endif

test:
	pytest tests/unit -v --cov=src/ner_pipeline --cov-report=term-missing --cov-fail-under=80

test-int:
	RUN_INTEGRATION=1 pytest tests/integration -v

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src/ner_pipeline

eval:
	$(PY) scripts/eval_conll.py --output eval/baselines/conll_test_distilbert.json

eval-smoke:
	RUN_EVAL_SMOKE=1 pytest tests/eval -v -s

serve:
	uvicorn ner_pipeline.api:app --host 0.0.0.0 --port 8000 --reload

build:
	docker build -t ner-pipeline:local .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
