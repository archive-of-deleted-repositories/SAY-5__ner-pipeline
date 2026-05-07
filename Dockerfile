# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install --prefix=/install \
        "torch>=2.2,<3.0" \
        "transformers>=4.40,<5.0" \
        "fastapi>=0.110,<1.0" \
        "uvicorn[standard]>=0.29,<1.0" \
        "sqlalchemy>=2.0,<3.0" \
        "alembic>=1.13,<2.0" \
        "psycopg[binary]>=3.1,<4.0" \
        "pydantic>=2.5,<3.0" \
        "pydantic-settings>=2.2,<3.0" \
        "structlog>=24.1,<25.0" \
        "prometheus-client>=0.20,<1.0"

# ---------- runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /install /usr/local
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

USER app
EXPOSE 8000

CMD ["uvicorn", "ner_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
