"""End-to-end FastAPI test against a real Postgres + MockExtractor."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ner_pipeline import db as db_module
from ner_pipeline.api import create_app
from ner_pipeline.extractor import Mention, MockExtractor
from ner_pipeline.models import Base

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    url = os.getenv(
        "NER_TEST_DATABASE_URL",
        "postgresql+psycopg://ner:ner@localhost:5432/ner_test",
    )
    engine = create_engine(url, future=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def client(pg_engine: Engine) -> Iterator[TestClient]:
    # Wire the global session factory to the test engine.
    db_module._engine = pg_engine
    db_module._SessionLocal = sessionmaker(bind=pg_engine, autoflush=False, expire_on_commit=False)

    extractor = MockExtractor(
        default=[Mention("Alice", 0, 5, "PER", 0.99)],
    )
    app = create_app(extractor_factory=lambda: extractor)
    with TestClient(app) as c:
        yield c

    db_module.reset_engine_for_tests()


def test_ingest_endpoint_happy_path(client: TestClient) -> None:
    res = client.post(
        "/v1/ingest",
        json={"source": "test", "text": "Alice waves."},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] is True
    assert body["entity_count"] == 1
    assert len(body["mentions"]) == 1
    assert body["mentions"][0]["surface_form"] == "Alice"


def test_ingest_endpoint_idempotent(client: TestClient) -> None:
    payload = {"source": "test", "text": "Alice waves twice."}
    a = client.post("/v1/ingest", json=payload).json()
    b = client.post("/v1/ingest", json=payload).json()
    assert a["document_id"] == b["document_id"]
    assert b["created"] is False
    assert b["mentions"] == []


def test_healthz(client: TestClient) -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["db"] is True
    assert body["status"] == "ok"
