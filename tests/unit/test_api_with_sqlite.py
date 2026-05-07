"""API tests using SQLite (no Postgres) and a MockExtractor."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ner_pipeline import db as db_module
from ner_pipeline.api import create_app, get_calibration
from ner_pipeline.calibration import CalibrationParams, PlattParams
from ner_pipeline.extractor import Mention, MockExtractor
from ner_pipeline.models import Base


@pytest.fixture()
def sqlite_app(tmp_path: Path) -> Iterator[tuple[TestClient, Engine]]:
    url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)

    db_module._engine = engine
    db_module._SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.99)])
    app = create_app(extractor_factory=lambda: extractor)
    with TestClient(app) as client:
        yield client, engine

    db_module.reset_engine_for_tests()


def test_ingest_endpoint(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    res = client.post("/v1/ingest", json={"source": "t", "text": "Alice waves."})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] is True
    assert body["entity_count"] == 1
    assert body["mentions"][0]["surface_form"] == "Alice"


def test_ingest_idempotent(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    payload = {"source": "t", "text": "Hello Alice."}
    a = client.post("/v1/ingest", json=payload).json()
    b = client.post("/v1/ingest", json=payload).json()
    assert a["document_id"] == b["document_id"]
    assert b["created"] is False
    assert b["mentions"] == []


def test_healthz(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["db"] is True
    assert body["status"] == "ok"


def test_request_id_header_echoed(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    res = client.get("/healthz", headers={"x-request-id": "abc-123"})
    assert res.headers.get("x-request-id") == "abc-123"


def test_request_id_generated_when_missing(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    res = client.get("/healthz")
    assert res.headers.get("x-request-id")  # something was set


def test_validation_error_envelope(sqlite_app: tuple[TestClient, Engine]) -> None:
    client, _ = sqlite_app
    res = client.post("/v1/ingest", json={"source": "", "text": ""})
    # Pydantic validation -> 422; not our envelope path, but should still return JSON.
    assert res.status_code == 422


@pytest.fixture()
def sqlite_app_calibrated(tmp_path: Path) -> Iterator[TestClient]:
    """API with a strong negative-bias calibration injected for PER."""
    url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    db_module._engine = engine
    db_module._SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.99)])
    cal = CalibrationParams(
        per_type={"PER": PlattParams(a=1.0, b=-3.0, n_train=0)},
        fallback=PlattParams(a=1.0, b=0.0, n_train=0),
    )
    app = create_app(extractor_factory=lambda: extractor)
    app.dependency_overrides[get_calibration] = lambda: cal
    with TestClient(app) as client:
        yield client
    db_module.reset_engine_for_tests()


def test_ingest_returns_calibrated_confidence_by_default(
    sqlite_app_calibrated: TestClient,
) -> None:
    res = sqlite_app_calibrated.post("/v1/ingest", json={"source": "t", "text": "Alice waves."})
    body = res.json()
    # Strong negative bias pulls calibrated confidence below the raw 0.99.
    conf = body["mentions"][0]["confidence"]
    assert 0.0 < conf < 0.95


def test_ingest_raw_confidence_flag_returns_uncalibrated(
    sqlite_app_calibrated: TestClient,
) -> None:
    res = sqlite_app_calibrated.post(
        "/v1/ingest",
        json={"source": "t", "text": "Alice raw.", "raw_confidence": True},
    )
    body = res.json()
    conf = body["mentions"][0]["confidence"]
    # Raw 0.99 returned unchanged (within Numeric(5,4) rounding).
    assert abs(conf - 0.99) < 1e-3
