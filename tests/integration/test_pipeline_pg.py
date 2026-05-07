"""Integration test: full ingest round-trip against a real Postgres."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ner_pipeline.extractor import Mention, MockExtractor
from ner_pipeline.models import Base, Document, Entity, EntityMention
from ner_pipeline.pipeline import ingest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    """Use a Postgres URL provided by CI / local docker-compose."""
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
def pg_session(pg_engine: Engine) -> Iterator[Session]:
    sm = sessionmaker(bind=pg_engine, autoflush=False, expire_on_commit=False)
    session = sm()
    try:
        yield session
        session.rollback()  # keep tables clean between tests
        # Truncate tables for isolation.
        session.execute(EntityMention.__table__.delete())
        session.execute(Entity.__table__.delete())
        session.execute(Document.__table__.delete())
        session.commit()
    finally:
        session.close()


def test_ingest_round_trip(pg_session: Session) -> None:
    extractor = MockExtractor(
        default=[
            Mention("Alice", 0, 5, "PER", 0.99),
            Mention("Acme", 15, 19, "ORG", 0.90),
        ]
    )
    res = ingest(
        source="pg-test",
        text="Alice works at Acme.",
        extractor=extractor,
        session=pg_session,
    )
    pg_session.commit()
    assert res.created is True
    assert res.entity_count == 2

    docs = pg_session.execute(select(Document)).scalars().all()
    assert len(docs) == 1


def test_idempotency_under_postgres(pg_session: Session) -> None:
    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.99)])
    text = "Alice waved hello."
    a = ingest(source="x", text=text, extractor=extractor, session=pg_session)
    pg_session.commit()
    b = ingest(source="y", text=text, extractor=extractor, session=pg_session)
    pg_session.commit()
    assert a.document_id == b.document_id
    assert b.created is False
