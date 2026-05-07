"""Unit tests for the ingestion pipeline."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ner_pipeline.extractor import Mention, MockExtractor
from ner_pipeline.models import Document, Entity, EntityMention
from ner_pipeline.pipeline import canonical_form, content_sha256, ingest


def test_content_sha256_stable() -> None:
    a = content_sha256("hello world")
    b = content_sha256("hello world")
    assert a == b
    assert len(a) == 64
    assert a != content_sha256("hello world!")


def test_canonical_form_collapses_whitespace_and_lowercases() -> None:
    assert canonical_form("Acme Corp") == "acme corp"
    assert canonical_form("  Acme   Corp  ") == "acme corp"
    assert canonical_form("ACME CORP") == "acme corp"


def test_ingest_writes_document_and_mentions(session: Session) -> None:
    text = "Alice met Bob at Acme."
    extractor = MockExtractor(
        default=[
            Mention("Alice", 0, 5, "PER", 0.99),
            Mention("Bob", 10, 13, "PER", 0.95),
            Mention("Acme", 17, 21, "ORG", 0.90),
        ]
    )
    res = ingest(source="test", text=text, extractor=extractor, session=session)
    assert res.created is True
    assert res.entity_count == 3

    docs = session.execute(select(Document)).scalars().all()
    assert len(docs) == 1
    assert docs[0].raw_text == text

    entities = session.execute(select(Entity)).scalars().all()
    # Two PER + one ORG = 3 distinct (canonical_form, type) pairs.
    assert len(entities) == 3
    canon_pairs = {(e.canonical_form, e.entity_type) for e in entities}
    assert canon_pairs == {("alice", "PER"), ("bob", "PER"), ("acme", "ORG")}

    mentions = session.execute(select(EntityMention)).scalars().all()
    assert len(mentions) == 3
    # Char offsets must roundtrip exactly.
    spans = {(m.char_start, m.char_end, m.surface_form) for m in mentions}
    assert spans == {(0, 5, "Alice"), (10, 13, "Bob"), (17, 21, "Acme")}


def test_ingest_is_idempotent_on_content_hash(session: Session) -> None:
    text = "Alice works at Acme."
    extractor = MockExtractor(
        default=[
            Mention("Alice", 0, 5, "PER", 0.99),
            Mention("Acme", 15, 19, "ORG", 0.90),
        ]
    )
    first = ingest(source="a", text=text, extractor=extractor, session=session)
    second = ingest(source="b", text=text, extractor=extractor, session=session)

    assert first.created is True
    assert second.created is False
    assert first.document_id == second.document_id

    # Extractor must not have been called the second time.
    assert len(extractor.calls) == 1

    docs = session.execute(select(Document)).scalars().all()
    assert len(docs) == 1
    mentions = session.execute(select(EntityMention)).scalars().all()
    assert len(mentions) == 2  # not 4


def test_ingest_dedupes_repeated_canonical_forms_within_document(
    session: Session,
) -> None:
    """Two surface mentions of "Apple" should produce ONE entity row but TWO mentions."""
    text = "Apple announced new products. Apple stock rose."
    extractor = MockExtractor(
        default=[
            Mention("Apple", 0, 5, "ORG", 0.98),
            Mention("Apple", 30, 35, "ORG", 0.97),
        ]
    )
    res = ingest(source="x", text=text, extractor=extractor, session=session)
    assert res.created is True
    assert res.entity_count == 2

    entities = session.execute(select(Entity)).scalars().all()
    assert len(entities) == 1
    assert entities[0].canonical_form == "apple"

    mentions = session.execute(select(EntityMention)).scalars().all()
    assert len(mentions) == 2
    assert all(m.entity_id == entities[0].id for m in mentions)


def test_ingest_empty_mentions_list(session: Session) -> None:
    extractor = MockExtractor(default=[])
    res = ingest(source="x", text="text with no entities", extractor=extractor, session=session)
    assert res.created is True
    assert res.entity_count == 0
    assert session.execute(select(Document)).scalars().first() is not None
    assert session.execute(select(Entity)).scalars().first() is None
    assert session.execute(select(EntityMention)).scalars().first() is None


def test_ingest_preserves_char_offsets_exactly(session: Session) -> None:
    text = "Visit Berlin in summer."
    extractor = MockExtractor(default=[Mention("Berlin", 6, 12, "LOC", 0.99)])
    res = ingest(source="t", text=text, extractor=extractor, session=session)

    mention = session.execute(select(EntityMention)).scalar_one()
    assert text[mention.char_start : mention.char_end] == "Berlin"
    assert mention.surface_form == "Berlin"
    _ = res
