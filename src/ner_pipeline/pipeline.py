"""Ingestion pipeline: text -> mentions -> dedup -> persist."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ner_pipeline.calibration import CalibrationParams, calibrated_confidence
from ner_pipeline.extractor import Extractor, Mention
from ner_pipeline.models import Document, Entity, EntityMention


def content_sha256(text: str) -> str:
    """Stable SHA-256 over the UTF-8-encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_form(surface: str) -> str:
    """Normalise a surface form into its canonical key (case-insensitive,
    whitespace-collapsed). Kept simple on purpose; a real entity-linker would
    do far more.
    """
    return " ".join(surface.split()).strip().lower()


@dataclass(frozen=True)
class IngestResult:
    """Returned by :func:`ingest`. ``created`` is False on idempotent re-runs."""

    document_id: int
    entity_count: int
    created: bool


def _get_or_create_entity(session: Session, canonical: str, entity_type: str) -> Entity:
    stmt = select(Entity).where(
        Entity.canonical_form == canonical,
        Entity.entity_type == entity_type,
    )
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return existing
    entity = Entity(canonical_form=canonical, entity_type=entity_type)
    session.add(entity)
    session.flush()
    return entity


def ingest(
    *,
    source: str,
    text: str,
    extractor: Extractor,
    session: Session,
    calibration: CalibrationParams | None = None,
) -> IngestResult:
    """Ingest ``text`` into the database.

    Idempotent on ``content_sha256``: a second call with the same text returns
    the existing document and does **not** rewrite mentions.

    If ``calibration`` is provided, the persisted ``EntityMention.confidence``
    is the **calibrated** probability under per-entity-type Platt scaling.
    Pass ``calibration=None`` to keep the raw model score (back-compat).
    """
    digest = content_sha256(text)
    existing = session.execute(
        select(Document).where(Document.content_sha256 == digest)
    ).scalar_one_or_none()
    if existing is not None:
        count = session.execute(
            select(EntityMention).where(EntityMention.document_id == existing.id)
        ).all()
        return IngestResult(
            document_id=existing.id,
            entity_count=len(count),
            created=False,
        )

    doc = Document(source=source, raw_text=text, content_sha256=digest)
    session.add(doc)
    session.flush()

    raw_mentions: list[Mention] = extractor.extract(text)
    if calibration is not None:
        mentions = [
            replace(
                m,
                confidence=calibrated_confidence(m.confidence, m.entity_type, calibration),
            )
            for m in raw_mentions
        ]
    else:
        mentions = raw_mentions

    # Dedup mentions by (canonical, type) within the document so we issue one
    # entity row per unique pair, but every char-span gets its own mention row.
    entity_cache: dict[tuple[str, str], Entity] = {}
    for m in mentions:
        canon = canonical_form(m.surface_form)
        key = (canon, m.entity_type)
        entity = entity_cache.get(key)
        if entity is None:
            entity = _get_or_create_entity(session, canon, m.entity_type)
            entity_cache[key] = entity
        session.add(
            EntityMention(
                document_id=doc.id,
                entity_id=entity.id,
                char_start=m.char_start,
                char_end=m.char_end,
                surface_form=m.surface_form,
                confidence=Decimal(f"{m.confidence:.4f}"),
            )
        )

    session.flush()
    return IngestResult(
        document_id=doc.id,
        entity_count=len(mentions),
        created=True,
    )
