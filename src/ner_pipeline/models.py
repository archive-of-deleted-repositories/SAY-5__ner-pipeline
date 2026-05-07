"""SQLAlchemy ORM models for the ingestion store."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime


class Base(DeclarativeBase):
    """Declarative base for all ORM classes."""


class Document(Base):
    """A single source document submitted for ingestion."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )

    mentions: Mapped[list[EntityMention]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_documents_content_sha256", "content_sha256", unique=True),)


class Entity(Base):
    """A canonical entity (deduplicated by ``(canonical_form, entity_type)``)."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_form: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(8), nullable=False)

    mentions: Mapped[list[EntityMention]] = relationship(back_populates="entity")

    __table_args__ = (
        UniqueConstraint(
            "canonical_form",
            "entity_type",
            name="uq_entities_canonical_form_type",
        ),
    )


class EntityMention(Base):
    """One occurrence of an entity inside a document."""

    __tablename__ = "entity_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="RESTRICT"),
        nullable=False,
    )
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    surface_form: Mapped[str] = mapped_column(String(512), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    document: Mapped[Document] = relationship(back_populates="mentions")
    entity: Mapped[Entity] = relationship(back_populates="mentions")

    __table_args__ = (
        Index("ix_entity_mentions_document_id", "document_id"),
        Index("ix_entity_mentions_entity_id", "entity_id"),
    )
