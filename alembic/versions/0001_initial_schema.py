"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("content_sha256", name="uq_documents_content_sha256"),
    )
    op.create_index(
        "ix_documents_content_sha256",
        "documents",
        ["content_sha256"],
        unique=True,
    )

    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("canonical_form", sa.String(length=512), nullable=False),
        sa.Column("entity_type", sa.String(length=8), nullable=False),
        sa.UniqueConstraint(
            "canonical_form",
            "entity_type",
            name="uq_entities_canonical_form_type",
        ),
    )

    op.create_table(
        "entity_mentions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("surface_form", sa.String(length=512), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["entities.id"], ondelete="RESTRICT"
        ),
    )
    op.create_index(
        "ix_entity_mentions_document_id",
        "entity_mentions",
        ["document_id"],
    )
    op.create_index(
        "ix_entity_mentions_entity_id",
        "entity_mentions",
        ["entity_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_entity_mentions_entity_id", table_name="entity_mentions")
    op.drop_index("ix_entity_mentions_document_id", table_name="entity_mentions")
    op.drop_table("entity_mentions")
    op.drop_table("entities")
    op.drop_index("ix_documents_content_sha256", table_name="documents")
    op.drop_table("documents")
