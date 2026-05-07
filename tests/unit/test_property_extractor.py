"""Hypothesis property tests for the extractor + pipeline.

These tests verify that the MockExtractor and pipeline survive arbitrary
text inputs (unicode, mixed scripts, very long, empty, whitespace-only)
and always return well-formed mentions or commit to the database without
crashing.

A fresh in-memory SQLite engine is built per hypothesis example so that
state does not leak between draws (hypothesis reuses the same function-scoped
fixture across examples by default).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from ner_pipeline.extractor import ENTITY_TYPES, EntityType, Mention, MockExtractor
from ner_pipeline.models import Base, Document, Entity, EntityMention
from ner_pipeline.pipeline import canonical_form, content_sha256, ingest


@contextmanager
def _fresh_session() -> Iterator[Session]:
    """Yield a session backed by a brand-new in-memory SQLite engine.

    Built per call so that hypothesis examples do not share state.
    """
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    sm = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = sm()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# Constrain to non-empty surface_form (Mention.__post_init__ rejects empty)
# and to unicode codepoints SQLite + Postgres can both store. We exclude
# surrogates (\ud800-\udfff) which are not legal UTF-8.
_safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),  # surrogates
        min_codepoint=0x20,
        max_codepoint=0x024F,  # Latin Extended-B
    ),
    min_size=0,
    max_size=2000,
)

_unicode_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=400,
)

_entity_type_strategy = st.sampled_from(ENTITY_TYPES)


# ---------------------------------------------------------------------------
# Property 1: random text → MockExtractor + pipeline never crash
# ---------------------------------------------------------------------------


@given(text=_safe_text)
@settings(max_examples=80, deadline=None)
def test_mock_extractor_never_crashes_on_arbitrary_text(text: str) -> None:
    extractor = MockExtractor(default=[])
    out = extractor.extract(text)
    assert isinstance(out, list)
    assert all(isinstance(m, Mention) for m in out)


@given(text=_unicode_text)
@settings(max_examples=60, deadline=None)
def test_pipeline_handles_unicode_and_mixed_scripts(text: str) -> None:
    """Pipeline must accept any unicode (or empty) text without crashing.

    The extractor returns no mentions; we just verify the document write path.
    """
    with _fresh_session() as session:
        extractor = MockExtractor(default=[])
        result = ingest(source="t", text=text, extractor=extractor, session=session)
        assert result.created is True
        assert result.entity_count == 0
        assert result.document_id > 0


# ---------------------------------------------------------------------------
# Property 2: dedup at entities table is correct under repeated insertion
#             (race-condition-equivalent: many sequential ingest calls
#             each try to insert the same canonical (form, type) pair).
# ---------------------------------------------------------------------------


@given(
    canonical=st.text(
        alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x7A),
        min_size=1,
        max_size=20,
    ),
    entity_type=_entity_type_strategy,
    repeats=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=40, deadline=None)
def test_entity_dedup_under_repeated_insertion(
    canonical: str, entity_type: EntityType, repeats: int
) -> None:
    """Repeated ingestion of the same (canonical_form, entity_type) yields one Entity row.

    This is the race-condition-equivalent: each call performs the same
    get-or-create inside its own transaction-equivalent unit. The dedup is
    enforced by the unique constraint + the canonicalisation function.
    """
    canon = canonical_form(canonical)
    if not canon:
        return  # canonical_form may collapse a whitespace-only string to empty

    with _fresh_session() as session:
        for i in range(repeats):
            # Each iteration uses a fresh document text so we exercise the
            # entity dedup path, not the document-idempotency short-circuit.
            text = f"doc-{i}-{canonical}"
            extractor = MockExtractor(
                default=[Mention(canonical, 0, len(canonical), entity_type, 0.9)]
            )
            ingest(source=f"src-{i}", text=text, extractor=extractor, session=session)

        entities = (
            session.execute(
                select(Entity).where(
                    Entity.canonical_form == canon,
                    Entity.entity_type == entity_type,
                )
            )
            .scalars()
            .all()
        )
        # Critical invariant: exactly one entity row per (canonical, type), no matter
        # how many times we re-ingested.
        assert len(entities) == 1


# ---------------------------------------------------------------------------
# Property 3: char-offset preservation
#             random text with random known-entity spans → returned
#             (char_start, char_end) slice equals the gold span text.
# ---------------------------------------------------------------------------


@st.composite
def text_with_span(draw: st.DrawFn) -> tuple[str, int, int, str]:
    """Draw a (text, start, end, surface) tuple where text[start:end] == surface."""
    prefix = draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=0x20, max_codepoint=0x7E, blacklist_categories=("Cs",)
            ),
            min_size=0,
            max_size=30,
        )
    )
    surface = draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=0x41, max_codepoint=0x7A, blacklist_categories=("Cs",)
            ),
            min_size=1,
            max_size=20,
        )
    )
    suffix = draw(
        st.text(
            alphabet=st.characters(
                min_codepoint=0x20, max_codepoint=0x7E, blacklist_categories=("Cs",)
            ),
            min_size=0,
            max_size=30,
        )
    )
    text = prefix + surface + suffix
    start = len(prefix)
    end = start + len(surface)
    return text, start, end, surface


@given(payload=text_with_span(), entity_type=_entity_type_strategy)
@settings(max_examples=80, deadline=None)
def test_char_offsets_roundtrip_through_pipeline(
    payload: tuple[str, int, int, str], entity_type: EntityType
) -> None:
    text, start, end, surface = payload
    with _fresh_session() as session:
        extractor = MockExtractor(default=[Mention(surface, start, end, entity_type, 0.9)])
        result = ingest(source="t", text=text, extractor=extractor, session=session)
        assert result.entity_count == 1

        mention = session.execute(
            select(EntityMention).where(EntityMention.document_id == result.document_id)
        ).scalar_one()
        # The char-span on the persisted row, sliced from the original text,
        # must equal the gold surface form.
        assert text[mention.char_start : mention.char_end] == surface
        assert mention.surface_form == surface


# ---------------------------------------------------------------------------
# Property 4: malformed inputs (whitespace-only, very long, mixed scripts)
# ---------------------------------------------------------------------------


@given(
    whitespace=st.text(
        alphabet=st.sampled_from([" ", "\t", "\n", "\r", " ", " "]),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=20, deadline=None)
def test_pipeline_accepts_whitespace_only_text(whitespace: str) -> None:
    with _fresh_session() as session:
        extractor = MockExtractor(default=[])
        result = ingest(source="ws", text=whitespace, extractor=extractor, session=session)
        assert result.created is True
        assert result.entity_count == 0
        docs = session.execute(select(Document)).scalars().all()
        assert len(docs) == 1


def test_pipeline_handles_very_long_text() -> None:
    """A very long input must not crash the pipeline."""
    with _fresh_session() as session:
        text = "Acme " * 5000  # 25k chars
        extractor = MockExtractor(default=[Mention("Acme", 0, 4, "ORG", 0.9)])
        result = ingest(source="long", text=text, extractor=extractor, session=session)
        assert result.created is True
        assert result.entity_count == 1


def test_pipeline_mixed_scripts_offsets() -> None:
    """Char offsets are codepoint-based, so mixed scripts must roundtrip."""
    with _fresh_session() as session:
        text = "Hello 北京 Berlin"
        start = text.index("Berlin")
        end = start + len("Berlin")
        extractor = MockExtractor(default=[Mention("Berlin", start, end, "LOC", 0.95)])
        result = ingest(source="mix", text=text, extractor=extractor, session=session)
        assert result.entity_count == 1

        mention = session.execute(select(EntityMention)).scalar_one()
        assert text[mention.char_start : mention.char_end] == "Berlin"


# ---------------------------------------------------------------------------
# Property 5: content_sha256 stability across encodings
# ---------------------------------------------------------------------------


@given(text=_unicode_text)
@settings(max_examples=60, deadline=None)
def test_content_sha256_is_deterministic(text: str) -> None:
    a = content_sha256(text)
    b = content_sha256(text)
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


@given(canonical=st.text(min_size=0, max_size=50))
@settings(max_examples=60, deadline=None)
def test_canonical_form_is_idempotent(canonical: str) -> None:
    once = canonical_form(canonical)
    twice = canonical_form(once)
    assert once == twice
